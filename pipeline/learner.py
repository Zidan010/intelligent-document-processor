"""
learner.py — Improvement from Operator Edits Module

Responsibilities:
  1. Load before/after edit pairs from sample_edits.json
  2. Use Groq to analyse what changed and WHY (not just diff)
  3. Distil reusable style patterns into a StyleGuide
  4. Persist the StyleGuide to disk (style_guide.json)
  5. Expose rules for injection into the generator
  6. Produce a measurable quality comparison: draft WITHOUT rules vs WITH rules

The improvement is demonstrated by:
  - Generating a draft_type that was NOT in the training edits (document_checklist
    or action_item_extract), first without style rules, then with.
  - Scoring both versions on a rubric aligned to the key_edits patterns.
  - Saving a side-by-side comparison report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groq import Groq

from config import DATA_DIR, GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, OUTPUTS_DIR


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class StylePattern:
    """A single learned pattern extracted from operator edits."""
    name:        str   # short label, e.g. "Use labeled sections"
    description: str   # full instruction for the LLM
    source_edit: str   # which edit pair this came from


@dataclass
class StyleGuide:
    """Collection of learned patterns."""
    patterns: list[StylePattern] = field(default_factory=list)

    def as_rules(self) -> list[str]:
        """Return a flat list of description strings for prompt injection."""
        return [p.description for p in self.patterns]

    def save(self, path: Path) -> None:
        payload = [
            {"name": p.name, "description": p.description, "source_edit": p.source_edit}
            for p in self.patterns
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  [LEARNER] Style guide saved → {path} ({len(self.patterns)} patterns)")

    @classmethod
    def load(cls, path: Path) -> StyleGuide:
        raw = json.loads(path.read_text(encoding="utf-8"))
        patterns = [
            StylePattern(r["name"], r["description"], r.get("source_edit", ""))
            for r in raw
        ]
        return cls(patterns=patterns)


# ── Prompts ───────────────────────────────────────────────────────────────────

_PATTERN_EXTRACTION_PROMPT = """
You are an expert at analysing how legal document drafts are improved by operators.

You will be given:
1. A SYSTEM DRAFT (what an AI generated)
2. AN OPERATOR-EDITED VERSION (what a lawyer/processor corrected it to)
3. KEY EDITS (a list of what changed and why, provided by the operator)

Your task: extract REUSABLE STYLE PATTERNS — general rules that can be applied
to ANY draft of this type (or any draft type) to make it better.

For each pattern:
- Give it a short name (3-6 words)
- Write it as a clear instruction the AI should follow
- Make it concrete and actionable, NOT vague

Return ONLY valid JSON as a list:
[
  {
    "name": "<short pattern name>",
    "description": "<concrete instruction for the AI to follow>"
  },
  ...
]

Aim for 6-10 high-quality patterns. Avoid redundancy.

---
SYSTEM DRAFT:
{system_draft}

OPERATOR-EDITED VERSION:
{operator_version}

KEY EDITS PROVIDED BY OPERATOR:
{key_edits}
"""

_SCORING_PROMPT = """
You are evaluating the quality of a legal case management draft document.

Score the draft on each of the following criteria (0-3 each):

1. SECTION STRUCTURE — Are there clearly labeled sections (e.g. LIENS, TAX STATUS, ACTION ITEMS)?
2. INSTRUMENT NUMBERS — Are recording instrument numbers included for all liens and encumbrances?
3. ACTION FLAGS — Are items requiring attorney attention explicitly flagged (ACTION REQUIRED)?
4. PRIORITIZATION — Are action items ranked by urgency (URGENT, HIGH, NORMAL)?
5. CROSS-DOC SYNTHESIS — Does the draft connect information from multiple source documents?
6. COMPLETENESS — Are all key facts from source documents included (contacts, dates, amounts)?
7. REVIEWER NOTES — Is there a section with actionable attorney guidance?
8. CITATION QUALITY — Are claims attributed to source documents?

Scoring guide: 0=absent, 1=partial/weak, 2=good, 3=excellent

Return ONLY valid JSON:
{{
  "scores": {{
    "section_structure": <0-3>,
    "instrument_numbers": <0-3>,
    "action_flags": <0-3>,
    "prioritization": <0-3>,
    "cross_doc_synthesis": <0-3>,
    "completeness": <0-3>,
    "reviewer_notes": <0-3>,
    "citation_quality": <0-3>
  }},
  "total": <sum>,
  "summary": "<2-3 sentence assessment>"
}}

DRAFT TO EVALUATE:
{draft_text}
"""


# ── Core functions ─────────────────────────────────────────────────────────────

def extract_patterns_from_edit(
    client: Groq,
    edit: dict[str, Any],
) -> list[StylePattern]:
    """
    Analyse one before/after edit pair and extract reusable patterns.
    """
    prompt = _PATTERN_EXTRACTION_PROMPT.format(
        system_draft=edit["system_draft"],
        operator_version=edit["operator_edited_version"],
        key_edits=json.dumps(edit["key_edits"], indent=2),
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract generalizable style patterns from document edits. "
                    "Return only valid JSON. No markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Pattern JSON parse failed; raw: {raw[:200]}")
        return []

    source_label = edit.get("draft_type", "unknown")
    return [
        StylePattern(
            name=item.get("name", "Unnamed pattern"),
            description=item.get("description", ""),
            source_edit=source_label,
        )
        for item in items
        if item.get("description")
    ]


def build_style_guide(
    edits_path: Path = DATA_DIR / "sample_edits.json",
    save_path: Path  = OUTPUTS_DIR / "style_guide.json",
) -> StyleGuide:
    """
    Load all edit pairs, extract patterns from each, deduplicate, and save.
    """
    client = Groq(api_key=GROQ_API_KEY)
    edits  = json.loads(edits_path.read_text(encoding="utf-8"))

    all_patterns: list[StylePattern] = []
    for edit in edits:
        print(f"  [LEARNER] Extracting patterns from '{edit['draft_type']}' edit …")
        patterns = extract_patterns_from_edit(client, edit)
        print(f"           ✓ {len(patterns)} patterns extracted")
        all_patterns.extend(patterns)

    # Deduplicate by name (case-insensitive)
    seen_names: set[str] = set()
    unique: list[StylePattern] = []
    for p in all_patterns:
        key = p.name.lower().strip()
        if key not in seen_names:
            seen_names.add(key)
            unique.append(p)

    guide = StyleGuide(patterns=unique)
    guide.save(save_path)
    return guide


def score_draft(client: Groq, draft_text: str) -> dict[str, Any]:
    """
    Score a draft on 8 quality dimensions using Groq as an evaluator.
    Returns a dict with scores, total, and summary.
    """
    prompt = _SCORING_PROMPT.format(draft_text=draft_text[:6000])  # truncate for token budget

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You score document quality. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Score parse failed", "raw": raw[:300]}


def demonstrate_improvement(
    generator,                         # DraftGenerator instance
    style_guide: StyleGuide,
    draft_type: str = "document_checklist",
    save_dir: Path  = OUTPUTS_DIR,
) -> dict[str, Any]:
    """
    Generate the same draft_type TWICE — without and with style rules —
    then score both versions to show measurable improvement.

    Returns a comparison report dict.
    """
    client = Groq(api_key=GROQ_API_KEY)

    print(f"\n  [LEARNER] Demonstrating improvement on '{draft_type}' …")

    # ── Draft A: No style rules (baseline) ───────────────────────────────────
    print("  [LEARNER] Generating BASELINE draft (no style rules) …")
    draft_baseline = generator.generate(draft_type, style_rules=None)
    draft_baseline.save(save_dir / f"{draft_type}_baseline.txt")

    # ── Draft B: With learned style rules ────────────────────────────────────
    print("  [LEARNER] Generating IMPROVED draft (with style rules) …")
    draft_improved = generator.generate(draft_type, style_rules=style_guide.as_rules())
    draft_improved.save_with_evidence(save_dir / f"{draft_type}_improved.txt")

    # ── Score both ────────────────────────────────────────────────────────────
    print("  [LEARNER] Scoring baseline …")
    score_a = score_draft(client, draft_baseline.content)
    print("  [LEARNER] Scoring improved …")
    score_b = score_draft(client, draft_improved.content)

    # ── Build comparison report ───────────────────────────────────────────────
    report = {
        "draft_type":          draft_type,
        "style_rules_count":   len(style_guide.patterns),
        "style_rules_applied": style_guide.as_rules(),
        "baseline_scores":     score_a,
        "improved_scores":     score_b,
        "improvement_delta":   None,
    }

    if "total" in score_a and "total" in score_b:
        delta = score_b["total"] - score_a["total"]
        report["improvement_delta"] = delta
        print(
            f"\n  ✅ IMPROVEMENT RESULT: "
            f"baseline={score_a['total']}/24, "
            f"improved={score_b['total']}/24, "
            f"delta=+{delta}"
        )
    else:
        print("  [WARN] Could not compute delta — scoring returned errors")

    # Save report
    report_path = save_dir / "improvement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  [LEARNER] Improvement report saved → {report_path}")

    # Save human-readable comparison
    _save_comparison_txt(
        draft_baseline.content,
        draft_improved.content,
        score_a, score_b,
        style_guide.as_rules(),
        save_dir / "improvement_comparison.txt",
    )

    return report


def _save_comparison_txt(
    baseline: str,
    improved: str,
    score_a: dict,
    score_b: dict,
    rules: list[str],
    path: Path,
) -> None:
    sep = "=" * 70
    lines = [
        sep,
        "IMPROVEMENT DEMONSTRATION — SIDE BY SIDE COMPARISON",
        sep,
        "",
        "STYLE RULES APPLIED (learned from operator edits):",
        *[f"  • {r}" for r in rules],
        "",
        sep,
        "SCORES",
        sep,
        f"  Baseline : {score_a.get('total', 'N/A')}/24",
        f"  Improved : {score_b.get('total', 'N/A')}/24",
        f"  Delta    : +{score_b.get('total', 0) - score_a.get('total', 0)}",
        "",
        "Baseline breakdown:",
        *[f"    {k}: {v}" for k, v in score_a.get("scores", {}).items()],
        "",
        "Improved breakdown:",
        *[f"    {k}: {v}" for k, v in score_b.get("scores", {}).items()],
        "",
        sep,
        "BASELINE DRAFT",
        sep,
        baseline,
        "",
        sep,
        "IMPROVED DRAFT (WITH LEARNED STYLE RULES)",
        sep,
        improved,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [LEARNER] Comparison report saved → {path}")
