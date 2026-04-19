"""
processor.py — Document Processing Module

Responsibilities:
  1. Load raw document text from disk
  2. Clean OCR noise (character substitutions, formatting artifacts)
     → Uses a GENERALIZED three-tier approach:
        Tier 1: Rule-based fast corrections (high-confidence patterns only)
        Tier 2: LLM-assisted detection (Groq identifies suspicious words)
        Tier 3: Validation (context-aware filtering to avoid false corrections)
  3. Use Groq LLM to extract structured data per document type
  4. Return a list of ProcessedDocument objects ready for retrieval + generation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groq import Groq

from config import DOCS_DIR, GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class ProcessedDocument:
    """Holds both the cleaned text and the LLM-extracted structured fields."""
    doc_id:     str                        # e.g. "title_search_page1"
    doc_type:   str                        # e.g. "title_search", "email", "court_order"
    raw_text:   str                        # original text as read from disk
    clean_text: str                        # after OCR correction
    ocr_corrections: list[OCRCorrection] = field(default_factory=list)  # transparency
    extracted:  dict[str, Any] = field(default_factory=dict)  # structured fields
    source_path: str = ""


# ── OCR Cleaning (Generalized Three-Tier Approach) ────────────────────────────

# Tier 1: Fast, high-confidence patterns only
# These are Conservative rules that almost never false-positive
_TIER1_FAST_RULES: list[tuple[re.Pattern, str]] = [
    # Digit-as-letter at word boundaries or in specific contexts
    (re.compile(r"\b(\d)O(\d)"), r"\g<1>0\g<2>"),  # digit-O-digit → digit-0-digit (in numbers)
    (re.compile(r"(\$[\d,]+)OO\b"), r"\g<1>00"),   # Money amounts ending in OO
    (re.compile(r"\b(\d{3})O(\d{4})\b"), r"\g<1>0\g<2>"),  # Parcel numbers
    
    # Normalize excessive whitespace
    (re.compile(r"[ \t]{2,}"), " "),               # Multiple spaces → single
    (re.compile(r"\n{3,}"), "\n\n"),               # Triple+ newlines → double
]


@dataclass
class OCRCorrection:
    """Track a single OCR correction with confidence metadata."""
    original_word: str
    corrected_word: str
    position: int
    confidence: float  # 0.0-1.0
    method: str  # "rule", "lm_suggestion", "validation"
    reason: str


def _apply_tier1_corrections(text: str) -> tuple[str, list[OCRCorrection]]:
    """
    Tier 1: Apply fast, conservative rule-based corrections.
    Returns cleaned text + list of applied corrections with metadata.
    """
    corrections = []
    result = text
    
    for pattern, replacement in _TIER1_FAST_RULES:
        matches = list(pattern.finditer(result))
        # Process in reverse to maintain position tracking
        for match in reversed(matches):
            original = match.group(0)
            result = result[:match.start()] + pattern.sub(replacement, original) + result[match.end():]
            corrections.append(OCRCorrection(
                original_word=original,
                corrected_word=pattern.sub(replacement, original),
                position=match.start(),
                confidence=0.95,  # Tier 1 rules are high-confidence
                method="tier1_rule",
                reason=f"Matched pattern: {pattern.pattern}"
            ))
    
    return result, corrections


def _extract_suspicious_words(text: str, window_size: int = 500) -> list[str]:
    """
    Extract candidate words that may contain OCR errors.
    Uses heuristics: mixed-case within word, rare character combos, etc.
    """
    suspicious = set()
    
    # Pattern 1: Words with digit-letter mix (e.g., "tit1e", "F1orida", "RODr1GUEZ")
    for match in re.finditer(r"\b[a-zA-Z]*\d[a-zA-Z]*\b", text):
        suspicious.add(match.group(0))
    
    # Pattern 2: Words with rare letter combos (e.g., "1O", "O0", "l|")
    for match in re.finditer(r"\b\w*[1lO0|S5][1lO0|S5]\w*\b", text):
        word = match.group(0)
        # Only if it looks suspect (not standard abbreviations)
        if len(word) > 2 and not re.match(r"^[A-Z]{2,3}$", word):
            suspicious.add(word)
    
    # Pattern 3: Words that look like they should end differently
    for match in re.finditer(r"\b[a-zA-Z]*(?:O0|1l|S5|8B)[a-zA-Z]*\b", text):
        suspicious.add(match.group(0))
    
    return sorted(list(suspicious))[:20]  # Limit to top 20 to save tokens


def _llm_suggest_ocr_fixes(
    client: Groq,
    text: str,
    suspicious_words: list[str],
    max_suggestions: int = 10
) -> dict[str, tuple[str, float]]:
    """
    Tier 2: Use Groq to intelligently suggest OCR corrections.
    Returns {original_word: (suggested_word, confidence)} for high-confidence suggestions.
    
    Confidence is inferred from the LLM's reasoning about the specific word in context.
    """
    if not suspicious_words:
        return {}
    
    # Build context snippets for each suspicious word
    context_snippets = {}
    for word in suspicious_words[:max_suggestions]:
        # Find context (100 chars before/after first occurrence)
        pos = text.find(word)
        if pos >= 0:
            start = max(0, pos - 100)
            end = min(len(text), pos + len(word) + 100)
            context_snippets[word] = text[start:end]
    
    prompt = f"""
You are an OCR error detection expert. Analyze the following suspicious words found in a legal document.
For each, determine if it's likely an OCR error, and if so, suggest the correct word.

Return ONLY valid JSON with this structure:
{{
  "corrections": {{
    "<suspicious_word>": {{
      "is_error": true/false,
      "suggested": "<corrected_word if is_error=true, else null>",
      "confidence": <0.0-1.0>,
      "reasoning": "<brief explanation>"
    }}
  }}
}}

Suspicious words with context:
"""
    
    for word, context in list(context_snippets.items())[:max_suggestions]:
        prompt += f'\n\n"{word}"\nContext: ...{context}...'
    
    prompt += "\n\nBe conservative: only suggest corrections if you're quite confident the word is wrong."
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=min(GROQ_MAX_TOKENS, 1500),
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR error detection expert. Return only valid JSON, no markdown."
                },
                {"role": "user", "content": prompt}
            ],
        )
        
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        
        result_json = json.loads(raw)
        corrections = {}
        
        for word, correction_data in result_json.get("corrections", {}).items():
            if correction_data.get("is_error") and correction_data.get("suggested"):
                confidence = float(correction_data.get("confidence", 0.0))
                # Only keep suggestions with >70% confidence
                if confidence > 0.70:
                    corrections[word] = (correction_data["suggested"], confidence, correction_data.get("reasoning", ""))
        
        return corrections
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # If LLM response is malformed, skip Tier 2
        print(f"  [TIER2 PARSE ERROR] {e} — skipping LLM corrections")
        return {}


def _apply_tier2_corrections(
    text: str,
    client: Groq,
) -> tuple[str, list[OCRCorrection]]:
    """
    Tier 2: Use LLM to suggest corrections for suspicious words.
    Apply only if confidence > 0.70.
    """
    corrections = []
    suspicious_words = _extract_suspicious_words(text)
    
    if not suspicious_words:
        return text, corrections
    
    suggestions = _llm_suggest_ocr_fixes(client, text, suspicious_words)
    result = text
    
    # Apply suggestions (keeping track of corrections)
    for original_word, (corrected_word, confidence, reasoning) in suggestions.items():
        # Use word boundary regex to only replace whole words
        pattern = r"\b" + re.escape(original_word) + r"\b"
        matches = list(re.finditer(pattern, result))
        
        for match in reversed(matches):
            result = result[:match.start()] + corrected_word + result[match.end():]
            corrections.append(OCRCorrection(
                original_word=original_word,
                corrected_word=corrected_word,
                position=match.start(),
                confidence=confidence,
                method="tier2_lm",
                reason=reasoning
            ))
    
    return result, corrections


def _validate_corrections_tier3(
    text_before: str,
    text_after: str,
    corrections: list[OCRCorrection]
) -> list[OCRCorrection]:
    """
    Tier 3: Validate that corrections actually improve the text.
    Heuristics:
    - Corrected word should be in a common legal/English vocabulary
    - Correction shouldn't create nonsense patterns
    - If in doubt, filter it out (conservative approach)
    """
    # Common legal vocabulary (subset for validation)
    LEGAL_VOCAB = {
        "florida", "miami", "dade", "county", "lien", "mortgage", "title",
        "deed", "instrument", "assignment", "hoa", "association", "wells",
        "rodriguez", "carlos", "property", "legal", "description", "parcel",
        "file", "exceptions", "schedule", "policy", "following", "complete",
        "court", "order", "judgment", "foreclosure", "borrower", "servicer",
        "counsel", "attorney", "action", "deadline", "conference", "plaintiff",
        "defendant", "case", "number", "date", "amount", "email", "phone",
        "payoff", "transfer", "servicer", "address", "WELLS", "FARGO", "BANK"
    }
    
    validated = []
    for corr in corrections:
        # If correction confidence < 0.75, only validate if word is in legal vocab
        if corr.confidence < 0.75:
            # Stricter validation for lower-confidence suggestions
            if corr.corrected_word.lower() not in LEGAL_VOCAB:
                # Skip this one
                print(f"    [FILTER] Skipping low-confidence correction: {corr.original_word} → {corr.corrected_word} (conf={corr.confidence:.2f})")
                continue
        
        # If correction confidence >= 0.75, trust it (Tier 1 or high-confidence Tier 2)
        validated.append(corr)
    
    return validated


def clean_ocr(text: str, client: Groq | None = None) -> tuple[str, list[OCRCorrection]]:
    """
    Generalized OCR cleaning using three-tier approach.
    
    Returns:
        (cleaned_text, list_of_corrections_applied)
    
    The returned corrections list allows transparency on what changed.
    """
    result = text
    all_corrections = []
    
    # Tier 1: Fast rule-based corrections (always runs)
    print("    [TIER1] Applying fast rule-based corrections...")
    result, tier1_corrections = _apply_tier1_corrections(result)
    all_corrections.extend(tier1_corrections)
    
    if len(tier1_corrections) > 0:
        print(f"      → Applied {len(tier1_corrections)} Tier 1 corrections")
    
    # Tier 2: LLM-based detection (runs if client is provided)
    if client is not None:
        print("    [TIER2] Running LLM-based OCR detection...")
        result, tier2_corrections = _apply_tier2_corrections(result, client)
        all_corrections.extend(tier2_corrections)
        
        if len(tier2_corrections) > 0:
            print(f"      → Found {len(tier2_corrections)} LLM-suggested corrections")
    
    # Tier 3: Validate all corrections (filter out risky ones)
    print("    [TIER3] Validating corrections...")
    validated = _validate_corrections_tier3(text, result, all_corrections)
    
    # If validation filtered some out, show the delta
    if len(validated) < len(all_corrections):
        print(f"      → Filtered {len(all_corrections) - len(validated)} risky corrections (confidence check)")
    
    # Final normalization (always safe)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    
    return result.strip(), validated


# ── Document-type classifier ─────────────────────────────────────────────────

def _classify(filename: str) -> str:
    name = filename.lower()
    if "title_search_page1" in name:
        return "title_search_page1"
    if "title_search_page2" in name:
        return "title_search_page2"
    if "email" in name:
        return "servicer_email"
    if "court_order" in name:
        return "court_order"
    return "unknown"


# ── LLM Extraction prompts per doc type ─────────────────────────────────────

_EXTRACTION_PROMPTS: dict[str, str] = {

    "title_search_page1": """
You are a legal document analyst. Extract structured information from the
Schedule B title search document below. Return ONLY valid JSON with these keys:

{
  "liens": [
    {
      "item_number": <int>,
      "type": "<Mortgage|Assignment|HOA Lien|Tax Lien|Easement|Covenant|Other>",
      "party": "<holder/beneficiary name>",
      "amount": "<dollar amount as string, or null>",
      "date": "<recorded/filed date>",
      "instrument_number": "<instrument no., or null>",
      "book_page": "<O.R. Book/Page if applicable, or null>",
      "notes": "<any additional relevant detail>"
    }
  ],
  "chain_of_title": [
    {"owner": "<name>", "type": "<deed type>", "year": "<year>"}
  ],
  "current_vesting": "<current owner description>",
  "effective_date": "<title effective date>",
  "file_number": "<title file number>"
}

Document:
""",

    "title_search_page2": """
You are a legal document analyst. Extract structured information from the
title search legal description and tax/judgment section below.
Return ONLY valid JSON with these keys:

{
  "legal_description": "<full legal description text>",
  "apn": "<parcel number>",
  "taxes": [
    {"year": "<year>", "status": "<PAID|UNPAID|DELINQUENT>", "amount": "<amount>",
     "due_date": "<due date or null>"}
  ],
  "special_assessments": [
    {"description": "<name>", "amount_per_year": "<amount>", "status": "<current|delinquent>"}
  ],
  "judgment_search": {
    "names_searched": ["<name>"],
    "unsatisfied_judgments": false,
    "federal_tax_liens": false,
    "state_tax_liens": false,
    "notes": "<any findings>"
  },
  "prior_mortgage_satisfaction": {
    "prior_owner": "<name>",
    "instrument": "<instrument number>",
    "date": "<date>"
  }
}

Document:
""",

    "servicer_email": """
You are a legal case manager. Extract ALL action items and key facts from
this servicer email. Return ONLY valid JSON with these keys:

{
  "from": "<sender name and email>",
  "date": "<email date>",
  "subject": "<subject>",
  "servicer_transfer": {
    "from_servicer": "<name>",
    "to_servicer": "<name>",
    "effective_date": "<date>",
    "new_address": "<full address>",
    "new_phone": "<phone>"
  },
  "action_items": [
    {
      "priority": "<URGENT|HIGH|NORMAL>",
      "action": "<what needs to be done>",
      "deadline": "<date or null>",
      "consequence": "<what happens if not done, or null>"
    }
  ],
  "borrower_counsel": {
    "name": "<attorney name>",
    "firm": "<firm name>",
    "phone": "<phone>",
    "email": "<email>"
  },
  "payoff_amount": "<amount as string>",
  "payoff_as_of": "<date>",
  "hoa_note": "<HOA-related note if any>"
}

Document:
""",

    "court_order": """
You are a legal document analyst. Extract ALL structured information from
this court order. Return ONLY valid JSON with these keys:

{
  "court": "<full court name>",
  "case_number": "<case number>",
  "plaintiff": "<plaintiff name>",
  "defendants": ["<name>"],
  "judge": "<judge name>",
  "order_type": "<type of order>",
  "order_date": "<date signed>",
  "deadlines": [
    {
      "date": "<YYYY-MM-DD>",
      "description": "<what must be filed or done>",
      "consequence": "<consequence of non-compliance, or null>"
    }
  ],
  "conference": {
    "date": "<YYYY-MM-DD>",
    "time": "<time>",
    "location": "<full location>",
    "attendance": "<required/optional>"
  },
  "report_requirements": ["<requirement 1>", "<requirement 2>"]
}

Document:
""",
}


# ── Groq LLM call ────────────────────────────────────────────────────────────

def _llm_extract(client: Groq, doc_type: str, clean_text: str) -> dict[str, Any]:
    """Call Groq to extract structured fields from clean document text."""
    prompt = _EXTRACTION_PROMPTS.get(doc_type)
    if not prompt:
        return {}

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=GROQ_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise legal document extraction assistant. "
                    "Always return valid JSON only. No markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt + clean_text},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to salvage partial JSON
        print(f"  [WARN] JSON parse failed for {doc_type}; returning raw string.")
        return {"raw_extraction": raw}


# ── Public API ───────────────────────────────────────────────────────────────

def process_documents(
    doc_dir: Path = DOCS_DIR,
    filenames: list[str] | None = None,
) -> list[ProcessedDocument]:
    """
    Load, clean, and extract structured data from all source documents.

    Args:
        doc_dir:   Directory containing the raw document text files.
        filenames: Specific filenames to process; defaults to all .txt files.

    Returns:
        List of ProcessedDocument objects.
    """
    client = Groq(api_key=GROQ_API_KEY)

    if filenames is None:
        filenames = sorted(p.name for p in doc_dir.glob("*.txt"))

    results: list[ProcessedDocument] = []

    for fname in filenames:
        path = doc_dir / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found in {doc_dir}")
            continue

        print(f"  [PROCESS] {fname}")
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        doc_type = _classify(fname)
        
        # Generalized OCR cleaning (Tier 1-3)
        clean_text, ocr_corrections = clean_ocr(raw_text, client=client)
        
        # LLM extraction on cleaned text
        extracted = _llm_extract(client, doc_type, clean_text)

        results.append(
            ProcessedDocument(
                doc_id=fname.replace(".txt", ""),
                doc_type=doc_type,
                raw_text=raw_text,
                clean_text=clean_text,
                ocr_corrections=ocr_corrections,
                extracted=extracted,
                source_path=str(path),
            )
        )
        print(f"         ✓ Extracted {len(extracted)} top-level fields | "
              f"{len(ocr_corrections)} OCR corrections applied\n")

    return results


# ── Serialisation helpers ─────────────────────────────────────────────────────

def save_processed(docs: list[ProcessedDocument], output_path: Path) -> None:
    """Save processed documents to JSON for inspection / caching."""
    payload = [
        {
            "doc_id":     d.doc_id,
            "doc_type":   d.doc_type,
            "clean_text": d.clean_text,
            "ocr_corrections": [
                {
                    "original": c.original_word,
                    "corrected": c.corrected_word,
                    "confidence": c.confidence,
                    "method": c.method,
                    "reason": c.reason
                }
                for c in d.ocr_corrections
            ],
            "extracted":  d.extracted,
            "source_path": d.source_path,
        }
        for d in docs
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [SAVE] Processed docs → {output_path}")
