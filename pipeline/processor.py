"""
processor.py — Document Processing Module

Responsibilities:
  1. Load raw document text from disk
  2. Clean OCR noise (character substitutions, formatting artifacts)
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
    extracted:  dict[str, Any] = field(default_factory=dict)  # structured fields
    source_path: str = ""


# ── OCR Cleaning ─────────────────────────────────────────────────────────────

# Pairs: (compiled_pattern_or_str, replacement)
_OCR_RULES: list[tuple[str, str]] = [
    # Digit-as-letter inside words (most common title-search OCR artifacts)
    (r"\bF1orida\b",    "Florida"),
    (r"\bF1\b",         "FL"),
    (r"1ien",           "lien"),
    (r"1iens",          "liens"),
    (r"tit1e",          "title"),
    (r"po1icy",         "policy"),
    (r"Po1icy",         "Policy"),
    (r"1ot\b",          "lot"),
    (r"1and\b",         "land"),
    (r"Officia1",       "Official"),
    (r"fo11owing",      "following"),
    (r"a11\b",          "all"),
    (r"pa1metto",       "palmetto"),
    (r"Pa1metto",       "Palmetto"),
    (r"tit1e",          "title"),
    (r"simp1e",         "simple"),
    (r"comp1ete",       "complete"),
    # Zero-as-letter inside numbers / identifiers
    (r"(?<=\d)O(?=\d)", "0"),   # digit-O-digit → digit-0-digit
    (r"(?<=\$[\d,]+)OO\b", "00"),  # $X,XXX.OO → $X,XXX.00
    # All-caps words with embedded 1 (e.g. WE11S, RODR1GUEZ, PALMETT0)
    (r"\bWE11S\b",      "WELLS"),
    (r"\bRODR1GUEZ\b",  "RODRIGUEZ"),
    (r"\bCAR1OS\b",     "CARLOS"),
    (r"\bPALMETT0\b",   "PALMETTO"),
    (r"\bASSOCIATI0N\b","ASSOCIATION"),
    (r"\bFi1e\b",       "File"),
    (r"\bExcept1ons\b", "Exceptions"),
    (r"\bEXCEPT1ONS\b", "EXCEPTIONS"),
    (r"\bSCHEDU1E\b",   "SCHEDULE"),
    # Instrument / parcel number OCR: O inside numeric strings
    (r"\b33-5O22-O14-O29O\b", "33-5022-014-0290"),
    (r"\b2O21\b",       "2021"),
    (r"\b2O25\b",       "2025"),
    (r"\b2O26\b",       "2026"),
    (r"\b2O15\b",       "2015"),
    (r"\b2O1[0-9]\b",   lambda m: m.group(0).replace("O", "0")),
    # Dollar amounts with O instead of 0
    (r"\$8,247\.OO",    "$8,247.00"),
    (r"\$445,OOO\.OO",  "$445,000.00"),
    (r"\$3,42O\.OO",    "$3,420.00"),
]


def clean_ocr(text: str) -> str:
    """Apply rule-based OCR corrections to raw document text."""
    result = text
    for pattern, replacement in _OCR_RULES:
        if callable(replacement):
            result = re.sub(pattern, replacement, result)
        else:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE
                            if pattern[0].islower() else 0)
    # Normalise whitespace: collapse runs of spaces/tabs, keep newlines
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


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

        print(f"  [PROCESS] {fname} …")
        raw_text   = path.read_text(encoding="utf-8", errors="replace")
        clean_text = clean_ocr(raw_text)
        doc_type   = _classify(fname)
        extracted  = _llm_extract(client, doc_type, clean_text)

        results.append(
            ProcessedDocument(
                doc_id=fname.replace(".txt", ""),
                doc_type=doc_type,
                raw_text=raw_text,
                clean_text=clean_text,
                extracted=extracted,
                source_path=str(path),
            )
        )
        print(f"         ✓ extracted {len(extracted)} top-level fields")

    return results


# ── Serialisation helpers ─────────────────────────────────────────────────────

def save_processed(docs: list[ProcessedDocument], output_path: Path) -> None:
    """Save processed documents to JSON for inspection / caching."""
    payload = [
        {
            "doc_id":     d.doc_id,
            "doc_type":   d.doc_type,
            "clean_text": d.clean_text,
            "extracted":  d.extracted,
            "source_path": d.source_path,
        }
        for d in docs
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [SAVE] Processed docs → {output_path}")
