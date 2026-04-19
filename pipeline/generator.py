# """
# generator.py — Draft Generation Module

# Responsibilities:
#   1. Accept a draft type request (title_review, case_status_memo, etc.)
#   2. Build targeted retrieval queries to pull relevant evidence
#   3. Inject retrieved evidence + case context into structured prompts
#   4. Call Groq to generate grounded, well-organised drafts
#   5. Return DraftOutput containing the text + evidence used (for traceability)

# Every claim in the output must be traceable to a retrieved source chunk.
# """

# from __future__ import annotations

# import json
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any

# from groq import Groq

# from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS
# from pipeline.retriever import DocumentIndex, RetrievedChunk, format_evidence


# # ── Data Model ───────────────────────────────────────────────────────────────

# @dataclass
# class DraftOutput:
#     """The result of a draft generation call."""
#     draft_type:     str
#     content:        str                          # the generated draft text
#     evidence_used:  list[RetrievedChunk] = field(default_factory=list)
#     style_rules_applied: list[str]       = field(default_factory=list)

#     def save(self, path: Path) -> None:
#         path.write_text(self.content, encoding="utf-8")
#         print(f"  [GENERATOR] Draft saved → {path}")

#     def save_with_evidence(self, path: Path) -> None:
#         """Save draft + appended evidence manifest."""
#         evidence_block = "\n\n" + "=" * 70 + "\nEVIDENCE MANIFEST\n" + "=" * 70 + "\n"
#         for c in self.evidence_used:
#             evidence_block += f"\n{c.citation()} score={c.score:.3f}\n{c.text}\n"
#         if self.style_rules_applied:
#             evidence_block += "\n" + "=" * 70 + "\nSTYLE RULES APPLIED\n" + "=" * 70 + "\n"
#             for r in self.style_rules_applied:
#                 evidence_block += f"  • {r}\n"
#         path.write_text(self.content + evidence_block, encoding="utf-8")
#         print(f"  [GENERATOR] Draft + evidence manifest saved → {path}")


# # ── Retrieval query plans per draft type ─────────────────────────────────────
# # Each draft type gets a list of targeted queries. This ensures we pull
# # relevant chunks from all relevant documents, not just the most obvious ones.

# _RETRIEVAL_QUERIES: dict[str, list[str]] = {
#     "title_review_summary": [
#         "mortgage lien amount instrument number recorded",
#         "assignment of mortgage Nationstar Mr. Cooper",
#         "HOA lis pendens Palmetto Bay unpaid assessments",
#         "property taxes delinquent 2025 tax parcel",
#         "chain of title ownership vesting Maria Santos Carlos Rodriguez",
#         "easement restrictive covenants encumbrances",
#         "judgment search federal tax liens unsatisfied",
#         "special assessment district",
#     ],
#     "case_status_memo": [
#         "servicer transfer Wells Fargo Nationstar April 1 effective date",
#         "fee authorization resubmit invoice rejected",
#         "borrower counsel attorney Rafael Mendez contact",
#         "payoff amount updated March 2026",
#         "court order case management conference April 22",
#         "proof of service deadline April 15",
#         "case management report due April 12 requirements",
#         "HOA lis pendens party defendant foreclosure",
#         "property taxes delinquent title concern",
#         "judge Navarro courtroom case number",
#     ],
#     "document_checklist": [
#         "documents filed on file case status",
#         "proof of service defendants named",
#         "case management report requirements filed",
#         "complaint filed pre-filing status",
#         "title search assignment chain complete",
#         "HOA party named defendant",
#         "mediation scheduled pending motions",
#         "payoff updated fee authorization submitted",
#     ],
#     "action_item_extract": [
#         "action required urgent deadline",
#         "resubmit fee authorization April 1 transfer",
#         "proof of service April 15 court order",
#         "case management report April 12 filing",
#         "HOA evaluate name party defendant",
#         "payoff update servicer email",
#         "borrower communication attorney route",
#         "delinquent taxes must be addressed",
#     ],
# }


# # ── System prompt (shared) ────────────────────────────────────────────────────

# _SYSTEM_PROMPT = """You are a precise legal case management assistant generating
# first-pass draft documents for attorneys and processors.

# RULES:
# 1. Base every factual claim ONLY on the provided EVIDENCE blocks below.
# 2. If a fact is not in the evidence, say "NOT FOUND IN SOURCE DOCUMENTS" — never fabricate.
# 3. Use the exact section structure and labels specified in the user prompt.
# 4. Include [Source: doc_id, chunk N] inline citations where a fact came from evidence.
# 5. Use active, actionable language (not passive summaries).
# 6. Flag items needing attorney attention with "ACTION REQUIRED:".
# """


# # ── Draft-specific prompt templates ──────────────────────────────────────────

# def _build_title_review_prompt(
#     case_context: dict[str, Any],
#     evidence: str,
#     style_rules: str,
# ) -> str:
#     return f"""
# {style_rules}

# Generate a TITLE REVIEW SUMMARY for the following case.

# CASE CONTEXT:
# {json.dumps(case_context, indent=2)}

# EVIDENCE FROM SOURCE DOCUMENTS:
# {evidence}

# Use this EXACT section structure:

# Title Review Summary — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

# Property: {case_context.get('property_address', '')}
# County: {case_context.get('county', '')} | State: {case_context.get('state', '')}
# Effective Date: [from evidence]

# LIENS & ENCUMBRANCES
# [Number each lien. For each: type, party, amount, dates, instrument number, status, ACTION REQUIRED if applicable]

# TAX STATUS
# [Tax years, amounts, PAID/DELINQUENT status, parcel number, special assessments]

# OWNERSHIP
# [Current vesting, prior owner, deed type, prior mortgage satisfaction if applicable]

# OTHER MATTERS
# [Easements, covenants, judgment search results]

# REVIEWER NOTES
# [3-5 actionable items the attorney must address, grounded in evidence]

# If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
# """


# def _build_case_status_memo_prompt(
#     case_context: dict[str, Any],
#     evidence: str,
#     style_rules: str,
# ) -> str:
#     return f"""
# {style_rules}

# Generate a CASE STATUS MEMO for the following case pulling from ALL source documents.

# CASE CONTEXT:
# {json.dumps(case_context, indent=2)}

# EVIDENCE FROM SOURCE DOCUMENTS:
# {evidence}

# Use this EXACT section structure:

# Case Status Memo — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})
# Prepared: [today's date from evidence or context]

# Borrower: ...
# Property: ...
# Servicer: ... → ... (effective [date])
# Borrower's Counsel: [name, firm, phone — from evidence]
# Court: ...
# Case No.: ...
# Judge: ...

# Current Status: [pre-filing / active / etc.]

# ACTION ITEMS (by priority)
# [List each as: N. URGENT/HIGH/NORMAL — [action] — deadline if applicable]
# [Include consequence of non-action where evidence supports it]

# UPCOMING DEADLINES
# [Date — description — what must happen]

# TITLE CONCERNS
# [Pull from title search evidence: delinquent taxes, HOA lien, assignment chain]

# NOTES
# [Payoff amount, new servicer address, other key facts from evidence]

# If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
# """


# def _build_document_checklist_prompt(
#     case_context: dict[str, Any],
#     evidence: str,
#     style_rules: str,
# ) -> str:
#     return f"""
# {style_rules}

# Generate a DOCUMENT CHECKLIST for the following case.

# CASE CONTEXT:
# {json.dumps(case_context, indent=2)}

# EVIDENCE FROM SOURCE DOCUMENTS:
# {evidence}

# Use this EXACT section structure:

# Document Checklist — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

# DOCUMENTS ON FILE
# [List each document we have, what it covers, its effective date]

# DOCUMENTS REQUIRED BUT NOT YET FILED
# [Infer from court order requirements and case status what still needs to be filed]

# UPCOMING FILING DEADLINES
# [Date — Document — Court/party it goes to]

# OUTSTANDING ACTION ITEMS
# [What is blocked or pending — cross-reference across all documents]

# If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
# """


# def _build_action_item_extract_prompt(
#     case_context: dict[str, Any],
#     evidence: str,
#     style_rules: str,
# ) -> str:
#     return f"""
# {style_rules}

# Generate a prioritized ACTION ITEM EXTRACT for the following case.

# CASE CONTEXT:
# {json.dumps(case_context, indent=2)}

# EVIDENCE FROM SOURCE DOCUMENTS:
# {evidence}

# Use this EXACT section structure:

# Action Item Extract — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

# URGENT ACTIONS (must complete immediately)
# [N. Action — Deadline — Source document — Consequence if missed]

# HIGH PRIORITY ACTIONS
# [N. Action — Deadline — Source document]

# NORMAL PRIORITY ACTIONS
# [N. Action — No hard deadline — Source document]

# PENDING / MONITOR
# [Items to watch but no immediate action needed]

# If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
# """


# _PROMPT_BUILDERS = {
#     "title_review_summary":  _build_title_review_prompt,
#     "case_status_memo":      _build_case_status_memo_prompt,
#     "document_checklist":    _build_document_checklist_prompt,
#     "action_item_extract":   _build_action_item_extract_prompt,
# }


# # ── Public API ────────────────────────────────────────────────────────────────

# class DraftGenerator:
#     """
#     Generates grounded draft outputs using RAG (retrieve → generate).

#     Usage:
#         gen = DraftGenerator(index, case_context)
#         draft = gen.generate("title_review_summary")
#         draft = gen.generate("case_status_memo", style_rules=["Always prioritize..."])
#     """

#     def __init__(
#         self,
#         index: DocumentIndex,
#         case_context: dict[str, Any],
#     ):
#         self._index   = index
#         self._context = case_context
#         self._client  = Groq(api_key=GROQ_API_KEY)

#     def generate(
#         self,
#         draft_type: str,
#         style_rules: list[str] | None = None,
#         top_k_per_query: int = 3,
#     ) -> DraftOutput:
#         """
#         Generate a draft of `draft_type` grounded in retrieved evidence.

#         Args:
#             draft_type:      One of the keys in _PROMPT_BUILDERS.
#             style_rules:     Optional list of learned style rules to prepend.
#             top_k_per_query: Chunks to retrieve per query (deduplicated).

#         Returns:
#             DraftOutput with generated text + evidence manifest.
#         """
#         if draft_type not in _PROMPT_BUILDERS:
#             raise ValueError(
#                 f"Unknown draft_type '{draft_type}'. "
#                 f"Choose from: {list(_PROMPT_BUILDERS.keys())}"
#             )

#         # ── Step 1: Retrieve evidence ─────────────────────────────────────────
#         queries = _RETRIEVAL_QUERIES.get(draft_type, [draft_type])
#         seen_keys: set[tuple] = set()
#         all_chunks: list[RetrievedChunk] = []

#         for query in queries:
#             for chunk in self._index.retrieve(query, top_k=top_k_per_query):
#                 key = (chunk.doc_id, chunk.chunk_index)
#                 if key not in seen_keys:
#                     seen_keys.add(key)
#                     all_chunks.append(chunk)

#         # Sort by score descending so the most relevant appears first in prompt
#         all_chunks.sort(key=lambda c: c.score, reverse=True)
#         evidence_text = format_evidence(all_chunks)

#         # ── Step 2: Build style-rule preamble ────────────────────────────────
#         if style_rules:
#             style_preamble = (
#                 "LEARNED STYLE RULES (apply these to every section of the output):\n"
#                 + "\n".join(f"  • {r}" for r in style_rules)
#                 + "\n"
#             )
#         else:
#             style_preamble = ""

#         # ── Step 3: Build the full prompt ────────────────────────────────────
#         prompt_fn = _PROMPT_BUILDERS[draft_type]
#         user_prompt = prompt_fn(self._context, evidence_text, style_preamble)

#         # ── Step 4: Call Groq ─────────────────────────────────────────────────
#         print(f"  [GENERATOR] Generating '{draft_type}' "
#               f"({len(all_chunks)} evidence chunks, "
#               f"{len(style_rules or [])} style rules) …")

#         response = self._client.chat.completions.create(
#             model=GROQ_MODEL,
#             max_tokens=GROQ_MAX_TOKENS,
#             messages=[
#                 {"role": "system", "content": _SYSTEM_PROMPT},
#                 {"role": "user",   "content": user_prompt},
#             ],
#         )

#         draft_text = response.choices[0].message.content.strip()
#         print(f"  [GENERATOR] ✓ Draft generated ({len(draft_text)} chars)")

#         return DraftOutput(
#             draft_type=draft_type,
#             content=draft_text,
#             evidence_used=all_chunks,
#             style_rules_applied=style_rules or [],
#         )


"""
generator.py — Draft Generation Module

Responsibilities:
  1. Accept a draft type request (title_review, case_status_memo, etc.)
  2. Build targeted retrieval queries to pull relevant evidence
  3. Inject retrieved evidence + case context into structured prompts
  4. Call Groq to generate grounded, well-organised drafts
  5. Return DraftOutput containing the text + evidence used (for traceability)

Every claim in the output must be traceable to a retrieved source chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS
from pipeline.retriever import DocumentIndex, RetrievedChunk, format_evidence


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class DraftOutput:
    """The result of a draft generation call."""
    draft_type:     str
    content:        str                          # the generated draft text
    evidence_used:  list[RetrievedChunk] = field(default_factory=list)
    style_rules_applied: list[str]       = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(self.content, encoding="utf-8")
        print(f"  [GENERATOR] Draft saved → {path}")

    def save_with_evidence(self, path: Path) -> None:
        """Save draft + appended evidence manifest."""
        evidence_block = "\n\n" + "=" * 70 + "\nEVIDENCE MANIFEST\n" + "=" * 70 + "\n"
        for c in self.evidence_used:
            evidence_block += f"\n{c.citation()} score={c.score:.3f}\n{c.text}\n"
        if self.style_rules_applied:
            evidence_block += "\n" + "=" * 70 + "\nSTYLE RULES APPLIED\n" + "=" * 70 + "\n"
            for r in self.style_rules_applied:
                evidence_block += f"  • {r}\n"
        path.write_text(self.content + evidence_block, encoding="utf-8")
        print(f"  [GENERATOR] Draft + evidence manifest saved → {path}")


# ── Retrieval query plans per draft type ─────────────────────────────────────
#
# Two-layer query strategy:
#
#   Layer 1 — GENERIC concept queries (case-independent)
#             These use legal/domain vocabulary that will match relevant chunks
#             regardless of borrower name, lender name, county, etc.
#             They work for ANY foreclosure/title case.
#
#   Layer 2 — DYNAMIC queries built at runtime from case_context
#             (see _build_dynamic_queries below)
#             These inject the actual borrower name, servicer name, county etc.
#             so FAISS can also find chunks that contain those specific terms.
#
# This replaces the previous hardcoded approach where queries contained
# specific names (Nationstar, Rodriguez, Palmetto Bay, Navarro) that would
# fail to retrieve anything useful on a different case.

_GENERIC_QUERIES: dict[str, list[str]] = {
    "title_review_summary": [
        "mortgage lien original amount recorded instrument number",
        "assignment of mortgage transfer lender servicer",
        "homeowners association lis pendens unpaid assessments lien",
        "property taxes delinquent unpaid tax parcel number",
        "chain of title ownership deed vesting current owner",
        "prior owner conveyance warranty deed satisfaction",
        "easement encumbrance right of way utility company",
        "restrictive covenants recorded official records book page",
        "judgment search federal tax lien state lien unsatisfied",
        "special assessment district municipal services",
    ],
    "case_status_memo": [
        "servicer transfer effective date new servicer address",
        "fee authorization invoice resubmit billing rejected",
        "borrower attorney counsel contact phone email",
        "payoff amount outstanding balance updated",
        "case management conference court order date time",
        "proof of service defendants deadline court filing",
        "case management report requirements service status motions",
        "homeowners association party defendant complaint naming",
        "property taxes delinquent title concern sale",
        "judge courtroom courthouse case number plaintiff",
    ],
    "document_checklist": [
        "documents on file case status pre-filing",
        "proof of service defendants served named",
        "case management report filed requirements",
        "complaint filed foreclosure action",
        "title search assignment chain verified complete",
        "homeowners association party named defendant",
        "mediation scheduled pending motions outstanding",
        "payoff updated fee authorization submitted servicer",
    ],
    "action_item_extract": [
        "action required urgent attorney review",
        "fee authorization resubmit transfer deadline",
        "proof of service filing court deadline",
        "case management report filing requirements deadline",
        "homeowners association evaluate name party defendant",
        "payoff update servicer transfer email notice",
        "borrower communication attorney route counsel",
        "delinquent taxes must address before sale",
    ],
}


def _build_dynamic_queries(
    draft_type: str,
    case_context: dict,
) -> list[str]:
    """
    Build case-specific retrieval queries from case_context at runtime.

    These complement the generic queries by including actual names, dates,
    and identifiers from this specific case — so chunks containing those
    specific terms are also retrieved.

    Works for any case: just reads from case_context dict, no hardcoding.
    """
    queries = []

    # Extract common fields (present in any case context)
    borrower    = case_context.get("borrower", "")
    servicer    = case_context.get("servicer", "")
    county      = case_context.get("county", "")
    state       = case_context.get("state", "")
    case_number = case_context.get("case_number", "")
    loan_number = case_context.get("loan_number", "")
    property_addr = case_context.get("property_address", "")

    # Build borrower name variants (last name is most useful for retrieval)
    borrower_last = borrower.split(",")[0].strip() if "," in borrower else borrower.split()[-1]

    if draft_type == "title_review_summary":
        if borrower_last:
            queries.append(f"{borrower_last} mortgage lien recorded")
            queries.append(f"{borrower_last} vesting ownership deed")
        if county:
            queries.append(f"{county} County official records instrument")
        if loan_number:
            queries.append(f"instrument {loan_number} recorded")
        if property_addr:
            # Just use street number + street name, skip city/state
            addr_short = " ".join(property_addr.split()[:4])
            queries.append(f"{addr_short} title exceptions")

    elif draft_type == "case_status_memo":
        if borrower_last:
            queries.append(f"{borrower_last} loan servicer transfer")
            queries.append(f"{borrower_last} borrower counsel attorney")
        if servicer:
            # servicer field often has "A transferring to B" — extract both parts
            queries.append(f"servicer {servicer[:40]} transfer effective")
        if case_number:
            queries.append(f"case {case_number} court order deadline")
        if county:
            queries.append(f"{county} circuit court judge conference")

    elif draft_type == "document_checklist":
        if borrower_last:
            queries.append(f"{borrower_last} case documents filed status")
        if case_number:
            queries.append(f"{case_number} complaint service defendants")

    elif draft_type == "action_item_extract":
        if borrower_last:
            queries.append(f"{borrower_last} action required deadline")
        if servicer:
            queries.append(f"servicer {servicer[:40]} invoice authorization")

    return queries


def _get_retrieval_queries(draft_type: str, case_context: dict) -> list[str]:
    """
    Return the full combined query list for a draft type:
    generic concept queries + dynamic case-specific queries.
    """
    generic  = _GENERIC_QUERIES.get(draft_type, [draft_type])
    dynamic  = _build_dynamic_queries(draft_type, case_context)
    return generic + dynamic


# ── System prompt (shared) ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise legal case management assistant generating
first-pass draft documents for attorneys and processors.

RULES:
1. Base every factual claim ONLY on the provided EVIDENCE blocks below.
2. If a fact is not in the evidence, say "NOT FOUND IN SOURCE DOCUMENTS" — never fabricate.
3. Use the exact section structure and labels specified in the user prompt.
4. Include [Source: doc_id, chunk N] inline citations where a fact came from evidence.
5. Use active, actionable language (not passive summaries).
6. Flag items needing attorney attention with "ACTION REQUIRED:".
"""


# ── Draft-specific prompt templates ──────────────────────────────────────────

def _build_title_review_prompt(
    case_context: dict[str, Any],
    evidence: str,
    style_rules: str,
) -> str:
    return f"""
{style_rules}

Generate a TITLE REVIEW SUMMARY for the following case.

CASE CONTEXT:
{json.dumps(case_context, indent=2)}

EVIDENCE FROM SOURCE DOCUMENTS:
{evidence}

Use this EXACT section structure:

Title Review Summary — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

Property: {case_context.get('property_address', '')}
County: {case_context.get('county', '')} | State: {case_context.get('state', '')}
Effective Date: [from evidence]

LIENS & ENCUMBRANCES
[Number each lien. For each: type, party, amount, dates, instrument number, status, ACTION REQUIRED if applicable]

TAX STATUS
[Tax years, amounts, PAID/DELINQUENT status, parcel number, special assessments]

OWNERSHIP
[Current vesting, prior owner, deed type, prior mortgage satisfaction if applicable]

OTHER MATTERS
[Easements, covenants, judgment search results]

REVIEWER NOTES
[3-5 actionable items the attorney must address, grounded in evidence]

If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
"""


def _build_case_status_memo_prompt(
    case_context: dict[str, Any],
    evidence: str,
    style_rules: str,
) -> str:
    return f"""
{style_rules}

Generate a CASE STATUS MEMO for the following case pulling from ALL source documents.

CASE CONTEXT:
{json.dumps(case_context, indent=2)}

EVIDENCE FROM SOURCE DOCUMENTS:
{evidence}

Use this EXACT section structure:

Case Status Memo — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})
Prepared: [today's date from evidence or context]

Borrower: ...
Property: ...
Servicer: ... → ... (effective [date])
Borrower's Counsel: [name, firm, phone — from evidence]
Court: ...
Case No.: ...
Judge: ...

Current Status: [pre-filing / active / etc.]

ACTION ITEMS (by priority)
[List each as: N. URGENT/HIGH/NORMAL — [action] — deadline if applicable]
[Include consequence of non-action where evidence supports it]

UPCOMING DEADLINES
[Date — description — what must happen]

TITLE CONCERNS
[Pull from title search evidence: delinquent taxes, HOA lien, assignment chain]

NOTES
[Payoff amount, new servicer address, other key facts from evidence]

If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
"""


def _build_document_checklist_prompt(
    case_context: dict[str, Any],
    evidence: str,
    style_rules: str,
) -> str:
    return f"""
{style_rules}

Generate a DOCUMENT CHECKLIST for the following case.

CASE CONTEXT:
{json.dumps(case_context, indent=2)}

EVIDENCE FROM SOURCE DOCUMENTS:
{evidence}

Use this EXACT section structure:

Document Checklist — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

DOCUMENTS ON FILE
[List each document we have, what it covers, its effective date]

DOCUMENTS REQUIRED BUT NOT YET FILED
[Infer from court order requirements and case status what still needs to be filed]

UPCOMING FILING DEADLINES
[Date — Document — Court/party it goes to]

OUTSTANDING ACTION ITEMS
[What is blocked or pending — cross-reference across all documents]

If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
"""


def _build_action_item_extract_prompt(
    case_context: dict[str, Any],
    evidence: str,
    style_rules: str,
) -> str:
    return f"""
{style_rules}

Generate a prioritized ACTION ITEM EXTRACT for the following case.

CASE CONTEXT:
{json.dumps(case_context, indent=2)}

EVIDENCE FROM SOURCE DOCUMENTS:
{evidence}

Use this EXACT section structure:

Action Item Extract — {case_context.get('borrower', 'Unknown')} ({case_context.get('case_number', '')})

URGENT ACTIONS (must complete immediately)
[N. Action — Deadline — Source document — Consequence if missed]

HIGH PRIORITY ACTIONS
[N. Action — Deadline — Source document]

NORMAL PRIORITY ACTIONS
[N. Action — No hard deadline — Source document]

PENDING / MONITOR
[Items to watch but no immediate action needed]

If any field is not found in the evidence, write "NOT FOUND IN SOURCE DOCUMENTS".
"""


_PROMPT_BUILDERS = {
    "title_review_summary":  _build_title_review_prompt,
    "case_status_memo":      _build_case_status_memo_prompt,
    "document_checklist":    _build_document_checklist_prompt,
    "action_item_extract":   _build_action_item_extract_prompt,
}


# ── Public API ────────────────────────────────────────────────────────────────

class DraftGenerator:
    """
    Generates grounded draft outputs using RAG (retrieve → generate).

    Usage:
        gen = DraftGenerator(index, case_context)
        draft = gen.generate("title_review_summary")
        draft = gen.generate("case_status_memo", style_rules=["Always prioritize..."])
    """

    def __init__(
        self,
        index: DocumentIndex,
        case_context: dict[str, Any],
    ):
        self._index   = index
        self._context = case_context
        self._client  = Groq(api_key=GROQ_API_KEY)

    def generate(
        self,
        draft_type: str,
        style_rules: list[str] | None = None,
        top_k_per_query: int = 3,
    ) -> DraftOutput:
        """
        Generate a draft of `draft_type` grounded in retrieved evidence.

        Args:
            draft_type:      One of the keys in _PROMPT_BUILDERS.
            style_rules:     Optional list of learned style rules to prepend.
            top_k_per_query: Chunks to retrieve per query (deduplicated).

        Returns:
            DraftOutput with generated text + evidence manifest.
        """
        if draft_type not in _PROMPT_BUILDERS:
            raise ValueError(
                f"Unknown draft_type '{draft_type}'. "
                f"Choose from: {list(_PROMPT_BUILDERS.keys())}"
            )

        # ── Step 1: Retrieve evidence ─────────────────────────────────────────
        # Generic concept queries + dynamic case-specific queries
        queries = _get_retrieval_queries(draft_type, self._context)
        seen_keys: set[tuple] = set()
        all_chunks: list[RetrievedChunk] = []

        for query in queries:
            for chunk in self._index.retrieve(query, top_k=top_k_per_query):
                key = (chunk.doc_id, chunk.chunk_index)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_chunks.append(chunk)

        # Sort by score descending so the most relevant appears first in prompt
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        evidence_text = format_evidence(all_chunks)

        # ── Step 2: Build style-rule preamble ────────────────────────────────
        if style_rules:
            style_preamble = (
                "LEARNED STYLE RULES (apply these to every section of the output):\n"
                + "\n".join(f"  • {r}" for r in style_rules)
                + "\n"
            )
        else:
            style_preamble = ""

        # ── Step 3: Build the full prompt ────────────────────────────────────
        prompt_fn = _PROMPT_BUILDERS[draft_type]
        user_prompt = prompt_fn(self._context, evidence_text, style_preamble)

        # ── Step 4: Call Groq ─────────────────────────────────────────────────
        print(f"  [GENERATOR] Generating '{draft_type}' "
              f"({len(all_chunks)} evidence chunks, "
              f"{len(style_rules or [])} style rules) …")

        response = self._client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        draft_text = response.choices[0].message.content.strip()
        print(f"  [GENERATOR] ✓ Draft generated ({len(draft_text)} chars)")

        return DraftOutput(
            draft_type=draft_type,
            content=draft_text,
            evidence_used=all_chunks,
            style_rules_applied=style_rules or [],
        )
