"""
main.py — Legal AI Pipeline Orchestrator

Runs all four pipeline stages in sequence:
  1. Document Processing   — clean OCR, extract structured data
  2. Retrieval Setup       — embed chunks, build FAISS index
  3. Draft Generation      — produce Title Review Summary + Case Status Memo
  4. Improvement from Edits — learn patterns, generate improved Document Checklist

Run:
    python main.py
"""

import json
import sys
from pathlib import Path

from config import DATA_DIR, DOCS_DIR, OUTPUTS_DIR, SOURCE_DOCS
from pipeline.processor import process_documents, save_processed
from pipeline.retriever import DocumentIndex
from pipeline.generator import DraftGenerator
from pipeline.learner   import build_style_guide, demonstrate_improvement


def main() -> None:
    print("\n" + "=" * 60)
    print("  LEGAL AI PIPELINE — Rodriguez Case 2025-FC-08891")
    print("=" * 60 + "\n")

    # ── Load case context ─────────────────────────────────────────────────────
    context_path = DATA_DIR / "case_context.json"
    if not context_path.exists():
        print(f"[ERROR] case_context.json not found at {context_path}")
        sys.exit(1)
    case_context = json.loads(context_path.read_text(encoding="utf-8"))
    print(f"[CONTEXT] Case: {case_context['case_number']} | "
          f"Borrower: {case_context['borrower']}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Document Processing
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("STAGE 1 — Document Processing")
    print("─" * 60)

    processed_docs = process_documents(doc_dir=DOCS_DIR, filenames=SOURCE_DOCS)

    if not processed_docs:
        print("[ERROR] No documents were processed. "
              f"Check that files exist in {DOCS_DIR}")
        sys.exit(1)

    # Save processed output for inspection
    save_processed(processed_docs, OUTPUTS_DIR / "processed_documents.json")
    print(f"\n✓ Stage 1 complete — {len(processed_docs)} documents processed\n")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Retrieval Setup
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("STAGE 2 — Building Retrieval Index")
    print("─" * 60)

    index = DocumentIndex()
    index.build(processed_docs)
    index.save(OUTPUTS_DIR / "faiss_index")
    print(f"\n✓ Stage 2 complete — FAISS index built\n")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Draft Generation
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("STAGE 3 — Draft Generation")
    print("─" * 60)

    generator = DraftGenerator(index=index, case_context=case_context)

    # Draft 1: Title Review Summary
    title_draft = generator.generate("title_review_summary")
    title_draft.save_with_evidence(OUTPUTS_DIR / "draft_title_review_summary.txt")

    # Draft 2: Case Status Memo
    memo_draft = generator.generate("case_status_memo")
    memo_draft.save_with_evidence(OUTPUTS_DIR / "draft_case_status_memo.txt")

    print(f"\n✓ Stage 3 complete — 2 drafts generated\n")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Improvement from Operator Edits
    # ══════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("STAGE 4 — Learning from Operator Edits")
    print("─" * 60)

    # Step 4a: Build style guide from sample_edits.json
    style_guide = build_style_guide(
        edits_path=DATA_DIR / "sample_edits.json",
        save_path=OUTPUTS_DIR / "style_guide.json",
    )

    # Step 4b: Demonstrate improvement on a NEW draft type (document_checklist)
    # — not present in the training edits, so we genuinely test generalisation
    report = demonstrate_improvement(
        generator=generator,
        style_guide=style_guide,
        draft_type="document_checklist",
        save_dir=OUTPUTS_DIR,
    )

    print(f"\n✓ Stage 4 complete — improvement demonstrated\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs written to: {OUTPUTS_DIR}\n")
    print("  processed_documents.json          — structured extraction output")
    print("  faiss_index/                       — persisted FAISS index")
    print("  draft_title_review_summary.txt     — Title Review + evidence")
    print("  draft_case_status_memo.txt         — Case Status Memo + evidence")
    print("  style_guide.json                   — learned operator patterns")
    print("  document_checklist_baseline.txt    — before style rules applied")
    print("  document_checklist_improved.txt    — after style rules applied")
    print("  improvement_report.json            — scores + delta")
    print("  improvement_comparison.txt         — side-by-side human-readable")

    if report.get("improvement_delta") is not None:
        print(f"\n  Quality improvement: +{report['improvement_delta']} points (out of 24)")
    print()


if __name__ == "__main__":
    main()
