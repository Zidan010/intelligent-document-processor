# Approach — Legal AI Pipeline

## Architecture Overview

The pipeline is split into four cleanly separated modules, each with a single responsibility:

```
Raw Documents
     │
     ▼
┌─────────────┐
│  processor  │  3-tier OCR cleaning + LLM-assisted structured extraction
└──────┬──────┘
       │ ProcessedDocument list
       ▼
┌─────────────┐
│  retriever  │  Chunk → embed (sentence-transformers) → FAISS index
└──────┬──────┘
       │ DocumentIndex
       ▼
┌─────────────┐
│  generator  │  Multi-query RAG → Groq → grounded draft + evidence manifest
└──────┬──────┘
       │ DraftOutput
       ▼
┌─────────────┐
│   learner   │  Edit analysis → StyleGuide → inject rules → score improvement
└─────────────┘
```

---

## Stage 1: Document Processing

### OCR Cleaning — 3-Tier Generalized Approach

The original approach used hardcoded regex rules for known OCR artifacts (e.g. `F1orida` → `Florida`). This was replaced with a three-tier system that generalizes to any document type, font, or OCR engine — not just the specific errors present in the provided sample documents.

**Tier 1 — Fast Rule-Based Corrections**
High-confidence regex patterns that fire only when context is unambiguous — for example, a digit `O` surrounded by other digits is almost certainly a zero. These are cheap, deterministic, and handle the bulk of common artifacts. Applied across all documents in milliseconds with no API cost.

**Tier 2 — LLM-Based Detection**
The cleaned text from Tier 1 is sent to Groq with a prompt asking: *"identify any words or tokens that look like OCR errors and suggest corrections with confidence scores."* This catches noise patterns that no fixed rule set could anticipate — unusual character substitutions, locale-specific OCR artifacts, or errors in uncommon proper nouns. The LLM reasons from context rather than pattern-matching against a fixed list.

**Tier 3 — Validation**
LLM suggestions are filtered before application. Only corrections that produce real English words, valid identifiers, or context-consistent strings are accepted. This prevents the LLM from "helpfully" changing something that was actually correct.

**Tradeoff — Tier 2 API cost:** Each document processed through Tier 2 makes one additional Groq API call. For this four-document case this is negligible. At production scale (hundreds of cases, 4-6 documents each), this adds 400-600 extra LLM calls per 100 cases. Mitigation options include: (a) using a smaller/cheaper model for Tier 2 detection only (e.g. `llama-3.1-8b-instant` instead of `llama-3.3-70b-versatile`), (b) making Tier 2 optional via a config flag (`USE_LLM_OCR_CLEANING=false` for clean documents), or (c) batching multiple documents into a single Tier 2 call.

**Structured Extraction:** Each document type has a tailored JSON extraction prompt sent to Groq. The prompts specify exact output schemas so downstream code can rely on consistent field names. JSON parse errors are handled gracefully — the raw text is saved as a fallback field.

---

## Stage 2: Retrieval

**Chunking:** Text is split into 400-character windows with 80-character overlap. Overlap prevents losing context at chunk boundaries. Breaks are attempted at newlines/spaces to avoid cutting mid-sentence. The chunker guarantees forward progress on every iteration to prevent infinite loops on short or repetitive text.

**Embedding:** `all-MiniLM-L6-v2` via sentence-transformers. Chosen for its speed/quality balance (384 dims, ~80MB, runs locally with no API dependency). Embeddings are L2-normalised so FAISS inner-product equals cosine similarity.

**Index:** FAISS `IndexFlatIP` (flat, exact, inner-product). No approximate indexing needed at this data scale (339 chunks across 4 documents). For hundreds of cases (thousands of documents), `IndexIVFFlat` with clustering would give sub-millisecond retrieval.

**Retrieval:** Each draft type has a hand-crafted list of 8-10 targeted queries covering every aspect of the output. Multiple queries are run and results are deduplicated by (doc_id, chunk_index). This ensures comprehensive evidence coverage across all source documents, not just the most obvious semantic match.

**Evidence traceability:** Every retrieved chunk carries `doc_id` and `chunk_index`. The generator injects these as `[Source: X, chunk N]` citations in its prompt, and the final output file includes a full evidence manifest so every claim is inspectable.

---

## Stage 3: Draft Generation

Each draft type has:
1. A **retrieval query plan** (list of targeted queries in `_RETRIEVAL_QUERIES`)
2. A **prompt template** that specifies exact section structure

The system prompt instructs the LLM to:
- Base every claim only on the provided evidence blocks
- Write "NOT FOUND IN SOURCE DOCUMENTS" rather than fabricate
- Flag attorney action items with "ACTION REQUIRED:"
- Include inline `[Source: doc_id, chunk N]` citations

**Hallucination control:** Grounding is enforced at the prompt level — the LLM is given the retrieved evidence and told explicitly what it may and may not assert. This is not a formal guarantee, but combined with the evidence manifest it makes unsupported claims easy to spot during attorney review.

---

## Stage 4: Improvement from Operator Edits

**Pattern Extraction:** Each before/after edit pair (plus the operator-provided `key_edits` list) is sent to Groq with a prompt asking for *specific, testable, generalizable rules* — not observations about this specific case, but concrete instructions that apply to any future draft. The prompt explicitly asks for formatting details, content organization rules, flagging conventions, and citation requirements. This produces 8-12 named `StylePattern` objects per edit pair (23 total across two edit pairs in the sample run).

**StyleGuide:** Patterns from all edit pairs are deduplicated by name and stored in `style_guide.json`. This file persists between runs — the guide grows incrementally as more operator edit pairs are added over time.

**Injection:** Style rules are prepended to every generation prompt as a clearly labeled `LEARNED STYLE RULES` block. The LLM is instructed to apply them across every section of the output.

**Measuring Improvement:** A `document_checklist` draft is generated twice — once without rules (baseline) and once with all learned rules injected. Quality is assessed on a 10-dimension rubric, each scored 0-10 (max 100):

| Dimension | What it measures |
|---|---|
| Section structure | Labeled headers, logical organization |
| Instrument numbers | Recording IDs and dates on liens |
| Action item flagging | Urgent items clearly marked |
| Prioritization | Actions sorted by urgency |
| Cross-document synthesis | Information woven from multiple sources |
| Completeness | All key facts, contacts, and amounts present |
| Reviewer notes | Actionable attorney guidance |
| Citation quality | Claims attributed with source markers |
| Metadata & context | Location, date, case number in header |
| Readability | Easy to scan and act on |

In the sample run this produced a measurable delta of **+26 points** (baseline 66/100 → improved 92/100).

The `document_checklist` type was specifically chosen for the demonstration because it does *not* appear in the training edit pairs (which cover `title_review_summary` and `case_status_memo`). Improvement on an unseen draft type demonstrates genuine pattern generalization, not memorization.

**Known limitation — LLM-as-judge bias:** The scoring is performed by Groq evaluating its own output. This is a well-established technique, but it carries a known self-preference bias: the judge model tends to score outputs higher when they follow the exact style instructions it was given. The +26 delta therefore reflects a combination of genuine quality improvement and the judge recognizing its own applied patterns. A more rigorous evaluation would use human annotators or a held-out model as the judge. This tradeoff is accepted here in exchange for a fully automated, reproducible evaluation with no labeled ground truth required.

---

## Assumptions and Tradeoffs

| Decision | Rationale |
|---|---|
| Groq + llama-3.3-70b-versatile | Fast inference, strong instruction-following, no rate limits for this use case |
| llama-3.1-8b-instant for Tier 2 OCR (recommended) | Smaller model sufficient for error detection; reduces cost at scale |
| sentence-transformers local embeddings | No external API dependency; reviewer can run fully offline after model download |
| FAISS flat index | Exact search, simple setup, appropriate for <10K chunks |
| Multi-query retrieval | Comprehensive evidence coverage; more token-efficient than retrieving all chunks |
| LLM-as-judge scoring | Consistent rubric; avoids needing ground-truth labeled data; bias acknowledged |
| 3-tier OCR cleaning | Generalizes to any document type; Tier 1 handles bulk cheaply, Tier 2 catches edge cases |

---

## Scalability Path

To handle hundreds of cases:
- Replace per-case FAISS index with a persistent vector store (Chroma, Weaviate, Pinecone) keyed by `case_number`
- Add a caching layer for LLM extraction results (avoid re-processing unchanged documents)
- Switch from flat FAISS to IVF clustering for sub-millisecond retrieval at scale
- Make Tier 2 OCR cleaning optional via config flag — clean documents (emails, court orders) rarely need it
- Use a smaller model (e.g. `llama-3.1-8b-instant`) for Tier 2 OCR detection to reduce per-document cost
- Expose a REST API (FastAPI) — the four pipeline stages map cleanly to `/process`, `/index`, `/generate`, `/learn` endpoints
- Move style guide from a flat JSON file to a versioned database table so pattern history is preserved across deployments
