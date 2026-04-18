# Approach — Legal AI Pipeline

## Architecture Overview

The pipeline is split into four cleanly separated modules, each with a single responsibility:

```
Raw Documents
     │
     ▼
┌─────────────┐
│  processor  │  OCR cleaning + LLM-assisted structured extraction
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

**OCR Cleaning:** Rule-based regex corrections handle the most common title-search OCR artifacts: digit-as-letter substitutions (`1`→`l`, `O`→`0`) in specific known patterns (e.g. `Fi1e` → `File`, `WE11S` → `WELLS`). Rules are applied in order and are conservative — they only fire on patterns known to occur in this document type, avoiding false replacements.

**Structured Extraction:** Each document type has a tailored JSON extraction prompt sent to Groq. The prompts specify exact output schemas so downstream code can rely on consistent field names. JSON parse errors are handled gracefully — the raw text is saved as a fallback field.

**Tradeoff:** LLM-based extraction is more robust to messy formatting than regex but costs tokens. For production at scale, a hybrid approach (regex fast-path + LLM for edge cases) would be preferred.

---

## Stage 2: Retrieval

**Chunking:** Text is split into 400-character windows with 80-character overlap. Overlap prevents losing context at chunk boundaries. Breaks are attempted at newlines/spaces to avoid cutting mid-sentence.

**Embedding:** `all-MiniLM-L6-v2` via sentence-transformers. Chosen for its speed/quality balance (384 dims, ~80MB, runs locally). Embeddings are L2-normalised so FAISS inner-product equals cosine similarity.

**Index:** FAISS `IndexFlatIP` (flat, exact, inner-product). No approximate indexing needed at this data scale. For hundreds of cases (thousands of documents), `IndexIVFFlat` with clustering would be appropriate.

**Retrieval:** Each draft type has a hand-crafted list of 8-10 targeted queries covering every aspect of the output. Multiple queries are run and results are deduplicated by (doc_id, chunk_index). This ensures comprehensive evidence coverage across all source documents, not just the most obvious match.

**Evidence traceability:** Every retrieved chunk carries `doc_id` and `chunk_index`. The generator injects these as `[Source: X, chunk N]` citations in its prompt, and the final output includes a full evidence manifest.

---

## Stage 3: Draft Generation

Each draft type has:
1. A **retrieval query plan** (list of targeted queries in `_RETRIEVAL_QUERIES`)
2. A **prompt template** that specifies exact section structure

The system prompt instructs the LLM to:
- Base every claim only on the provided evidence
- Write "NOT FOUND IN SOURCE DOCUMENTS" rather than fabricate
- Flag attorney action items with "ACTION REQUIRED:"
- Include inline source citations

**Hallucination control:** Grounding is enforced at the prompt level — the LLM is given the evidence and told explicitly what it may and may not do. This is not a guarantee, but combined with the evidence manifest it makes unsupported claims easy to spot and correct.

---

## Stage 4: Improvement from Operator Edits

**Pattern Extraction:** Each before/after edit pair (plus the operator-provided `key_edits` list) is sent to Groq with a prompt that asks for *generalizable reusable rules* — not observations about this specific case, but instructions that apply to any future draft. This produces 6-10 named `StylePattern` objects per edit pair.

**StyleGuide:** Patterns from all edit pairs are deduplicated by name and stored in `style_guide.json`. This persists between runs — the guide grows as more edit pairs are added.

**Injection:** Style rules are prepended to every generation prompt as a clearly labeled `LEARNED STYLE RULES` block. The LLM is instructed to apply them across every section.

**Measuring Improvement:** A `document_checklist` draft is generated twice — once without rules (baseline) and once with. An 8-dimension rubric (section structure, instrument numbers, action flags, prioritization, cross-doc synthesis, completeness, reviewer notes, citation quality) is scored 0-3 each by Groq acting as an evaluator. The total score delta is the primary improvement metric.

The `document_checklist` type was specifically chosen for the demonstration because it does *not* appear in the training edit pairs (which cover `title_review_summary` and `case_status_memo`). A genuine improvement here demonstrates that patterns generalize across draft types.

---

## Assumptions and Tradeoffs

| Decision | Rationale |
|---|---|
| Groq + llama-3.3-70b-versatile | Fast inference, good instruction-following, no rate limits for this use case |
| sentence-transformers local embeddings | No external API dependency for embeddings; reviewer can run fully offline after model download |
| FAISS flat index | Exact search, simple setup, appropriate for <10K chunks |
| Multi-query retrieval | Ensures comprehensive evidence coverage; more token-efficient than retrieving all chunks |
| LLM-as-judge scoring | Consistent rubric; avoids needing ground-truth labeled data |
| Rule-based OCR cleaning | Predictable, inspectable, fast; covers known patterns without touching valid text |

---

## Scalability Path

To handle hundreds of cases:
- Replace per-case FAISS index with a persistent vector store (Chroma, Weaviate, Pinecone) keyed by case_number
- Add a caching layer for LLM extraction results (avoid re-processing unchanged documents)
- Switch from flat FAISS to IVF clustering for sub-millisecond retrieval at scale
- Expose a REST API (FastAPI) — the four pipeline stages map cleanly to `/process`, `/index`, `/generate`, `/learn` endpoints
- Move style guide from a flat JSON file to a versioned database table
