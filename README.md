# Legal AI Pipeline

AI-powered document processing and draft generation system for legal case management.

Built for the Rodriguez foreclosure case (2025-FC-08891).

---

## Setup

### 1. Clone / unzip and enter the project directory

```bash
cd legal_ai_pipeline
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` embedding model (~80MB). Subsequent runs use the cached version.

### 4. Set your Groq API key

Copy `.env.example` to `.env` and fill in your key:

```bash
copy .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Place source documents

Put the four source documents in `data/documents/`:

```
data/
  documents/
    title_search_page1.txt
    title_search_page2.txt
    servicer_email.txt
    court_order.txt
  case_context.json
  sample_edits.json
```

### 6. Run the pipeline

```bash
python main.py
```

---

## Outputs

All outputs are written to `outputs/`:

| File | Contents |
|---|---|
| `processed_documents.json` | Cleaned text + structured extraction for all 4 docs |
| `faiss_index/` | Persisted FAISS vector index + metadata |
| `draft_title_review_summary.txt` | Generated Title Review Summary with evidence manifest |
| `draft_case_status_memo.txt` | Generated Case Status Memo with evidence manifest |
| `style_guide.json` | Learned operator style patterns |
| `document_checklist_baseline.txt` | Document Checklist without style rules |
| `document_checklist_improved.txt` | Document Checklist with learned style rules |
| `improvement_report.json` | Numeric quality scores + improvement delta |
| `improvement_comparison.txt` | Side-by-side human-readable comparison |

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Switch to `llama-3.1-8b-instant` for faster/cheaper runs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `CHUNK_SIZE` | 400 | Characters per retrieval chunk |
| `CHUNK_OVERLAP` | 80 | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | 5 | Chunks returned per retrieval query |

---

## Project Structure

```
legal_ai_pipeline/
├── main.py                  # Orchestrates all four stages
├── config.py                # All constants and paths
├── requirements.txt
├── .env.example
├── README.md
├── APPROACH.md
├── data/
│   ├── documents/           # Source document text files
│   ├── case_context.json
│   └── sample_edits.json
├── pipeline/
│   ├── __init__.py
│   ├── processor.py         # Stage 1: OCR cleaning + LLM extraction
│   ├── retriever.py         # Stage 2: Chunking, embedding, FAISS index
│   ├── generator.py         # Stage 3: Grounded draft generation
│   └── learner.py           # Stage 4: Edit analysis + improvement loop
└── outputs/                 # All generated files (created at runtime)
```
