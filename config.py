"""
config.py — Central configuration for the Legal AI Pipeline.
All model names, paths, and tunable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
DOCS_DIR    = DATA_DIR / "documents"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"   # change to llama-3.1-8b-instant for speed
GROQ_MAX_TOKENS = 4096

# ── Embeddings / Retrieval ───────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim
CHUNK_SIZE       = 400    # characters per chunk
CHUNK_OVERLAP    = 80     # overlap between consecutive chunks
TOP_K_RETRIEVAL  = 5      # how many chunks to retrieve per query

# ── Document source file names ───────────────────────────────────────────────
SOURCE_DOCS = [
    "title_search_page1.txt",
    "title_search_page2.txt",
    "servicer_email.txt",
    "court_order.txt",
]
