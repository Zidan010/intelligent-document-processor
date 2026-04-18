"""
retriever.py — Retrieval Module

Responsibilities:
  1. Chunk processed document text into overlapping windows
  2. Embed chunks with sentence-transformers (all-MiniLM-L6-v2)
  3. Build a FAISS index for fast similarity search
  4. Retrieve the top-K most relevant chunks for a query
  5. Return chunks with full source attribution (doc_id, chunk index, text)

This makes every generated output grounded and traceable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL

if TYPE_CHECKING:
    from pipeline.processor import ProcessedDocument


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved text chunk with provenance."""
    doc_id:      str
    doc_type:    str
    chunk_index: int
    text:        str
    score:       float   # cosine similarity (higher = more relevant)

    def citation(self) -> str:
        """Human-readable source reference."""
        return f"[Source: {self.doc_id}, chunk {self.chunk_index}]"


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level windows.
    Tries to break on newlines/spaces to avoid cutting mid-word.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to extend to end of line or word
        if end < len(text):
            newline = text.rfind("\n", start, end)
            space   = text.rfind(" ",  start, end)
            break_at = max(newline, space)
            if break_at > start:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap
        start = next_start if next_start > start else start + 1
    return chunks


# ── DocumentIndex ─────────────────────────────────────────────────────────────

class DocumentIndex:
    """
    In-memory FAISS index over all document chunks.

    Usage:
        index = DocumentIndex()
        index.build(processed_docs)
        chunks = index.retrieve("HOA lis pendens action required")
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"  [RETRIEVER] Loading embedding model '{model_name}' …")
        self._model    = SentenceTransformer(model_name)
        self._index    = None          # faiss.IndexFlatIP
        self._metadata: list[dict]  = []   # parallel list to FAISS vectors
        self._built    = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, docs: list[ProcessedDocument]) -> None:
        """Chunk all documents, embed, and build FAISS index."""
        all_chunks: list[str]  = []
        all_meta:   list[dict] = []

        for doc in docs:
            chunks = _chunk_text(doc.clean_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "doc_id":      doc.doc_id,
                    "doc_type":    doc.doc_type,
                    "chunk_index": i,
                    "text":        chunk,
                })

        print(f"  [RETRIEVER] Embedding {len(all_chunks)} chunks …")
        embeddings = self._model.encode(
            all_chunks,
            normalize_embeddings=True,   # cosine via inner product
            show_progress_bar=False,
            batch_size=64,
        ).astype("float32")

        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine
        self._index.add(embeddings)
        self._metadata = all_meta
        self._built    = True
        print(f"  [RETRIEVER] Index built: {self._index.ntotal} vectors, dim={dim}")

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-K chunks most relevant to `query`.

        Args:
            query:           Natural-language query string.
            top_k:           Number of results to return.
            filter_doc_type: If set, only return chunks from this doc_type.

        Returns:
            List of RetrievedChunk sorted by descending relevance score.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call .build(docs) first.")

        q_vec = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        # Fetch extra candidates if filtering, to have enough after filter
        fetch_k = top_k * 4 if filter_doc_type else top_k
        scores, indices = self._index.search(q_vec, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if filter_doc_type and meta["doc_type"] != filter_doc_type:
                continue
            results.append(
                RetrievedChunk(
                    doc_id=meta["doc_id"],
                    doc_type=meta["doc_type"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, dir_path: Path) -> None:
        """Persist FAISS index + metadata to disk."""
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "faiss.index"))
        (dir_path / "metadata.json").write_text(
            json.dumps(self._metadata, indent=2), encoding="utf-8"
        )
        print(f"  [RETRIEVER] Index saved → {dir_path}")

    def load(self, dir_path: Path) -> None:
        """Load a previously saved FAISS index + metadata."""
        self._index    = faiss.read_index(str(dir_path / "faiss.index"))
        self._metadata = json.loads((dir_path / "metadata.json").read_text())
        self._built    = True
        print(f"  [RETRIEVER] Index loaded ← {dir_path} ({self._index.ntotal} vectors)")


# ── Convenience: format retrieved chunks for prompt injection ────────────────

def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks as a labelled evidence block
    suitable for injection into an LLM prompt.
    """
    if not chunks:
        return "[No relevant evidence retrieved]"
    parts = []
    for c in chunks:
        parts.append(
            f"--- {c.citation()} (score={c.score:.3f}) ---\n{c.text}"
        )
    return "\n\n".join(parts)
