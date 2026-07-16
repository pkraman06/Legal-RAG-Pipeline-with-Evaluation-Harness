"""
ingest.py
Chunks CUAD contracts and builds two retrieval indices:
  1. A sparse BM25 index (keyword matching — good for exact legal terms)
  2. A dense FAISS index over sentence-transformer embeddings (semantic matching)

Run directly to build and cache the indices to disk:
    python ingest.py

On Apple Silicon (M1/M2/M3/M4), this automatically uses the "mps" device
for embedding generation, which is noticeably faster than CPU. Falls back
to CPU automatically on other machines.
"""

import os
import pickle
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_loader import load_cuad, get_unique_contracts

INDEX_DIR = "indices"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"   # strong open embedding model
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def get_device() -> str:
    """Uses Apple Silicon's MPS backend if available (Mac M-series), else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunk_contracts(contracts: dict, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits each contract into overlapping chunks. Returns a list of dicts:
    {chunk_id, title, text}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for title, text in contracts.items():
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{title}::chunk_{i}",
                "title": title,
                "text": piece,
            })
    print(f"[ingest] Built {len(chunks)} chunks from {len(contracts)} contracts.")
    return chunks


def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


def build_dense_index(chunks, model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)
    return index, embeddings


def build_and_save_indices(max_records=300):
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("[ingest] Loading CUAD...")
    records = load_cuad(max_records=max_records)
    contracts = get_unique_contracts(records)
    chunks = chunk_contracts(contracts)

    print("[ingest] Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    device = get_device()
    print(f"[ingest] Loading embedding model: {EMBED_MODEL_NAME} (device={device})")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    print("[ingest] Building dense FAISS index...")
    faiss_index, embeddings = build_dense_index(chunks, embed_model)

    # Save everything needed to reload at query time
    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "dense.index"))

    # Also cache the QA eval records for evaluate.py
    with open(os.path.join(INDEX_DIR, "qa_records.pkl"), "wb") as f:
        pickle.dump(records, f)

    print(f"[ingest] Done. Indices saved to ./{INDEX_DIR}/")


if __name__ == "__main__":
    build_and_save_indices(max_records=300)
