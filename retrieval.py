"""
retrieval.py
Implements three retrieval modes over the indices built by ingest.py:
  1. dense_only     - pure semantic search via FAISS
  2. hybrid         - BM25 + dense fused with Reciprocal Rank Fusion (RRF)
  3. hybrid_rerank  - hybrid retrieval, then top candidates reranked with a
                       cross-encoder for higher precision

These three modes are what evaluate.py compares against each other.
"""

import os
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_DIR = "indices"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_device() -> str:
    """Uses Apple Silicon's MPS backend if available (Mac M-series), else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LegalRetriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.dense_index = faiss.read_index(os.path.join(index_dir, "dense.index"))

        self.device = get_device()
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=self.device)
        self._reranker = None  # lazy-loaded, only needed for hybrid_rerank

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL_NAME, device=self.device)
        return self._reranker

    def _dense_search(self, query: str, top_k: int):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, idxs = self.dense_index.search(q_emb, top_k)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))

    def _bm25_search(self, query: str, top_k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idxs]

    @staticmethod
    def _reciprocal_rank_fusion(rank_lists, k: int = 60):
        """
        rank_lists: list of ranked (idx, score) lists from different retrievers.
        RRF score for a doc = sum(1 / (k + rank)) across all lists it appears in.
        This avoids needing to normalize scores across BM25 and dense similarity,
        which are on very different scales.
        """
        fused_scores = {}
        for rank_list in rank_lists:
            for rank, (idx, _) in enumerate(rank_list):
                fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def retrieve(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5, candidate_k: int = 20):
        """
        mode: "dense_only" | "hybrid" | "hybrid_rerank"
        Returns a list of dicts: {chunk_id, title, text, score}
        """
        if mode == "dense_only":
            results = self._dense_search(query, top_k)
            ranked_idxs = [idx for idx, _ in results]

        elif mode == "hybrid":
            dense_results = self._dense_search(query, candidate_k)
            bm25_results = self._bm25_search(query, candidate_k)
            fused = self._reciprocal_rank_fusion([dense_results, bm25_results])
            ranked_idxs = [idx for idx, _ in fused[:top_k]]

        elif mode == "hybrid_rerank":
            dense_results = self._dense_search(query, candidate_k)
            bm25_results = self._bm25_search(query, candidate_k)
            fused = self._reciprocal_rank_fusion([dense_results, bm25_results])
            candidate_idxs = [idx for idx, _ in fused[:candidate_k]]

            pairs = [[query, self.chunks[idx]["text"]] for idx in candidate_idxs]
            rerank_scores = self.reranker.predict(pairs)
            reranked = sorted(
                zip(candidate_idxs, rerank_scores), key=lambda x: x[1], reverse=True
            )
            ranked_idxs = [idx for idx, _ in reranked[:top_k]]

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        return [
            {
                "chunk_id": self.chunks[idx]["chunk_id"],
                "title": self.chunks[idx]["title"],
                "text": self.chunks[idx]["text"],
            }
            for idx in ranked_idxs
        ]


if __name__ == "__main__":
    retriever = LegalRetriever()
    query = "What is the governing law of this agreement?"
    for mode in ["dense_only", "hybrid", "hybrid_rerank"]:
        print(f"\n--- mode: {mode} ---")
        results = retriever.retrieve(query, mode=mode, top_k=3)
        for r in results:
            print(f"[{r['title']}] {r['text'][:120]}...")
