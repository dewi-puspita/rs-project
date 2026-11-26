"""
MODUL 1: HYBRID RAG ENGINE (Pencarian Semantik E5 + Kata Kunci BM25 + Fusi RRF)
Fungsi Inti:
1. Sparse Retrieval: Mencari buku berdasarkan kata kunci leksikal (BM25Okapi).
2. Dense Retrieval : Mencari buku berdasarkan makna semantik (Multilingual E5 Large).
3. RRF Fusion      : Menggabungkan peringkat kedua metode dengan formula: RRF_Score(d) = sum(1 / (k + rank(d))) dengan k = 60.
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
ELIGIBLE_DOCS_PATH = VECTOR_STORE_DIR / "eligible_documents.csv"
BM25_INDEX_PATH = VECTOR_STORE_DIR / "bm25_index.pkl"
BM25_IDS_PATH = VECTOR_STORE_DIR / "bm25_book_ids.json"
CHROMA_E5_DIR = VECTOR_STORE_DIR / "chroma_e5_large"
COLLECTION_NAME_E5 = "books_multilingual_e5_large"


class HybridRAGEngine:
    @staticmethod
    def build_document_text(title: str, genre: str, themes: str, description: str) -> str:
        """
        [Document Builder]
        Menyusun representasi teks dokumen gabungan untuk korpus retrieval
        sesuai format standar: 'Title: ... | Genre: ... | Themes: ... | Description: ...'
        """
        parts = []
        if title and str(title).strip():
            parts.append(f"Title: {str(title).strip()}")
        if genre and str(genre).strip():
            parts.append(f"Genre: {str(genre).strip()}")
        if themes and str(themes).strip():
            parts.append(f"Themes: {str(themes).strip()}")
        if description and str(description).strip():
            parts.append(f"Description: {str(description).strip()}")
        return " | ".join(parts)

    @staticmethod
    def build_e5_passage_text(document_text: str) -> str:
        """Menambahkan prefix asymmetric 'passage: ' sesuai spesifikasi model Multilingual E5."""
        return f"passage: {document_text.strip()}"

    def __init__(self):
        print("[RAG Engine] Menginisialisasi komponen Hybrid Retrieval...")
        
        # 1. Muat metadata buku
        self.df_meta = pd.read_csv(ELIGIBLE_DOCS_PATH, keep_default_na=False).set_index("ID")
        
        # 2. Muat Indeks BM25
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_model: BM25Okapi = pickle.load(f)
        with open(BM25_IDS_PATH, "r", encoding="utf-8") as f:
            self.bm25_book_ids: List[str] = json.load(f)

        # 3. Muat Koleksi Dense ChromaDB E5
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_E5_DIR))
        self.chroma_collection = self.chroma_client.get_collection(COLLECTION_NAME_E5)

        # 4. Model Embedding E5 (CPU Mode)
        self.e5_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
        print("[RAG Engine] Siap digunakan (19.778 buku terindeks).")

    def search_sparse_bm25(self, query_keywords_en: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Mencari buku menggunakan algoritma BM25Okapi."""
        tokens = query_keywords_en.lower().strip().split()
        if not tokens:
            return []
        
        scores = self.bm25_model.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        hits = []
        for rank, idx in enumerate(top_indices, start=1):
            hits.append({
                "book_id": str(self.bm25_book_ids[idx]),
                "rank": rank,
                "bm25_score": round(float(scores[idx]), 4)
            })
        return hits

    def search_dense_e5(self, query_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Mencari buku menggunakan Dense Vector Multilingual E5 (prefix 'query: ')."""
        formatted_query = f"query: {query_id.strip()}"
        emb = self.e5_model.encode([formatted_query], normalize_embeddings=True).tolist()
        
        res = self.chroma_collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        hits = []
        for rank, (bid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            hits.append({
                "book_id": str(bid),
                "rank": rank,
                "dense_score": round(1.0 - dist, 4)
            })
        return hits

    @staticmethod
    def apply_rrf_fusion(dense_hits: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], k: int = 60, top_candidates: int = 20) -> List[Dict[str, Any]]:
        """Menggabungkan peringkat Dense dan Sparse dengan Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for item in dense_hits:
            bid = item["book_id"]
            dense_ranks[bid] = item["rank"]
            rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k + item["rank"]))

        for item in bm25_hits:
            bid = item["book_id"]
            bm25_ranks[bid] = item["rank"]
            rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k + item["rank"]))

        sorted_books = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_candidates]
        
        results = []
        for rank, (bid, score) in enumerate(sorted_books, start=1):
            results.append({
                "book_id": bid,
                "rrf_rank": rank,
                "rrf_score": score,
                "dense_rank": dense_ranks.get(bid),
                "bm25_rank": bm25_ranks.get(bid)
            })
        return results

    def search(self, query_id: str, keywords_en: Optional[str] = None, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Pencarian Hybrid Retrieval End-to-End.
        
        Args:
            query_id: Kueri asli Bahasa Indonesia dari pengguna.
            keywords_en: Kata kunci Bahasa Inggris untuk BM25 (jika None, menggunakan query_id).
            top_k: Jumlah kandidat hasil retrieval yang diinginkan.
        """
        sparse_query = keywords_en if keywords_en else query_id
        
        # 1. Jalankan Dense & Sparse Search
        dense_hits = self.search_dense_e5(query_id, top_k=top_k)
        bm25_hits = self.search_sparse_bm25(sparse_query, top_k=top_k)
        
        # 2. Fusi RRF
        fused = self.apply_rrf_fusion(dense_hits, bm25_hits, k=60, top_candidates=top_k)
        
        # 3. Lengkapi metadata lengkap buku
        enriched = []
        for c in fused:
            bid_str = c["book_id"]
            bid_int = int(bid_str) if bid_str.isdigit() else None
            meta = self.df_meta.loc[bid_int] if bid_int is not None and bid_int in self.df_meta.index else {}
            
            enriched.append({
                "book_id": bid_str,
                "title": meta.get("Title", "Judul tidak tersedia") if isinstance(meta, pd.Series) else "Judul tidak tersedia",
                "author": meta.get("Author", "Penulis tidak tersedia") if isinstance(meta, pd.Series) else "Penulis tidak tersedia",
                "genre": meta.get("Genre", "") if isinstance(meta, pd.Series) else "",
                "themes": meta.get("Themes", "") if isinstance(meta, pd.Series) else "",
                "average_rating": float(meta.get("Average_Rating", 0.0)) if isinstance(meta, pd.Series) and str(meta.get("Average_Rating", "")).strip() else 0.0,
                "description": meta.get("Description", "") if isinstance(meta, pd.Series) else "",
                "rrf_rank": c["rrf_rank"],
                "rrf_score": c["rrf_score"],
                "dense_rank": c["dense_rank"],
                "bm25_rank": c["bm25_rank"]
            })
        return enriched
