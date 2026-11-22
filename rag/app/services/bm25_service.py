"""
Modul BM25 Service.
Sesuai Rantug v2 (Bab 6.3 & Bab 8.2).

Tugas:
1. Tokenisasi korpus teks bahasa Inggris (Title + Author + Description).
2. Membangun indeks BM25Okapi dari 19.778 buku eligible.
3. Menyimpan objek indeks (bm25_index.pkl) dan mapping posisi ke canonical book_id (bm25_book_ids.json).
4. Menyediakan fungsi pencarian sparse retrieval Top-K.
"""

import re
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BM25_INDEX_PATH = BASE_DIR / "vector_store" / "bm25_index.pkl"
BM25_IDS_PATH = BASE_DIR / "vector_store" / "bm25_book_ids.json"
if not BM25_INDEX_PATH.parent.exists():
    BM25_INDEX_PATH = BASE_DIR / "rag" / "vector_store" / "bm25_index.pkl"
    BM25_IDS_PATH = BASE_DIR / "rag" / "vector_store" / "bm25_book_ids.json"


def tokenize_english_text(text: str) -> List[str]:
    """
    Fungsi tokenisasi teks Inggris sederhana, konsisten, dan cepat:
    - Lowercase
    - Hapus tanda baca/karakter non-alphanumeric
    - Split whitespace
    - Filter token berkarakter tunggal atau kosong
    """
    if not text:
        return []
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    return tokens


class BM25Service:
    def __init__(self, index_path: Path = BM25_INDEX_PATH, ids_path: Path = BM25_IDS_PATH):
        self.index_path = index_path
        self.ids_path = ids_path
        self.bm25: BM25Okapi = None
        self.book_ids: List[str] = []
        if self.index_path.exists() and self.ids_path.exists():
            self.load()

    def build_and_save(self, book_ids: List[str], documents: List[str]):
        """Membangun indeks BM25 dan menyimpannya ke disk."""
        print(f"Melakukan tokenisasi {len(documents):,} dokumen korpus...")
        tokenized_corpus = [tokenize_english_text(doc) for doc in documents]
        
        print("Membangun model BM25Okapi...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.book_ids = [str(bid).strip() for bid in book_ids]

        # Simpan
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.book_ids, f)
        print(f"[SUKSES] Indeks BM25 tersimpan di: {self.index_path}")
        print(f"[SUKSES] ID mapping tersimpan di : {self.ids_path}")

    def load(self):
        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.book_ids = json.load(f)

    def search(self, query_en: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Mencari kandidat buku berbasis BM25.
        Query harus dalam bahasa Inggris.
        Returns: list of dict {'book_id': ..., 'rank': ..., 'bm25_score': ...}
        """
        if self.bm25 is None or not self.book_ids:
            self.load()

        tokenized_query = tokenize_english_text(query_en)
        if not tokenized_query:
            return []

        # Hitung skor BM25 untuk semua dokumen
        scores = self.bm25.get_scores(tokenized_query)
        
        # Ambil top_k index dokumen dengan skor tertinggi
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue  # Lewati jika tidak ada kata kunci yang cocok sama sekali
            results.append({
                "book_id": self.book_ids[idx],
                "rank": rank,
                "bm25_score": round(score, 4)
            })

        return results
