"""
Skrip untuk membangun indeks BM25Okapi dari 19.778 buku eligible.
Sesuai Rantug v2 (Bab 6.3).
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.bm25_service import BM25Service

ELIGIBLE_DOCS_PATH = BASE_DIR / "rag" / "vector_store" / "eligible_documents.csv"


def run_build_bm25():
    print("=" * 65)
    print("TAHAP 5A: MEMBANGUN SPARSE INDEX BM25OKAPI")
    print("=" * 65)

    if not ELIGIBLE_DOCS_PATH.exists():
        raise FileNotFoundError(f"File korpus belum ada: {ELIGIBLE_DOCS_PATH}")

    df = pd.read_csv(ELIGIBLE_DOCS_PATH, keep_default_na=False)
    total_books = len(df)
    print(f"Memuat {total_books:,} buku eligible dari {ELIGIBLE_DOCS_PATH.name}")

    book_ids = df["ID"].astype(str).tolist()
    # Menggunakan document_text murni (Title + Author + Description) tanpa prefix
    documents = df["document_text"].tolist()

    service = BM25Service()
    service.build_and_save(book_ids, documents)

    # Uji coba query cepat
    test_query = "friendship adventure teenager struggle"
    print(f"\nUji Coba Pencarian BM25 Query: '{test_query}'")
    hits = service.search(test_query, top_k=3)
    for h in hits:
        sample_title = df[df["ID"].astype(str) == h["book_id"]]["Title"].values[0]
        print(f" - Rank {h['rank']}: ID {h['book_id']} ('{sample_title}') | Skor BM25: {h['bm25_score']}")

    print("\n[SELESAI] Indeks BM25 siap digunakan!")


if __name__ == "__main__":
    run_build_bm25()
