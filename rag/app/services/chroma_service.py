"""
Modul ChromaDB Indexing Service.
Sesuai Rantug v2 (Bab 5 & Bab 6.2).

Tugas:
1. Membangun koleksi ChromaDB persisten terpisah:
   - rag/vector_store/chroma_e5_large/ (books_multilingual_e5_large)
   - rag/vector_store/chroma_bge_large/ (books_bge_large_en_v1_5)
2. Batching ramah memori (batch_size=8) dengan progress logging.
3. Menyimpan metadata lengkap: book_id, isbn, title, author, average_rating, age, genre, themes.
"""

from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_E5_DIR = BASE_DIR / "vector_store" / "chroma_e5_large"
CHROMA_BGE_DIR = BASE_DIR / "vector_store" / "chroma_bge_large"
if not CHROMA_E5_DIR.parent.exists():
    CHROMA_E5_DIR = BASE_DIR / "rag" / "vector_store" / "chroma_e5_large"
    CHROMA_BGE_DIR = BASE_DIR / "rag" / "vector_store" / "chroma_bge_large"

COLLECTION_NAME_E5 = "books_multilingual_e5_large"
COLLECTION_NAME_BGE = "books_bge_large_en_v1_5"


def get_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


class ChromaIndexer:
    def __init__(self, model_name: str, persist_dir: Path, collection_name: str):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = get_chroma_client(persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"Memuat model embedding: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.model.max_seq_length = 512

    def index_dataframe(self, df: pd.DataFrame, text_column: str, batch_size: int = 8, start_idx: int = 0):
        self.load_model()
        total_rows = len(df)
        print(f"\nMemulai indexing ke ChromaDB: {self.collection_name}")
        print(f"Total buku: {total_rows:,} | Batch size: {batch_size} | Start index: {start_idx:,}")

        # Cek data yang sudah ada di koleksi
        existing_count = self.collection.count()
        print(f"Jumlah entri saat ini di ChromaDB: {existing_count:,}")

        for i in range(start_idx, total_rows, batch_size):
            batch_df = df.iloc[i : i + batch_size]

            ids = [str(r["ID"]).strip() for _, r in batch_df.iterrows()]
            texts = batch_df[text_column].tolist()
            
            # Siapkan metadata
            metadatas = []
            for _, r in batch_df.iterrows():
                metadatas.append({
                    "book_id": str(r["ID"]).strip(),
                    "isbn": str(r.get("ISBN", "")).strip(),
                    "title": str(r.get("Title", "")).strip() or "Judul tidak tersedia",
                    "author": str(r.get("Author", "")).strip() or "Penulis tidak tersedia",
                    "average_rating": float(r.get("Average_Rating", 0.0)) if str(r.get("Average_Rating", "")).strip() else 0.0,
                    "age": float(r.get("Age", 0.0)) if str(r.get("Age", "")).strip() else 0.0,
                    "genre": str(r.get("Genre", "")).strip(),
                    "themes": str(r.get("Themes", "")).strip()
                })

            # Hitung embedding
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            # Simpan ke ChromaDB (upsert untuk menghindari duplikasi)
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            if (i + len(batch_df)) % 100 == 0 or (i + len(batch_df)) == total_rows:
                pct = ((i + len(batch_df)) / total_rows) * 100
                print(f"[{i + len(batch_df):,}/{total_rows:,}] ({pct:.1f}%) buku berhasil di-index ke ChromaDB...")

        print(f"\n[SELESAI] Koleksi {self.collection_name} sekarang memiliki {self.collection.count():,} buku ter-index!")
