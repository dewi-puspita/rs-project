"""
Skrip untuk mengeksekusi Tahap A: Penyusunan Teks Dokumen (document_text).

Membaca dataset final:
  data_pipeline/enrichment/Fixed_Genre_Themes_GoodReads_Emotion.csv
Menyusun kolom 'document_text' sesuai format standar Bab 5.2.
Menyimpan dataset siap-index ke:
  rag/vector_store/books_ready_for_indexing.csv
"""

import sys
from pathlib import Path
import pandas as pd

# Tambahkan root path ke sys.path agar modul internal bisa diimport
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.document_builder import build_document_text

INPUT_DATA_PATH = BASE_DIR / "data_pipeline" / "enrichment" / "Fixed_Genre_Themes_GoodReads_Emotion.csv"
OUTPUT_DATA_PATH = BASE_DIR / "rag" / "vector_store" / "books_ready_for_indexing.csv"


def run_document_preparation():
    print("=" * 65)
    print("TAHAP A: PENYUSUNAN TEKS DOKUMEN BUKU (DOCUMENT BUILDER)")
    print("=" * 65)

    if not INPUT_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset input tidak ditemukan: {INPUT_DATA_PATH}")

    print(f"1. Membaca dataset dari: {INPUT_DATA_PATH.name}")
    df = pd.read_csv(INPUT_DATA_PATH, keep_default_na=False)
    total_books = len(df)
    print(f"   Total buku: {total_books:,}")

    print("\n2. Membangun field 'document_text' untuk setiap buku...")
    df["document_text"] = df.apply(lambda row: build_document_text(row.to_dict()), axis=1)

    # Validasi tidak ada document_text yang kosong
    empty_docs = (df["document_text"].str.strip() == "").sum()
    print(f"   Penyusunan teks selesai. Teks kosong: {empty_docs}")

    # Simpan dataset
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\n3. Dataset siap-index disimpan ke: {OUTPUT_DATA_PATH}")

    # Tampilkan preview 2 contoh dokumen
    print("\n" + "=" * 45)
    print("CONTOH HASIL PENYUSUNAN TEKS DOKUMEN")
    print("=" * 45)
    for i in range(2):
        row = df.iloc[i]
        print(f"\n--- Buku ID: {row['ID']} ({row['Title']}) ---")
        print(row["document_text"][:350] + "...")
    print("=" * 45)


if __name__ == "__main__":
    run_document_preparation()
