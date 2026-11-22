"""
Skrip untuk membangun Dense Index ChromaDB untuk Multilingual E5 Large.
Sesuai Rantug v2 (Bab 5.1 & Bab 6.2).
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.chroma_service import (
    ChromaIndexer,
    CHROMA_E5_DIR,
    COLLECTION_NAME_E5
)

ELIGIBLE_DOCS_PATH = BASE_DIR / "rag" / "vector_store" / "eligible_documents.csv"


def run_build_e5_index():
    print("=" * 65)
    print("TAHAP 5B: MEMBANGUN DENSE INDEX CHROMADB (MULTILINGUAL E5 LARGE)")
    print("=" * 65)

    if not ELIGIBLE_DOCS_PATH.exists():
        raise FileNotFoundError(f"File korpus belum ada: {ELIGIBLE_DOCS_PATH}")

    df = pd.read_csv(ELIGIBLE_DOCS_PATH, keep_default_na=False)
    print(f"Dataset: {len(df):,} buku eligible")

    indexer = ChromaIndexer(
        model_name="intfloat/multilingual-e5-large",
        persist_dir=CHROMA_E5_DIR,
        collection_name=COLLECTION_NAME_E5
    )

    # Menggunakan kolom e5_passage_text (berawalan 'passage: ')
    indexer.index_dataframe(df, text_column="e5_passage_text", batch_size=8)


if __name__ == "__main__":
    run_build_e5_index()
