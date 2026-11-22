"""
Skrip untuk membangun Dense Index ChromaDB untuk BGE Large English v1.5.
Sesuai Rantug v2 (Bab 5.2 & Bab 6.2).
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.chroma_service import (
    ChromaIndexer,
    CHROMA_BGE_DIR,
    COLLECTION_NAME_BGE
)

ELIGIBLE_DOCS_PATH = BASE_DIR / "rag" / "vector_store" / "eligible_documents.csv"


def run_build_bge_index():
    print("=" * 65)
    print("TAHAP 5C: MEMBANGUN DENSE INDEX CHROMADB (BGE LARGE ENGLISH V1.5)")
    print("=" * 65)

    if not ELIGIBLE_DOCS_PATH.exists():
        raise FileNotFoundError(f"File korpus belum ada: {ELIGIBLE_DOCS_PATH}")

    df = pd.read_csv(ELIGIBLE_DOCS_PATH, keep_default_na=False)
    print(f"Dataset: {len(df):,} buku eligible")

    indexer = ChromaIndexer(
        model_name="BAAI/bge-large-en-v1.5",
        persist_dir=CHROMA_BGE_DIR,
        collection_name=COLLECTION_NAME_BGE
    )

    # Menggunakan kolom bge_passage_text (murni Title + Author + Description tanpa prefix)
    indexer.index_dataframe(df, text_column="bge_passage_text", batch_size=8)


if __name__ == "__main__":
    run_build_bge_index()
