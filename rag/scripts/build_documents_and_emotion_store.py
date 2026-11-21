"""
Skrip Eksekusi Langkah 2:
1. Membangun dataset dokumen siap index (19.778 buku eligible, 6 dikecualikan).
2. Membangun Emotion Companion Store lengkap (19.784 buku).
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.document_builder import build_baseline_document
from rag.app.services.emotion_store import EmotionCompanionStore, EXPECTED_EMOTIONS

DATASET_PATH = BASE_DIR / "data_pipeline" / "enrichment" / "Fixed_Genre_Themes_GoodReads_Emotion.csv"
OUTPUT_CORPUS_CSV = BASE_DIR / "rag" / "vector_store" / "eligible_documents.csv"


def run_step_2():
    print("=" * 65)
    print("LANGKAH 2: DOCUMENT BUILDER & EMOTION COMPANION STORE")
    print("=" * 65)

    df = pd.read_csv(DATASET_PATH, keep_default_na=False)
    total_books = len(df)
    print(f"Dataset sumber: {DATASET_PATH.name} ({total_books:,} baris)")

    # 1. Emotion Companion Store (Seluruh 19.784 buku)
    print("\n1. Membangun Emotion Companion Store...")
    store = EmotionCompanionStore()
    stored_count = store.build_from_dataframe(df)
    print(f"   [SUKSES] Tersimpan {stored_count:,} / {total_books:,} record emosi di:")
    print(f"   -> {store.store_path}")

    # Uji verifikasi lookup O(1)
    sample_id = str(df.iloc[0]["ID"])
    sample_vec = store.get_vector(sample_id)
    print(f"   Verifikasi lookup ID {sample_id} ({df.iloc[0]['Title']}):")
    print(f"   -> {sample_vec}")

    # 2. Document Builder (Eligible 19.778 buku)
    print("\n2. Memproses Dokumen Baseline untuk Retrieval...")
    doc_texts = []
    e5_passages = []
    bge_passages = []
    eligible_rows = []
    excluded_count = 0

    for idx, row in df.iterrows():
        doc_t, e5_p, bge_p, meta = build_baseline_document(row.to_dict())
        if doc_t is not None:
            doc_texts.append(doc_t)
            e5_passages.append(e5_p)
            bge_passages.append(bge_p)
            # Simpan baris yang layak
            r_dict = row.to_dict()
            r_dict["document_text"] = doc_t
            r_dict["e5_passage_text"] = e5_p
            r_dict["bge_passage_text"] = bge_p
            eligible_rows.append(r_dict)
        else:
            excluded_count += 1

    df_eligible = pd.DataFrame(eligible_rows)
    OUTPUT_CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_eligible.to_csv(OUTPUT_CORPUS_CSV, index=False)

    print(f"   [SUKSES] Dokumen eligible: {len(df_eligible):,} buku")
    print(f"   [INFO] Dokumen dikecualikan: {excluded_count:,} buku")
    print(f"   -> File korpus siap index: {OUTPUT_CORPUS_CSV}")

    # Cetak cuplikan dokumen pertama
    print("\n" + "=" * 45)
    print("CUPLIKAN DOKUMEN BASELINE (CONTOH 1)")
    print("=" * 45)
    print(f"[Document Text untuk BM25 & BGE]:\n{doc_texts[0][:250]}...\n")
    print(f"[E5 Passage Text dengan Prefix]:\n{e5_passages[0][:250]}...\n")
    print("=" * 45)


if __name__ == "__main__":
    run_step_2()
