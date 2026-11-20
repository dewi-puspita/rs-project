"""
Script untuk melakukan Metadata Enrichment pada dataset buku bersih (books_clean.csv)
menggunakan dataset eksternal Goodreads 100k (GoodReads_100k_books.csv).

Kolom yang ditambahkan:
- Genre (string daftar genre yang dipisahkan koma)
- genre_source (metode pencocokan: kaggle_id, kaggle_isbn13, kaggle_isbn, kaggle_title_author, atau none)

Strategi Pencocokan Bertingkat:
1. Tahap 1: Pencocokan berbasis Goodreads Book ID (diekstrak dari kolom link, akurasi 100%)
2. Tahap 2: Pencocokan berbasis ISBN13
3. Tahap 3: Pencocokan berbasis ISBN10/ISBN
4. Tahap 4: Pencocokan berbasis Title + Author yang dinormalisasi (Fallback)

Output:
- Enrichment/books_enriched.csv
- reports/enrichment_report.json & reports/enrichment_report.md
"""

import json
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = BASE_DIR / "Pre-processing" / "Cleaned_Teenage_GoodReads_Emotion.csv"
# Fallback jika nama file books_clean.csv
if not CLEAN_DATA_PATH.exists():
    CLEAN_DATA_PATH = BASE_DIR / "Pre-processing" / "books_clean.csv"
KAGGLE_DATA_PATH = BASE_DIR / "GoodReads_100k_books.csv"
OUTPUT_ENRICHED_PATH = BASE_DIR / "Enrichment" / "books_enriched.csv"
REPORT_JSON_PATH = BASE_DIR / "reports" / "enrichment_report.json"
REPORT_MD_PATH = BASE_DIR / "reports" / "enrichment_report.md"


def clean_str(val) -> str:
    """Membersihkan string untuk pencocokan title dan author."""
    if pd.isna(val):
        return ""
    text = str(val).strip().lower()
    return re.sub(r"\s+", " ", text)


def clean_isbn(val) -> str:
    """Membersihkan nilai ISBN dari tanda hubung dan format float."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("-", "").strip()
    return s


def run_enrichment():
    print("=" * 65)
    print("METADATA ENRICHMENT: GOODREADS 100K -> BOOKS_CLEAN")
    print("=" * 65)

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"File data bersih belum ada: {CLEAN_DATA_PATH}")

    if not KAGGLE_DATA_PATH.exists():
        raise FileNotFoundError(f"File Kaggle belum ditemukan: {KAGGLE_DATA_PATH}")

    # 1. Baca dataset utama
    print(f"\n1. Membaca dataset utama: {CLEAN_DATA_PATH.name}")
    df_books = pd.read_csv(CLEAN_DATA_PATH, keep_default_na=False)
    total_books = len(df_books)
    print(f"   Total buku yang akan di-enrich: {total_books:,}")

    # 2. Baca dataset Kaggle
    print(f"\n2. Membaca dataset Kaggle: {KAGGLE_DATA_PATH.name}")
    usecols = ["link", "isbn", "isbn13", "title", "author", "genre"]
    df_kaggle = pd.read_csv(KAGGLE_DATA_PATH, usecols=usecols, keep_default_na=False, low_memory=False)
    print(f"   Total baris di dataset Kaggle: {len(df_kaggle):,}")

    # 3. Ekstraksi Goodreads Book ID dari kolom link
    print("\n3. Mengekstrak Goodreads ID dan membangun index pencarian...")
    # Pola link: https://goodreads.com/book/show/1001053.Between_Two_Fires atau /show/1001053-title
    df_kaggle["book_id"] = df_kaggle["link"].str.extract(r"/show/(\d+)")[0].fillna("")
    df_kaggle["isbn_clean"] = df_kaggle["isbn"].apply(clean_isbn)
    df_kaggle["isbn13_clean"] = df_kaggle["isbn13"].apply(clean_isbn)
    df_kaggle["title_norm"] = df_kaggle["title"].apply(clean_str)
    df_kaggle["author_norm"] = df_kaggle["author"].apply(clean_str)
    df_kaggle["genre_clean"] = df_kaggle["genre"].str.strip()

    # Bangun dictionary lookup untuk O(1) matching cepat
    lookup_by_id = {}
    lookup_by_isbn13 = {}
    lookup_by_isbn = {}
    lookup_by_ta = {}

    for _, row in df_kaggle.iterrows():
        g = row["genre_clean"]
        if not g:
            continue  # Lewati jika genre kosong di Kaggle

        # Index ID (Goodreads ID)
        bid = row["book_id"]
        if bid and bid not in lookup_by_id:
            lookup_by_id[bid] = g

        # Index ISBN13
        isbn13 = row["isbn13_clean"]
        if isbn13 and isbn13 not in ["0", "9780000000000"] and isbn13 not in lookup_by_isbn13:
            lookup_by_isbn13[isbn13] = g

        # Index ISBN biasa
        isbn = row["isbn_clean"]
        if isbn and isbn not in ["0"] and isbn not in lookup_by_isbn:
            lookup_by_isbn[isbn] = g

        # Index Title + Author
        t = row["title_norm"]
        a = row["author_norm"]
        if t and a:
            ta_key = f"{t}___{a}"
            if ta_key not in lookup_by_ta:
                lookup_by_ta[ta_key] = g

    print(f"   -> Entri dengan Genre via Goodreads ID : {len(lookup_by_id):,}")
    print(f"   -> Entri dengan Genre via ISBN13       : {len(lookup_by_isbn13):,}")
    print(f"   -> Entri dengan Genre via ISBN         : {len(lookup_by_isbn):,}")
    print(f"   -> Entri dengan Genre via Title+Author : {len(lookup_by_ta):,}")

    # 4. Pencocokan Bertingkat ke books_clean
    print("\n4. Mencocokkan genre ke dataset buku...")
    matched_genres = []
    genre_sources = []

    matched_id = 0
    matched_isbn13 = 0
    matched_isbn = 0
    matched_ta = 0
    unmatched = 0

    clean_isbns = df_books["ISBN"].apply(clean_isbn).tolist()
    clean_ids = df_books["ID"].astype(str).str.strip().tolist()
    clean_titles = df_books["Title"].apply(clean_str).tolist()
    clean_authors = df_books["Author"].apply(clean_str).tolist()

    for i in range(total_books):
        book_id = clean_ids[i]
        isbn = clean_isbns[i]
        ta_key = f"{clean_titles[i]}___{clean_authors[i]}"

        genre_found = ""
        src = "none"

        # Tier 1: Cek via Goodreads Book ID
        if book_id in lookup_by_id:
            genre_found = lookup_by_id[book_id]
            src = "kaggle_id"
            matched_id += 1
        # Tier 2: Cek via ISBN13
        elif isbn in lookup_by_isbn13:
            genre_found = lookup_by_isbn13[isbn]
            src = "kaggle_isbn13"
            matched_isbn13 += 1
        # Tier 3: Cek via ISBN biasa
        elif isbn in lookup_by_isbn:
            genre_found = lookup_by_isbn[isbn]
            src = "kaggle_isbn"
            matched_isbn += 1
        # Tier 4: Fallback via Title + Author
        elif ta_key in lookup_by_ta:
            genre_found = lookup_by_ta[ta_key]
            src = "kaggle_title_author"
            matched_ta += 1
        else:
            unmatched += 1

        matched_genres.append(genre_found)
        genre_sources.append(src)

    # Tambahkan ke DataFrame
    df_books["Genre"] = matched_genres
    df_books["genre_source"] = genre_sources

    # 5. Simpan file hasil enrichment
    OUTPUT_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_books.to_csv(OUTPUT_ENRICHED_PATH, index=False)
    print(f"\n[SUKSES] Dataset enriched disimpan di: {OUTPUT_ENRICHED_PATH}")

    total_matched = matched_id + matched_isbn13 + matched_isbn + matched_ta
    match_rate = (total_matched / total_books) * 100

    print("\n" + "=" * 45)
    print("HASIL STATISTIK ENRICHMENT")
    print("=" * 45)
    print(f"Total Buku               : {total_books:,}")
    print(f"Cocok via Goodreads ID   : {matched_id:,} ({matched_id/total_books*100:.2f}%)")
    print(f"Cocok via ISBN13         : {matched_isbn13:,} ({matched_isbn13/total_books*100:.2f}%)")
    print(f"Cocok via ISBN           : {matched_isbn:,} ({matched_isbn/total_books*100:.2f}%)")
    print(f"Cocok via Title + Author : {matched_ta:,} ({matched_ta/total_books*100:.2f}%)")
    print("-" * 45)
    print(f"Total Genre Terisi       : {total_matched:,} ({match_rate:.2f}%)")
    print(f"Belum Terisi (Unmatched) : {unmatched:,} ({100 - match_rate:.2f}%)")
    print("=" * 45)

    # 6. Buat Laporan JSON dan Markdown
    report_dict = {
        "dataset_clean": str(CLEAN_DATA_PATH.name),
        "dataset_kaggle": str(KAGGLE_DATA_PATH.name),
        "output_enriched": str(OUTPUT_ENRICHED_PATH.name),
        "total_books": total_books,
        "matched_by_id": matched_id,
        "matched_by_isbn13": matched_isbn13,
        "matched_by_isbn": matched_isbn,
        "matched_by_title_author": matched_ta,
        "total_matched": total_matched,
        "unmatched": unmatched,
        "match_rate_percentage": round(match_rate, 2),
    }

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    md_report = f"""# Laporan Metadata Enrichment (GoodReads 100k)

- **Dataset Bersih**: `{CLEAN_DATA_PATH.name}`
- **Dataset Eksternal**: `{KAGGLE_DATA_PATH.name}`
- **Output File**: `{OUTPUT_ENRICHED_PATH.name}`

## 1. Statistik Pencocokan Genre
- **Total Buku**: {total_books:,}
- **Cocok via Goodreads Book ID**: {matched_id:,} ({matched_id/total_books*100:.2f}%)
- **Cocok via ISBN13**: {matched_isbn13:,} ({matched_isbn13/total_books*100:.2f}%)
- **Cocok via ISBN**: {matched_isbn:,} ({matched_isbn/total_books*100:.2f}%)
- **Cocok via Title + Author**: {matched_ta:,} ({matched_ta/total_books*100:.2f}%)
- **Total Genre Berhasil Terisi**: **{total_matched:,} ({match_rate:.2f}%)**
- **Belum Ada Genre**: {unmatched:,} ({100 - match_rate:.2f}%)

## 2. Kepatuhan Spesifikasi
- **Transparansi**: Kolom `genre_source` mencatat asal pencocokan untuk setiap record.
- **Integritas Data**: Nilai `EmotionVector` asli tidak disentuh sama sekali.
- **Tanpa Fakta Palsu**: Buku yang tidak ditemukan genrenya di Kaggle dibiarkan bernilai string kosong.
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\nLaporan enrichment tersimpan di:")
    print(f" - {REPORT_JSON_PATH}")
    print(f" - {REPORT_MD_PATH}")


if __name__ == "__main__":
    run_enrichment()
