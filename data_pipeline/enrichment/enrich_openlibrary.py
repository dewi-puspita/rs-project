"""
Script untuk melengkapi (enrich) Genre buku yang masih kosong
menggunakan Open Library Books API (Batch Request via ISBN).

Fitur:
- Batching: Mengirim 20 ISBN per request (efisien & cepat).
- Resumable Caching: Menyimpan progress ke 'data/openlibrary_cache.json'.
  Jika dihentikan di tengah jalan atau koneksi terputus, bisa dilanjutkan tanpa mengulang.
- Auto-save: Menyimpan kemajuan ke 'Enrichment/books_enriched.csv' setiap 10 batch.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_CSV_PATH = BASE_DIR / "Enrichment" / "books_enriched.csv"
CACHE_PATH = BASE_DIR / "data" / "openlibrary_cache.json"
REPORT_MD_PATH = BASE_DIR / "reports" / "enrichment_report.md"
REPORT_JSON_PATH = BASE_DIR / "reports" / "enrichment_report.json"

BATCH_SIZE = 20
DELAY_BETWEEN_REQUESTS = 0.6  # detik, agar ramah ke server Open Library
HEADERS = {
    "User-Agent": "BookRecommenderHybridRAG/1.0 (academic research project; contact: student@univ.ac.id)"
}


def load_cache() -> dict:
    """Memuat cache respons Open Library yang sudah berhasil diambil sebelumnya."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    """Menyimpan cache ke file JSON."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_batch_openlibrary(isbns: list) -> dict:
    """
    Mengambil metadata untuk beberapa ISBN sekaligus (maksimal 20-30 ISBN per call).
    Returns: dict mapping dari ISBN -> list of subjects/genres
    """
    valid_isbns = [x for x in isbns if x and len(x) >= 9]
    if not valid_isbns:
        return {}

    bibkeys = ",".join([f"ISBN:{isbn}" for isbn in valid_isbns])
    url = f"https://openlibrary.org/api/books?bibkeys={bibkeys}&format=json&jscmd=data"
    req = urllib.request.Request(url, headers=HEADERS)

    results = {}
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for bibkey, book_info in data.items():
                isbn_key = bibkey.replace("ISBN:", "").strip()
                # Ambil subjek/genre buku (maksimal 8 subjek utama)
                subjects = [s.get("name", "") for s in book_info.get("subjects", []) if s.get("name")]
                results[isbn_key] = subjects[:8]
    except urllib.error.HTTPError as e:
        print(f"   [HTTP Error {e.code}] pada batch, dilewati...")
    except Exception as e:
        print(f"   [Koneksi Error] {e}, dilewati...")

    return results


def run_openlibrary_enrichment():
    print("=" * 65)
    print("ENRICHMENT TAHAP 2: OPEN LIBRARY API (BATCH REQUEST)")
    print("=" * 65)

    if not ENRICHED_CSV_PATH.exists():
        raise FileNotFoundError(f"File {ENRICHED_CSV_PATH} belum ada. Jalankan enrich_kaggle.py dulu!")

    df = pd.read_csv(ENRICHED_CSV_PATH, keep_default_na=False)
    total_books = len(df)

    # Identifikasi buku yang genrenya masih kosong
    missing_mask = df["Genre"].astype(str).str.strip() == ""
    missing_indices = df[missing_mask].index.tolist()
    total_missing = len(missing_indices)

    print(f"Total buku di dataset   : {total_books:,}")
    print(f"Sudah punya Genre       : {(total_books - total_missing):,}")
    print(f"Genre masih kosong      : {total_missing:,}")

    if total_missing == 0:
        print("\nSemua buku sudah memiliki genre! Tidak perlu memanggil API.")
        return

    # Muat cache
    cache = load_cache()
    print(f"Cache lokal terdeteksi  : {len(cache):,} entri tersimpan.")

    # Siapkan list ISBN yang perlu diambil
    unresolved_isbns = []
    for idx in missing_indices:
        isbn = str(df.at[idx, "ISBN"]).strip()
        if isbn and isbn not in cache:
            unresolved_isbns.append(isbn)

    unresolved_isbns = list(dict.fromkeys(unresolved_isbns))  # hapus duplikat jika ada
    print(f"ISBN unik yang akan dipanggil ke API: {len(unresolved_isbns):,}")

    # Proses Batching
    batches = [unresolved_isbns[i:i + BATCH_SIZE] for i in range(0, len(unresolved_isbns), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Dibagi menjadi {total_batches:,} batch (@ {BATCH_SIZE} ISBN per request).\n")

    start_time = time.time()
    try:
        for batch_num, batch_isbns in enumerate(batches, start=1):
            batch_results = fetch_batch_openlibrary(batch_isbns)

            # Masukkan hasil ke cache (termasuk yang kosong agar tidak dipanggil ulang)
            for isbn in batch_isbns:
                cache[isbn] = batch_results.get(isbn, [])

            # Cetak progress
            elapsed = time.time() - start_time
            print(f"[{batch_num}/{total_batches}] Batch selesai. Ditemukan {len(batch_results)} buku di batch ini. (Waktu: {elapsed:.1f}s)")

            # Simpan cache setiap 10 batch
            if batch_num % 10 == 0 or batch_num == total_batches:
                save_cache(cache)

            time.sleep(DELAY_BETWEEN_REQUESTS)

    except KeyboardInterrupt:
        print("\n[PEMBERITAHUAN] Proses dihentikan manual oleh user. Menyimpan progress yang sudah didapat...")
        save_cache(cache)

    # Terapkan hasil cache ke DataFrame
    print("\nMenerapkan hasil Open Library ke DataFrame...")
    applied_count = 0
    for idx in missing_indices:
        isbn = str(df.at[idx, "ISBN"]).strip()
        if isbn in cache and cache[isbn]:
            genres_list = cache[isbn]
            if genres_list:
                df.at[idx, "Genre"] = ", ".join(genres_list)
                df.at[idx, "genre_source"] = "openlibrary_api"
                applied_count += 1

    # Simpan kembali ke CSV
    df.to_csv(ENRICHED_CSV_PATH, index=False)
    save_cache(cache)

    total_with_genre = int((df["Genre"].astype(str).str.strip() != "").sum())
    final_match_rate = (total_with_genre / total_books) * 100

    print("\n" + "=" * 45)
    print("HASIL AKHIR METADATA ENRICHMENT")
    print("=" * 45)
    print(f"Total Buku               : {total_books:,}")
    print(f"Genre dari Open Library  : {applied_count:,}")
    print(f"Total Keseluruhan Genre  : {total_with_genre:,} ({final_match_rate:.2f}%)")
    print(f"Belum Ada Genre          : {(total_books - total_with_genre):,} ({100 - final_match_rate:.2f}%)")
    print(f"Dataset diperbarui di    : {ENRICHED_CSV_PATH}")
    print("=" * 45)


if __name__ == "__main__":
    run_openlibrary_enrichment()
