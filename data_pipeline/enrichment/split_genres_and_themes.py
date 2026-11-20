"""
Script untuk memisahkan metadata campuran (Subjek Open Library & Kaggle)
menjadi dua kolom terpisah yang rapi:
1. 'Genre'  : Hanya genre / bentuk literatur murni
2. 'Themes' : Hanya tema / topik narasi cerita

Fleksibel:
- Bisa dijalankan sekarang dengan 'Enrichment/Genre_Teenager_GoodReads_Emotion.csv'
- Dan bisa dijalankan ulang nanti kapan saja jika temanmu sudah memberikan
  file genre yang sudah ditambah Google Books!
"""

import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

# Path input utama: Genre_Teenager_GoodReads_Emotion.csv di folder Enrichment
DEFAULT_INPUT_FILE = BASE_DIR / "Enrichment" / "Genre_Teenager_GoodReads_Emotion.csv"

# Daftar file fallback jika file utama tidak ditemukan
POSSIBLE_INPUTS = [
    DEFAULT_INPUT_FILE,
    BASE_DIR / "Enrichment" / "books_enriched.csv",
    Path("Genre_Teenager_GoodReads_Emotion.csv"),
    Path("Enrichment/Genre_Teenager_GoodReads_Emotion.csv"),
]

OUTPUT_FILE = BASE_DIR / "Enrichment" / "books_enriched_with_themes.csv"

# ==========================================
# TAKSONOMI KATA KUNCI: GENRE vs THEMES
# ==========================================

# 1. Kata kunci Genre (Bentuk literatur, kategori pasar)
GENRE_KEYWORDS = {
    "fiction", "nonfiction", "non-fiction", "fantasy", "science fiction", "sci-fi",
    "mystery", "thriller", "suspense", "horror", "romance", "historical fiction",
    "history", "biography", "autobiography", "memoir", "poetry", "drama", "plays",
    "comics", "graphic novel", "manga", "adventure", "action", "young adult",
    "juvenile fiction", "children", "middle grade", "classics", "humor", "comedy",
    "crime", "detective", "dystopia", "dystopian", "paranormal", "urban fantasy",
    "contemporary", "literary fiction", "novela", "short stories", "folklore", "mythology"
}

# 2. Kata kunci Tema Narasi (Topik, isu emosional/kehidupan, pesan cerita)
THEME_KEYWORDS = {
    "friendship", "best friends", "female friendship", "family", "family life",
    "fathers and daughters", "mothers and daughters", "brothers and sisters",
    "parents", "grandparents", "love", "love stories", "man-woman relationships",
    "relationships", "coming of age", "identity", "loss", "grief", "death", "mort",
    "survival", "war", "world war", "bullying", "prejudices", "racism", "discrimination",
    "mental health", "depression", "suicide", "hope", "courage", "courageous",
    "secrets", "betrayal", "guilt", "forgiveness", "space and time", "time travel",
    "loneliness", "isolation", "redemption", "faith", "morality", "growing up",
    "conduct of life", "self-discovery", "social issues", "poverty"
}


def clean_term(term: str) -> str:
    """Membersihkan whitespace dan karakter aneh pada satu istilah."""
    t = term.strip()
    # Bersihkan jika ada tanda kurung ekstra atau format aneh
    t = re.sub(r"\s+", " ", t)
    return t


def split_subjects(raw_str: str):
    """
    Memecah string subjek (koma) menjadi:
    - cleaned_genres: list string genre murni
    - cleaned_themes: list string tema narasi
    """
    if pd.isna(raw_str) or not str(raw_str).strip() or str(raw_str).strip().lower() == "nan":
        return [], []

    terms = [clean_term(x) for x in str(raw_str).split(",") if clean_term(x)]
    
    genres = []
    themes = []

    for term in terms:
        t_low = term.lower()

        # Cek apakah ini tema narasi
        is_theme = any(tk in t_low for tk in THEME_KEYWORDS)
        # Cek apakah ini genre murni
        is_genre = any(gk in t_low for gk in GENRE_KEYWORDS)

        # Prioritas pemilahan
        if is_theme and not is_genre:
            themes.append(term)
        elif is_genre and not is_theme:
            genres.append(term)
        elif is_theme and is_genre:
            # Contoh: "love stories" -> bisa tema, "romantic fiction" -> genre
            if "fiction" in t_low or "novel" in t_low:
                genres.append(term)
            else:
                themes.append(term)
        else:
            # Jika istilah umum atau tidak spesifik tema, default ke genre
            # kecual ada kata kunci sosial/emosi
            genres.append(term)

    # Hapus duplikasi dengan menjaga urutan
    genres_unique = list(dict.fromkeys(genres))
    themes_unique = list(dict.fromkeys(themes))

    return genres_unique, themes_unique


def run_split_and_enrich(input_csv_path: str = None):
    print("=" * 65)
    print("PEMISAHAN METADATA: GENRE vs THEMES")
    print("=" * 65)

    if input_csv_path:
        src_path = Path(input_csv_path)
    else:
        src_path = None
        for p in POSSIBLE_INPUTS:
            if p.exists():
                src_path = p
                break

    if not src_path or not src_path.exists():
        raise FileNotFoundError(f"File dataset enrichment belum ditemukan! Jalur yang dicek: {[str(p.name) for p in POSSIBLE_INPUTS]}")

    print(f"Membaca dataset sumber: {src_path.name}")
    df = pd.read_csv(src_path, keep_default_na=False)
    total_books = len(df)
    print(f"Total baris buku     : {total_books:,}")

    # Kolom awal
    if "Genre" not in df.columns:
        raise KeyError("Kolom 'Genre' tidak ditemukan di dataset sumber!")

    final_genres = []
    final_themes = []
    themes_sources = []

    count_with_genre = 0
    count_with_theme = 0

    for idx, row in df.iterrows():
        raw_val = row["Genre"]
        src_genre = row.get("genre_source", "")

        genres, themes = split_subjects(raw_val)

        # Gabungkan kembali menjadi string koma
        genre_str = ", ".join(genres)
        theme_str = ", ".join(themes)

        final_genres.append(genre_str)
        final_themes.append(theme_str)

        if genre_str:
            count_with_genre += 1

        if theme_str:
            count_with_theme += 1
            themes_sources.append("openlibrary_subjects" if "openlibrary" in str(src_genre) else "external_metadata")
        else:
            themes_sources.append("none")

    # Update kolom di DataFrame
    df["Genre"] = final_genres
    df["Themes"] = final_themes
    df["themes_source"] = themes_sources

    # Simpan hasil
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 45)
    print("HASIL PEMISAHAN GENRE & THEMES")
    print("=" * 45)
    print(f"Total Buku               : {total_books:,}")
    print(f"Buku dengan Genre murni  : {count_with_genre:,} ({count_with_genre/total_books*100:.2f}%)")
    print(f"Buku dengan Themes narasi: {count_with_theme:,} ({count_with_theme/total_books*100:.2f}%)")
    print(f"Buku tanpa Themes (jujur): {(total_books - count_with_theme):,} ({(total_books - count_with_theme)/total_books*100:.2f}%)")
    print("-" * 45)
    print(f"File hasil disimpan ke   : {OUTPUT_FILE}")
    print("=" * 45)

    print("\nContoh 3 buku hasil pemisahan:")
    sample = df[df["Themes"] != ""].head(3)
    for _, r in sample.iterrows():
        print(f"\n[ID: {r.ID}] {r.Title} - {r.Author}")
        print(f"  -> Genre  : {r.Genre}")
        print(f"  -> Themes : {r.Themes} (Source: {r.themes_source})")


if __name__ == "__main__":
    import sys
    custom_input = sys.argv[1] if len(sys.argv) > 1 else None
    run_split_and_enrich(custom_input)
