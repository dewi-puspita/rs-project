"""
Modul penyusun teks dokumen buku (Document Text Builder) untuk keperluan
embedding (Dense Retrieval) dan BM25 (Sparse Retrieval).

Sesuai Spesifikasi Rancangan Tugas (Bab 5.2):
Format:
Title: {title}
Author: {author}
Description: {description}
Genre: {genre_if_available}
Themes: {themes_if_available}

Aturan:
- Description menjadi sumber teks utama.
- Genre dan Themes hanya ditambahkan jika datanya tersedia dan bukan string kosong / NaN.
- Nilai kosong TIDAK menampilkan 'None' atau 'NaN'.
- Metadata teknis (ISBN, ID, EmotionVector, genre_source, themes_source) TIDAK dimasukkan
  ke dalam teks representasi agar tidak mengotori makna semantik embedding.
"""

from typing import Dict, Any
import pandas as pd


def build_document_text(row: Dict[str, Any], lowercase: bool = True) -> str:
    """
    Menyusun teks dokumen tunggal dari baris data buku.
    Input bisa berupa dictionary atau pandas Series.
    
    Argumen:
    - lowercase: Jika True, memastikan seluruh teks (Title, Author, Desc, Genre, Themes)
      berada dalam huruf kecil (case-folded) agar konsisten untuk Dense Embedding & BM25.
    """
    parts = []

    def clean_val(val: Any) -> str:
        if val is None or pd.isna(val):
            return ""
        s = str(val).strip()
        if s.lower() == "nan":
            return ""
        return s.lower() if lowercase else s

    # 1. Title
    title = clean_val(row.get("Title", ""))
    if title:
        parts.append(f"Title: {title}")

    # 2. Author
    author = clean_val(row.get("Author", ""))
    if author:
        parts.append(f"Author: {author}")

    # 3. Description (Komponen Utama)
    description = clean_val(row.get("Description", ""))
    if description:
        parts.append(f"Description: {description}")

    # 4. Genre (Hanya jika tersedia)
    genre = clean_val(row.get("Genre", ""))
    if genre:
        parts.append(f"Genre: {genre}")

    # 5. Themes (Hanya jika tersedia)
    themes = clean_val(row.get("Themes", ""))
    if themes:
        parts.append(f"Themes: {themes}")

    return "\n".join(parts)


def prepare_documents_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan kolom 'document_text' ke DataFrame buku.
    """
    df = df.copy()
    df["document_text"] = df.apply(lambda r: build_document_text(r.to_dict()), axis=1)
    return df
