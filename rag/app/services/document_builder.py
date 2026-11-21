"""
Modul Document Builder Baseline.
Sesuai Spesifikasi Rantug v2 (Bab 3.4 & Bab 5).

Tugas:
1. Menyusun teks dasar representasi buku: Title + Author + Description.
2. Tidak menambahkan placeholder buatan ([untitled], ID, dll) dalam teks representasi.
3. Menghasilkan 3 variasi teks logis:
   - 'document_text'     : Teks dasar untuk BM25
   - 'e5_passage_text'   : Teks berawalan 'passage: ' untuk Multilingual E5
   - 'bge_passage_text'  : Teks murni (tanpa prefix) untuk BGE English
4. Genre dan Themes tidak dicampur ke teks representasi baseline, melainkan
   disediakan sebagai payload dictionary metadata terpisah.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd


def clean_text_field(val: Any) -> str:
    """Membersihkan whitespace dan mengabaikan nilai NaN / string kosong."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def build_baseline_document(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Menyusun teks dokumen baseline dari baris data buku.

    Returns:
        (document_text, e5_passage_text, bge_passage_text, metadata_dict)
        Jika buku tidak memiliki Title dan Description, mengembalikan (None, None, None, metadata_dict)
    """
    book_id = str(row.get("ID", "")).strip()
    title = clean_text_field(row.get("Title", ""))
    author = clean_text_field(row.get("Author", ""))
    description = clean_text_field(row.get("Description", ""))

    # Kriteria kelayakan: Wajib memiliki Title ATAU Description
    if not title and not description:
        return None, None, None, {}

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if author:
        parts.append(f"Author: {author}")
    if description:
        parts.append(f"Description: {description}")

    doc_text = "\n".join(parts)

    # 1. Dokumen untuk BM25 & BGE: Teks dasar
    bge_passage = doc_text
    # 2. Dokumen untuk Multilingual E5: Prefix 'passage: '
    e5_passage = f"passage: {doc_text}"

    # Metadata terpisah untuk ChromaDB
    metadata = {
        "book_id": book_id,
        "isbn": clean_text_field(row.get("ISBN", "")),
        "title": title or "Judul tidak tersedia",
        "author": author or "Penulis tidak tersedia",
        "average_rating": float(row.get("Average_Rating", 0.0)) if clean_text_field(row.get("Average_Rating", "")) else 0.0,
        "age": float(row.get("Age", 0.0)) if clean_text_field(row.get("Age", "")) else 0.0,
        "genre": clean_text_field(row.get("Genre", "")),
        "genre_source": clean_text_field(row.get("genre_source", "")),
        "themes": clean_text_field(row.get("Themes", "")),
        "themes_source": clean_text_field(row.get("themes_source", ""))
    }

    return doc_text, e5_passage, bge_passage, metadata
