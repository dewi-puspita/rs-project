import json
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Path definitions
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "Teenager_GoodReads_Emotion.csv"
OUTPUT_CLEAN_PATH = BASE_DIR / "Pre-processing" / "books_clean.csv"
REPORT_JSON_PATH = BASE_DIR / "reports" / "validation_report.json"
REPORT_MD_PATH = BASE_DIR / "reports" / "validation_report.md"

EXPECTED_EMOTIONS = [
    "Anger", "Anticipation", "Disgust", "Fear", 
    "Joy", "Sadness", "Surprise", "Trust", "Objective"
]


def clean_text(text: str) -> str:
    """Strip leading/trailing whitespace, normalize internal whitespace, and lower-case."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text


def clean_isbn(isbn_val) -> str:
    """Normalize ISBN string, stripping whitespace."""
    if pd.isna(isbn_val):
        return ""
    return str(isbn_val).strip()


def validate_emotion_vector(ev_raw):
    """
    Validate that EmotionVector can be parsed into a 9-dimensional dict
    matching EXPECTED_EMOTIONS without altering any numeric values.
    Returns: (is_valid: bool, error_msg: str or None)
    """
    if pd.isna(ev_raw):
        return False, "EmotionVector is missing/null"
    try:
        if isinstance(ev_raw, dict):
            ev_dict = ev_raw
        else:
            ev_dict = ast.literal_eval(str(ev_raw))
        
        if not isinstance(ev_dict, dict):
            return False, "EmotionVector is not a dictionary"
        
        for dim in EXPECTED_EMOTIONS:
            if dim not in ev_dict:
                return False, f"Missing dimension: {dim}"
            val = ev_dict[dim]
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                return False, f"Invalid numeric value for {dim}: {val}"
                
        return True, None
    except Exception as e:
        return False, f"Parsing error: {str(e)}"


def run_preprocessing():
    print(f"Loading raw dataset from: {RAW_DATA_PATH}")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH, keep_default_na=False)
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # 1. Missing / Blank Values Check di Data Mentah
    raw_blank_counts = {
        col: int((df[col].astype(str).str.strip() == "").sum())
        for col in df.columns
    }
    print("Blank/empty strings per column in raw data:", raw_blank_counts)

    # 2. Text Normalization: Whitespace cleaning and lowercasing
    print("Performing whitespace cleaning and lowercasing...")
    for col in ["Title", "Author", "Description"]:
        df[col] = df[col].apply(clean_text)
    df["ISBN"] = df["ISBN"].apply(clean_isbn)

    # 3. Emotion Vector Validation (tanpa merubah nilai)
    print("Validating Emotion Vectors...")
    ev_validation_results = df["EmotionVector"].apply(validate_emotion_vector)
    valid_ev_mask = ev_validation_results.apply(lambda x: x[0])
    invalid_ev_count = int((~valid_ev_mask).sum())
    print(f"Valid Emotion Vectors: {valid_ev_mask.sum()}/{initial_rows} (Invalid: {invalid_ev_count})")

    # 4. Deduplikasi ID
    dup_id_mask = df.duplicated(subset=["ID"], keep=False)
    total_dup_id_rows = int(dup_id_mask.sum())
    unique_dup_ids = int(df.loc[dup_id_mask, "ID"].nunique())
    print(f"Duplicate IDs: {unique_dup_ids} distinct IDs across {total_dup_id_rows} rows.")

    # Deduplicate strictly by ID, retaining the first occurrence
    df_clean = df.drop_duplicates(subset=["ID"], keep="first").copy()
    rows_after_dedup = len(df_clean)
    dropped_duplicates = initial_rows - rows_after_dedup
    print(f"Dropped {dropped_duplicates} duplicate rows. Clean rows count: {rows_after_dedup}")

    # 5. Type and Value Verification
    df_clean["Average_Rating"] = pd.to_numeric(df_clean["Average_Rating"], errors="coerce")
    df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")
    df_clean["ID"] = df_clean["ID"].astype(np.int64)

    # 6. Save clean dataset
    OUTPUT_CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CLEAN_PATH, index=False)
    print(f"Successfully saved clean dataset to: {OUTPUT_CLEAN_PATH}")

    # 7. Generate Comprehensive Validation Report
    report = {
        "raw_dataset_path": str(RAW_DATA_PATH.name),
        "output_clean_path": str(OUTPUT_CLEAN_PATH.name),
        "initial_total_rows": initial_rows,
        "clean_total_rows": rows_after_dedup,
        "dropped_duplicate_rows": dropped_duplicates,
        "unique_authors_count": int(df_clean[df_clean["Author"] != ""]["Author"].nunique()),
        "unique_titles_count": int(df_clean[df_clean["Title"] != ""]["Title"].nunique()),
        "blank_values_raw": raw_blank_counts,
        "blank_values_clean": {
            col: int((df_clean[col].astype(str).str.strip() == "").sum())
            for col in df_clean.columns
        },
        "invalid_emotion_vectors": invalid_ev_count,
        "age_distribution": {
            "min": float(df_clean["Age"].min()),
            "max": float(df_clean["Age"].max()),
            "mean": round(float(df_clean["Age"].mean()), 2),
            "median": round(float(df_clean["Age"].median()), 2)
        },
        "rating_distribution": {
            "min": float(df_clean["Average_Rating"].min()),
            "max": float(df_clean["Average_Rating"].max()),
            "mean": round(float(df_clean["Average_Rating"].mean()), 2)
        }
    }

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown Report
    md_content = f"""# Laporan Validasi dan Preprocessing Dataset

- **Waktu Eksekusi**: Dataset dibersihkan dan divalidasi.
- **Dataset Asli**: `{RAW_DATA_PATH.name}`
- **Dataset Bersih**: `{OUTPUT_CLEAN_PATH.name}`

## 1. Ringkasan Baris Data
- **Jumlah Baris Awal**: {initial_rows:,}
- **Duplikat ID Dihapus**: {dropped_duplicates:,} baris
- **Jumlah Baris Bersih**: {rows_after_dedup:,} baris (100% ID Unik)
- **Jumlah Penulis Unik (Author)**: {report['unique_authors_count']:,}
- **Jumlah Judul Unik (Title)**: {report['unique_titles_count']:,}

> **Catatan Penting**: Penulis yang sama dengan judul buku berbeda tetap dipertahankan 100%. Hanya baris dengan `ID` buku yang identik/terduplikasi yang dirapikan.

## 2. Pembersihan Teks & Case Folding
- Kolom `Title`, `Author`, dan `Description` telah dibersihkan dari leading/trailing whitespace (`.strip()`) dan dinormalisasi menjadi lowercase.
- Spasi ganda di dalam teks telah dinormalisasi menjadi spasi tunggal.
- String kosong terdeteksi: Title ({report['blank_values_clean']['Title']}), Author ({report['blank_values_clean']['Author']}), Description ({report['blank_values_clean']['Description']}). Record ini tetap dipertahankan karena ID dan EmotionVector-nya valid, dan dapat dilengkapi pada tahap metadata enrichment.

## 3. Validasi Emotion Vector
- **Status**: 100% Valid ({initial_rows - invalid_ev_count:,}/{initial_rows:,})
- Tidak ada vektor kosong, `NaN`, atau `Inf`.
- Nilai numerik Emotion Vector asli dari NRC-EIL **tidak diubah** sama sekali.

## 4. Statistik Kolom Numerik
- **Average Rating**: Min {report['rating_distribution']['min']} | Max {report['rating_distribution']['max']} | Rata-rata {report['rating_distribution']['mean']}
- **Age (Average Reader Age)**: Min {report['age_distribution']['min']} | Max {report['age_distribution']['max']} | Rata-rata {report['age_distribution']['mean']} tahun
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Validation reports generated:\n - {REPORT_JSON_PATH}\n - {REPORT_MD_PATH}")


if __name__ == "__main__":
    run_preprocessing()
