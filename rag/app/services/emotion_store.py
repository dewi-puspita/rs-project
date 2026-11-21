"""
Modul Emotion Companion Store.
Sesuai Spesifikasi Rantug v2 (Bab 6.4 & Bab 9).

Tugas:
1. Membaca kolom EmotionVector via ast.literal_eval().
2. Memvalidasi 9 dimensi numerik finite.
3. Menyimpan seluruh 19.784 buku (termasuk 6 buku non-retrieval)
   ke rag/vector_store/emotion_companion_store.json dengan kunci canonical book_id.
4. Menyediakan antarmuka lookup cepat O(1) dan Cosine Similarity.
"""

import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMOTION_STORE_PATH = BASE_DIR / "vector_store" / "emotion_companion_store.json"
if not EMOTION_STORE_PATH.parent.exists():
    EMOTION_STORE_PATH = BASE_DIR / "rag" / "vector_store" / "emotion_companion_store.json"

EXPECTED_EMOTIONS = [
    "Anger", "Anticipation", "Disgust", "Fear",
    "Joy", "Sadness", "Surprise", "Trust", "Objective"
]


class EmotionCompanionStore:
    def __init__(self, store_path: Path = EMOTION_STORE_PATH):
        self.store_path = store_path
        self._store: Dict[str, Dict[str, float]] = {}
        if self.store_path.exists():
            self.load()

    def load(self):
        with open(self.store_path, "r", encoding="utf-8") as f:
            self._store = json.load(f)

    def save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2)

    def build_from_dataframe(self, df: pd.DataFrame) -> int:
        """Membangun store dari DataFrame dan memvalidasi nilainya."""
        self._store = {}
        success_count = 0

        for _, row in df.iterrows():
            book_id = str(row["ID"]).strip()
            raw_ev = row["EmotionVector"]

            if pd.isna(raw_ev):
                continue

            try:
                ev_dict = ast.literal_eval(str(raw_ev)) if not isinstance(raw_ev, dict) else raw_ev
                validated_vector = {}
                is_valid = True

                for dim in EXPECTED_EMOTIONS:
                    val = ev_dict.get(dim)
                    if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                        is_valid = False
                        break
                    validated_vector[dim] = float(val)

                if is_valid and len(validated_vector) == 9:
                    self._store[book_id] = validated_vector
                    success_count += 1
            except Exception:
                continue

        self.save()
        return success_count

    def get_vector(self, book_id: str) -> Optional[Dict[str, float]]:
        """Mengambil dictionary 9 dimensi emosi buku berdasarkan ID."""
        return self._store.get(str(book_id).strip())

    def get_vector_array(self, book_id: str) -> Optional[np.ndarray]:
        """Mengambil array numpy berurutan 9 dimensi untuk kalkulasi kosinus."""
        vec_dict = self.get_vector(book_id)
        if not vec_dict:
            return None
        return np.array([vec_dict[dim] for dim in EXPECTED_EMOTIONS], dtype=np.float32)

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """
        Menghitung Cosine Similarity antara dua vektor emosi 9 dimensi.
        Menangani kasus vektor nol atau invalid dengan aman.
        """
        if vec_a is None or vec_b is None:
            return None
        if len(vec_a) != 9 or len(vec_b) != 9:
            return None

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0 or np.isnan(norm_a) or np.isnan(norm_b):
            return None

        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Pembatasan numerik -1.0 s.d 1.0
        return max(-1.0, min(1.0, sim))
