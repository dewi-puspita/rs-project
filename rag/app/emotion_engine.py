"""
=============================================================================
MODUL 2: EMOTION RECOMMENDER ENGINE (Profil Emosi 9 Dimensi & Fusi Skor Akhir)
=============================================================================
Slide PPT: "Modul Emotion Recommender & Final Ranking"

Fungsi Inti:
1. Emotion Companion Store: Lookup O(1) vektor emosi buku (19.784 buku).
2. Cosine Similarity     : Menghitung keselarasan preferensi emosi pembaca
                           terhadap EmotionVector buku (9 dimensi NRC-EIL).
3. Min-Max Normalization : Menyetarakan skala skor RRF dan skor emosi ke [0, 1].
4. Final Linear Fusion   : S_final = alpha * S_retrieval_norm + beta * S_emotion_norm
=============================================================================
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
EMOTION_STORE_PATH = BASE_DIR / "vector_store" / "emotion_companion_store.json"

# 9 Dimensi Emosi NRC-EIL sesuai Paper Acuan
EXPECTED_EMOTIONS = [
    "Anger", "Anticipation", "Disgust", "Fear",
    "Joy", "Sadness", "Surprise", "Trust", "Objective"
]


class EmotionRecommenderEngine:
    def __init__(self, store_path: Path = EMOTION_STORE_PATH):
        print("[Emotion Engine] Memuat Emotion Companion Store (9 Dimensi)...")
        with open(store_path, "r", encoding="utf-8") as f:
            self._emotion_store: Dict[str, Dict[str, float]] = json.load(f)
        print(f"[Emotion Engine] Siap digunakan ({len(self._emotion_store):,} profil emosi buku termuat).")

    def get_book_emotion_vector(self, book_id: str) -> Optional[np.ndarray]:
        """Mengambil array numpy 9 dimensi emosi buku secara O(1)."""
        vec_dict = self._emotion_store.get(str(book_id).strip())
        if not vec_dict:
            return None
        return np.array([vec_dict[dim] for dim in EXPECTED_EMOTIONS], dtype=np.float32)

    @staticmethod
    def create_user_emotion_vector(emotion_weights: Optional[Dict[str, float]]) -> Optional[np.ndarray]:
        """
        Membangun vektor preferensi emosi pengguna dari input eksplisit (UI / Chat).
        Case-insensitive agar 'joy' cocok dengan 'Joy', dsb.
        Jika semua dimensi 0.0, mengembalikan None (mengaktifkan mode fallback).
        """
        if not emotion_weights:
            return None
        lower_weights = {str(k).strip().lower(): float(v) for k, v in emotion_weights.items()}
        vec = [max(0.0, float(lower_weights.get(dim.lower(), 0.0))) for dim in EXPECTED_EMOTIONS]
        arr = np.array(vec, dtype=np.float32)
        if np.linalg.norm(arr) == 0.0:
            return None
        return arr

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """Menghitung Cosine Similarity antara dua vektor 9 dimensi."""
        if vec_a is None or vec_b is None:
            return None
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0 or np.isnan(norm_a) or np.isnan(norm_b):
            return None
        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, sim))

    @staticmethod
    def normalize_min_max(values: List[float]) -> List[float]:
        """Menormalisasi nilai ke rentang [0.0, 1.0]."""
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v == 0.0:
            return [1.0 for _ in values]
        return [round((v - min_v) / (max_v - min_v), 5) for v in values]

    def re_rank(
        self,
        retrieval_candidates: List[Dict[str, Any]],
        user_emotion_vector: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Menghitung skor akhir dan melakukan re-ranking kandidat rekomendasi.

        Rumus:
        S_final = alpha * S_retrieval_norm + beta * S_emotion_norm
        """
        if not retrieval_candidates:
            return []

        # 1. Normalisasi skor RRF
        raw_rrf = [float(c.get("rrf_score", 0.0)) for c in retrieval_candidates]
        norm_retrieval = self.normalize_min_max(raw_rrf)

        # 2. Cek apakah preferensi emosi aktif
        is_emotion_active = False
        if user_emotion_vector is not None and len(user_emotion_vector) == 9:
            if np.linalg.norm(user_emotion_vector) > 0.0:
                is_emotion_active = True

        # 3. Hitung skor untuk setiap kandidat
        scored_candidates = []
        for idx, c in enumerate(retrieval_candidates):
            bid = str(c["book_id"]).strip()
            s_ret_norm = norm_retrieval[idx]

            raw_cosine = None
            s_emo_norm = None

            if is_emotion_active:
                b_vec = self.get_book_emotion_vector(bid)
                if b_vec is not None:
                    raw_cosine = self.calculate_cosine_similarity(user_emotion_vector, b_vec)
                    if raw_cosine is not None:
                        # Skalakan [-1.0, 1.0] ke [0.0, 1.0]
                        s_emo_norm = round((raw_cosine + 1.0) / 2.0, 5)

            # Hitung skor fusi akhir
            if is_emotion_active and s_emo_norm is not None:
                final_score = round((alpha * s_ret_norm) + (beta * s_emo_norm), 5)
                emotion_status = "active"
            else:
                # Fallback jika emosi tidak disetel
                final_score = round(s_ret_norm, 5)
                emotion_status = "fallback_retrieval_only"

            scored = dict(c)
            scored.update({
                "retrieval_score_norm": s_ret_norm,
                "raw_cosine_similarity": round(raw_cosine, 4) if raw_cosine is not None else None,
                "emotion_score_norm": s_emo_norm,
                "final_score": final_score,
                "alpha": alpha,
                "beta": beta,
                "emotion_status": emotion_status
            })
            scored_candidates.append(scored)

        # 4. Urutkan berdasarkan final_score tertinggi (tie-breaker: rrf_score)
        scored_candidates.sort(key=lambda x: (x["final_score"], x["rrf_score"]), reverse=True)

        # 5. Berikan final_rank
        for rank, item in enumerate(scored_candidates[:top_n], start=1):
            item["final_rank"] = rank

        return scored_candidates[:top_n]
