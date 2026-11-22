"""
Modul Emotion Ranking & Final Hybrid Re-Ranking Service.
Sesuai Spesifikasi Rantug v2 (Bab 8.4, Bab 9, dan Bab 10).

Tugas:
1. Mengambil EmotionVector 9 dimensi dari EmotionCompanionStore secara O(1).
2. Menghitung Cosine Similarity antara User Emotion Preference (P_u) dan Book Vector (V_b).
3. Melakukan normalisasi skor RRF (S_retrieval_norm) dan skor emosi (S_emotion_norm).
4. Melakukan fusi linier:
   S_final(b) = alpha * S_retrieval_norm(b) + beta * S_emotion_norm(b)
5. Mendukung fallback retrieval-only jika profil emosi tidak tersedia atau bernilai nol.
6. Menyimpan komponen skor terpisah (RRF, normalized retrieval, emotion, final) untuk audit transparansi.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from rag.app.services.emotion_store import (
    EmotionCompanionStore,
    EXPECTED_EMOTIONS
)


class EmotionRanker:
    def __init__(self, emotion_store: Optional[EmotionCompanionStore] = None):
        self.emotion_store = emotion_store or EmotionCompanionStore()

    @staticmethod
    def create_explicit_user_vector(emotion_weights: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Membangun User Emotion Preference dari preferensi emosi eksplisit (misal dari slider / form UI).
        Dimensi yang tidak ditentukan akan bernilai 0.0.
        Jika seluruh dimensi bernilai 0.0, mengembalikan None (fallback mode).
        """
        vec = []
        for dim in EXPECTED_EMOTIONS:
            val = float(emotion_weights.get(dim, 0.0))
            vec.append(max(0.0, val))
        
        arr = np.array(vec, dtype=np.float32)
        if np.linalg.norm(arr) == 0.0:
            return None
        return arr

    @staticmethod
    def normalize_min_max(values: List[float]) -> List[float]:
        """Normalisasi min-max ke rentang [0.0, 1.0]."""
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v == 0.0:
            return [1.0 for _ in values]
        return [round((v - min_v) / (max_v - min_v), 5) for v in values]

    def rank_candidates(
        self,
        retrieval_candidates: List[Dict[str, Any]],
        user_emotion_vector: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Melakukan final re-ranking menggabungkan skor retrieval dan skor emosi.

        Args:
            retrieval_candidates: List kandidat hasil fusi RRF (harus memuat 'book_id' dan 'rrf_score').
            user_emotion_vector: Vektor emosi pengguna (array 9 dimensi float), atau None.
            alpha: Bobot komponen retrieval (default: 0.5).
            beta: Bobot komponen emosi (default: 0.5).
            top_n: Jumlah rekomendasi akhir yang diinginkan (default: 5).

        Returns:
            List kandidat buku berperingkat akhir dengan komponen skor transparan.
        """
        if not retrieval_candidates:
            return []

        # 1. Normalisasi skor retrieval (RRF)
        raw_rrf_scores = [float(c.get("rrf_score", 0.0)) for c in retrieval_candidates]
        norm_retrieval_scores = self.normalize_min_max(raw_rrf_scores)

        # 2. Cek validitas user emotion vector
        is_emotion_active = False
        if user_emotion_vector is not None and len(user_emotion_vector) == 9:
            norm_u = np.linalg.norm(user_emotion_vector)
            if norm_u > 0.0 and not np.isnan(norm_u):
                is_emotion_active = True

        # 3. Hitung skor emosi dan final score untuk setiap kandidat
        scored_candidates = []
        for idx, c in enumerate(retrieval_candidates):
            book_id = str(c["book_id"]).strip()
            s_ret_norm = norm_retrieval_scores[idx]
            
            raw_cosine = None
            s_emo_norm = None
            book_vec_dict = self.emotion_store.get_vector(book_id)

            if is_emotion_active and book_vec_dict is not None:
                book_vec = np.array([book_vec_dict[dim] for dim in EXPECTED_EMOTIONS], dtype=np.float32)
                raw_cosine = self.emotion_store.calculate_cosine_similarity(user_emotion_vector, book_vec)
                
                if raw_cosine is not None:
                    # Cosine sim [-1.0, 1.0] diskalakan ke [0.0, 1.0]
                    s_emo_norm = round((raw_cosine + 1.0) / 2.0, 5)

            # Hitung skor akhir dengan formula Rantug v2
            if is_emotion_active and s_emo_norm is not None:
                final_score = round((alpha * s_ret_norm) + (beta * s_emo_norm), 5)
                emotion_status = "active"
            else:
                # Fallback retrieval-only
                final_score = round(s_ret_norm, 5)
                emotion_status = "fallback_retrieval_only"

            entry = {
                "book_id": book_id,
                "title": c.get("title", "Judul tidak tersedia"),
                "author": c.get("author", "Penulis tidak tersedia"),
                "genre": c.get("genre", ""),
                "themes": c.get("themes", ""),
                "average_rating": c.get("average_rating", 0.0),
                "rrf_rank": c.get("rrf_rank", idx + 1),
                "rrf_score": round(float(c.get("rrf_score", 0.0)), 5),
                "retrieval_score_norm": s_ret_norm,
                "raw_cosine_similarity": round(raw_cosine, 4) if raw_cosine is not None else None,
                "emotion_score_norm": s_emo_norm,
                "final_score": final_score,
                "alpha": alpha,
                "beta": beta,
                "emotion_status": emotion_status
            }
            scored_candidates.append(entry)

        # 4. Urutkan berdasarkan final_score descending (tie-breaker: rrf_score)
        scored_candidates.sort(key=lambda x: (x["final_score"], x["rrf_score"]), reverse=True)

        # 5. Tetapkan final_rank dan batasi Top-N
        for rank, item in enumerate(scored_candidates[:top_n], start=1):
            item["final_rank"] = rank

        return scored_candidates[:top_n]
