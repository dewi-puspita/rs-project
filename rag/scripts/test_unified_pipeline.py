"""
=============================================================================
SKRIP DEMO & PENGUJIAN PIPELINE TUNGGAL 3 MODUL INTI
=============================================================================
Slide PPT: "Alur Eksekusi End-to-End Chatbot Rekomendasi Buku"

Alur:
1. Input Kueri User
2. [Modul 1] rag_engine.py      -> Hybrid Retrieval (Dense E5 + Sparse BM25 + RRF)
3. [Modul 2] emotion_engine.py  -> Re-ranking Emosi (Cosine Sim + Fusi Skor Alpha/Beta)
4. [Modul 3] llm_engine.py      -> Grounded Explanation (Ollama Local / Offline)
=============================================================================
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.rag_engine import HybridRAGEngine
from rag.app.emotion_engine import EmotionRecommenderEngine
from rag.app.llm_engine import LLMEngine


def main():
    print("=" * 70)
    print("UJI COBA PIPELINE TERPADU 3 MODUL (RAG -> EMOTION -> LLM)")
    print("=" * 70)

    user_prompt = "Aku ingin membaca cerita petualangan dan persahabatan remaja yang seru dan penuh kejutan."
    print(f"\n[USER INPUT]: \"{user_prompt}\"\n")

    # Inisialisasi 3 Engine
    rag = HybridRAGEngine()
    emotion = EmotionRecommenderEngine()
    llm = LLMEngine(model_name="llama3.1:8b")

    # LANGKAH 1: Analisis Kueri
    print("\n--- [LANGKAH 1] Parsing Kueri ---")
    parsed = llm.parse_user_query(user_prompt)
    print(f" -> Kata Kunci BM25 : {parsed['bm25_keywords_en']}")
    print(f" -> Deteksi Emosi   : {parsed['detected_emotions']}")

    # LANGKAH 2: Hybrid RAG Search (Dense E5 + Sparse BM25 + RRF)
    print("\n--- [LANGKAH 2: rag_engine.py] Menjalankan Hybrid Retrieval ---")
    retrieval_candidates = rag.search(
        query_id=parsed["query_id"],
        keywords_en=parsed["bm25_keywords_en"],
        top_k=20
    )
    print(f" -> Ditemukan {len(retrieval_candidates)} kandidat buku dari Hybrid RRF.")
    print(f" -> Top-1 RRF: '{retrieval_candidates[0]['title']}' (RRF Score: {retrieval_candidates[0]['rrf_score']:.5f})")

    # LANGKAH 3: Emotion Re-Ranking (9 Dimensi + Fusi Alpha/Beta)
    print("\n--- [LANGKAH 3: emotion_engine.py] Menjalankan Emotion Re-ranking ---")
    user_vec = emotion.create_user_emotion_vector(parsed["detected_emotions"])
    ranked_candidates = emotion.re_rank(
        retrieval_candidates=retrieval_candidates,
        user_emotion_vector=user_vec,
        alpha=0.5,
        beta=0.5,
        top_n=5
    )
    print(f" -> Top-5 Hasil Akhir Rekomendasi (Setelah Re-ranking Emosi):")
    for b in ranked_candidates:
        print(f"    Rank {b['final_rank']} | Skor: {b['final_score']:.4f} (RetNorm: {b['retrieval_score_norm']:.4f}, EmoNorm: {b['emotion_score_norm']}) | {b['title']} - {b['author']}")

    # LANGKAH 4: LLM Grounded Explanation
    print("\n--- [LANGKAH 4: llm_engine.py] Menghasilkan Respons Grounded ---")
    response_text = llm.generate_recommendation(user_prompt, ranked_candidates)
    print("\n" + "=" * 70)
    print("OUTPUT CHATBOT UNTUK PENGGUNA:")
    print("=" * 70)
    print(response_text)
    print("=" * 70)
    print("[SELESAI] Pipeline 3 modul berjalan 100% sempurna!")


if __name__ == "__main__":
    main()
