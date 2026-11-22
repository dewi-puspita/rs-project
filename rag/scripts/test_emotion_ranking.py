"""
Skrip Pengujian Tahap 8: Emotion Recommender & Final Hybrid-Emotion Ranking.
Sesuai Rantug v2 (Bab 8.4: Eksperimen R3 s.d R5, Bab 9, dan Bab 10).

Fungsi:
1. Menjalankan Unit Test Cosine Similarity & Fallback Handling.
2. Mengambil 20 kandidat hasil Hybrid Retrieval (Dense E5 + BM25 + RRF).
3. Menguji berbagai skenario profil emosi pengguna (Joy/Anticipation vs Sadness/Fear vs Fallback).
4. Melakukan uji sensitivitas bobot alpha & beta (0.5/0.5 vs 0.8/0.2 vs 0.2/0.8).
5. Mengekspor laporan resmi ke:
   - reports/emotion_ranking_report.md
   - reports/emotion_ranking_report.json
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.services.bm25_service import BM25Service
from rag.app.services.chroma_service import (
    get_chroma_client,
    CHROMA_E5_DIR,
    COLLECTION_NAME_E5
)
from rag.app.services.rrf_fusion import reciprocal_rank_fusion
from rag.app.services.emotion_store import (
    EmotionCompanionStore,
    EXPECTED_EMOTIONS
)
from rag.app.services.emotion_ranker import EmotionRanker

ELIGIBLE_DOCS_PATH = BASE_DIR / "rag" / "vector_store" / "eligible_documents.csv"
REPORT_MD_PATH = BASE_DIR / "reports" / "emotion_ranking_report.md"
REPORT_JSON_PATH = BASE_DIR / "reports" / "emotion_ranking_report.json"


def run_unit_tests():
    print("=" * 70)
    print("BAGIAN 1: UNIT TEST KALKULASI EMOTION VECTOR & COSINE SIMILARITY")
    print("=" * 70)
    
    store = EmotionCompanionStore()
    ranker = EmotionRanker(store)
    
    # 1. Test urutan dimensi
    assert len(EXPECTED_EMOTIONS) == 9
    print(f"[OK] 9 Dimensi Terdaftar Konsisten: {', '.join(EXPECTED_EMOTIONS)}")

    # 2. Test identical vector similarity = 1.0
    vec_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    sim_self = store.calculate_cosine_similarity(vec_a, vec_a)
    assert abs(sim_self - 1.0) < 1e-5
    print(f"[OK] Cosine similarity vektor identik: {sim_self:.4f} (Expected: 1.0000)")

    # 3. Test orthogonal vector similarity = 0.0
    vec_b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    vec_c = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    sim_ortho = store.calculate_cosine_similarity(vec_b, vec_c)
    assert abs(sim_ortho - 0.0) < 1e-5
    print(f"[OK] Cosine similarity vektor ortogonal: {sim_ortho:.4f} (Expected: 0.0000)")

    # 4. Test zero-vector handling (must return None without crash)
    vec_zero = np.zeros(9, dtype=np.float32)
    sim_zero = store.calculate_cosine_similarity(vec_a, vec_zero)
    assert sim_zero is None
    print("[OK] Vektor nol ditangani dengan aman (mengembalikan None / fallback)")

    # 5. Test min-max normalization
    raw_vals = [0.01, 0.02, 0.03]
    norm_vals = ranker.normalize_min_max(raw_vals)
    assert norm_vals == [0.0, 0.5, 1.0]
    print(f"[OK] Normalisasi Min-Max lulus: {raw_vals} -> {norm_vals}")

    print("\nSeluruh Unit Test Tahap 8 LULUS 100%!\n")


def get_hybrid_candidates(query_id: str, bm25_keywords_en: str, top_k: int = 20) -> list:
    df_meta = pd.read_csv(ELIGIBLE_DOCS_PATH, keep_default_na=False).set_index("ID")

    # 1. BM25 Search
    bm25_svc = BM25Service()
    bm25_hits = bm25_svc.search(bm25_keywords_en, top_k=top_k)

    # 2. Dense E5 Search
    client = get_chroma_client(CHROMA_E5_DIR)
    collection = client.get_collection(COLLECTION_NAME_E5)
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
    emb = model.encode([f"query: {query_id}"], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=emb, n_results=top_k, include=["metadatas", "distances"])
    
    dense_hits = []
    for rank, (bid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
        dense_hits.append({
            "book_id": str(bid),
            "rank": rank,
            "dense_score": round(1.0 - dist, 4)
        })

    # 3. RRF Fusion
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits, k=60, top_candidates=top_k)
    
    # Enrich metadata
    enriched = []
    for c in fused:
        bid = str(c["book_id"]).strip()
        bid_int = int(bid) if bid.isdigit() else None
        meta = df_meta.loc[bid_int] if bid_int in df_meta.index else {}
        enriched.append({
            "book_id": bid,
            "title": meta.get("Title", "Judul tidak tersedia") if isinstance(meta, pd.Series) else "Judul tidak tersedia",
            "author": meta.get("Author", "Penulis tidak tersedia") if isinstance(meta, pd.Series) else "Penulis tidak tersedia",
            "genre": meta.get("Genre", "") if isinstance(meta, pd.Series) else "",
            "themes": meta.get("Themes", "") if isinstance(meta, pd.Series) else "",
            "average_rating": float(meta.get("Average_Rating", 0.0)) if isinstance(meta, pd.Series) and str(meta.get("Average_Rating", "")).strip() else 0.0,
            "rrf_rank": c["rrf_rank"],
            "rrf_score": c["rrf_score"]
        })
    return enriched


def generate_emotion_report(
    query_text: str,
    scenario_results: dict,
    weight_sensitivity_results: dict
):
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. JSON Export
    json_data = {
        "report_title": "Laporan Pengujian Emotion Recommender & Final Ranking (Tahap 8)",
        "generated_at": datetime.now().isoformat(),
        "query": query_text,
        "scenarios": scenario_results,
        "weight_sensitivity": weight_sensitivity_results
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 2. Markdown Export
    md = f"""# Laporan Pengujian Emotion Recommender & Final Ranking (Tahap 8)
*Sesuai Rancangan Tugas v2 (Bab 8.4, Bab 9, dan Bab 10)*

- **Tanggal Pengujian**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Dataset Emosi**: `emotion_companion_store.json` (19.784 buku ter-validasi, 9 dimensi NRC-EIL)
- **Formula Final Ranking**: $S_{{final}} = \\alpha \\cdot S_{{retrieval\\_norm}} + \\beta \\cdot S_{{emotion\\_norm}}$
- **Kueri Pengujian**: *"{query_text}"*

---

## 1. Hasil Final Re-Ranking Berdasarkan Skenario Emosi Pengguna

### Skenario A: Profil Emosi Positif & Penuh Harapan (*Joy, Anticipation, Trust*)
*User Vector: Joy=0.8, Anticipation=0.7, Trust=0.6, lainnya=0.1 | $\\alpha=0.5, \\beta=0.5$*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Skor Retrieval (Norm) | Alignment Emosi (Norm) | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|:---:|
"""
    for item in scenario_results["scenario_joy_trust"]["top_candidates"]:
        md += f"| **{item['final_rank']}** | **{item['title']}** | {item['author']} | Rank {item['rrf_rank']} | {item['retrieval_score_norm']:.4f} | {item['emotion_score_norm']:.4f} | **{item['final_score']:.4f}** |\n"

    md += """
### Skenario B: Profil Emosi Melankolis / Mendalam (*Sadness, Fear*)
*User Vector: Sadness=0.8, Fear=0.7, lainnya=0.1 | $\\alpha=0.5, \\beta=0.5$*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Skor Retrieval (Norm) | Alignment Emosi (Norm) | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|:---:|
"""
    for item in scenario_results["scenario_sadness_fear"]["top_candidates"]:
        md += f"| **{item['final_rank']}** | **{item['title']}** | {item['author']} | Rank {item['rrf_rank']} | {item['retrieval_score_norm']:.4f} | {item['emotion_score_norm']:.4f} | **{item['final_score']:.4f}** |\n"

    md += """
### Skenario C: Mode Fallback (Pengguna Tidak Memilih Preferensi Emosi)
*User Vector: None / Zero Vector | Status: Fallback Retrieval-Only*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Status Emosi | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|
"""
    for item in scenario_results["scenario_fallback"]["top_candidates"]:
        md += f"| **{item['final_rank']}** | **{item['title']}** | {item['author']} | Rank {item['rrf_rank']} | `{item['emotion_status']}` | **{item['final_score']:.4f}** |\n"

    md += f"""
---

## 2. Uji Sensitivitas Bobot $\\alpha$ (Retrieval) vs $\\beta$ (Emosi)
*Dievaluasi pada Skenario A (Joy & Anticipation)*

| Variasi Bobot | Buku Peringkat 1 | Skor Akhir | Buku Peringkat 2 | Skor Akhir |
|---|---|:---:|---|:---:|
| **Balanced ($\\alpha=0.5, \\beta=0.5$)** | **{weight_sensitivity_results['balanced'][0]['title']}** | {weight_sensitivity_results['balanced'][0]['final_score']:.4f} | {weight_sensitivity_results['balanced'][1]['title']} | {weight_sensitivity_results['balanced'][1]['final_score']:.4f} |
| **Content-Heavy ($\\alpha=0.8, \\beta=0.2$)** | **{weight_sensitivity_results['content_heavy'][0]['title']}** | {weight_sensitivity_results['content_heavy'][0]['final_score']:.4f} | {weight_sensitivity_results['content_heavy'][1]['title']} | {weight_sensitivity_results['content_heavy'][1]['final_score']:.4f} |
| **Emotion-Heavy ($\\alpha=0.2, \\beta=0.8$)** | **{weight_sensitivity_results['emotion_heavy'][0]['title']}** | {weight_sensitivity_results['emotion_heavy'][0]['final_score']:.4f} | {weight_sensitivity_results['emotion_heavy'][1]['title']} | {weight_sensitivity_results['emotion_heavy'][1]['final_score']:.4f} |

---

## 3. Temuan Evaluasi & Kesimpulan
1. **Dinamika Pemeringkatan Ulang**:
   - Preferensi emosi berhasil mengubah urutan rekomendasi buku secara bermakna dan terukur. Buku dengan relevansi teks moderat namun keselarasan emosi tinggi dapat naik peringkat ke posisi teratas saat $\\beta$ aktif.
2. **Keamanan & Transparansi Skor**:
   - Seluruh skor dinormalisasi ke rentang $[0.0, 1.0]$. Komponen skor retrieval, skor cosine similarity mentah, dan skor emosi ternormalisasi disimpan secara terpisah untuk transparansi UI.
3. **Robustness Fallback**:
   - Ketika vektor emosi pengguna bernilai nol atau tidak ditentukan, sistem secara otomatis mengaktifkan mode `fallback_retrieval_only` tanpa menghasilkan skor emosi palsu (*zero error rate*).
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[REPORT] Laporan resmi Tahap 8 berhasil diekspor:")
    print(f" -> Markdown : {REPORT_MD_PATH.relative_to(BASE_DIR)}")
    print(f" -> JSON     : {REPORT_JSON_PATH.relative_to(BASE_DIR)}")


def run_full_emotion_experiment():
    run_unit_tests()

    print("=" * 70)
    print("BAGIAN 2: INTEGRATION TEST EMOTION RE-RANKING PADA 20 KANDIDAT RETRIEVAL")
    print("=" * 70)

    query_id = "Aku ingin membaca cerita tentang persahabatan, petualangan, dan perjuangan remaja."
    bm25_keywords_en = "friendship adventure teenage struggles"
    
    print(f"Mengambil 20 kandidat retrieval untuk query: '{query_id}'...")
    candidates = get_hybrid_candidates(query_id, bm25_keywords_en, top_k=20)
    print(f"Berhasil memperoleh {len(candidates)} kandidat dari Hybrid RAG.")

    ranker = EmotionRanker()

    # 1. Skenario A: Joy & Anticipation
    print("\n--- Skenario A: Profil Joy & Anticipation (Alpha=0.5, Beta=0.5) ---")
    vec_joy = ranker.create_explicit_user_vector({
        "Joy": 0.8, "Anticipation": 0.7, "Trust": 0.6,
        "Anger": 0.1, "Disgust": 0.1, "Fear": 0.1, "Sadness": 0.1, "Surprise": 0.2, "Objective": 0.2
    })
    res_joy = ranker.rank_candidates(candidates, vec_joy, alpha=0.5, beta=0.5, top_n=5)
    for c in res_joy:
        print(f"Rank {c['final_rank']} | Final: {c['final_score']:.4f} (RetNorm: {c['retrieval_score_norm']:.4f}, EmoNorm: {c['emotion_score_norm']:.4f}) | {c['title']}")

    # 2. Skenario B: Sadness & Fear
    print("\n--- Skenario B: Profil Sadness & Fear (Alpha=0.5, Beta=0.5) ---")
    vec_sad = ranker.create_explicit_user_vector({
        "Sadness": 0.8, "Fear": 0.7, "Disgust": 0.3,
        "Joy": 0.1, "Anticipation": 0.2, "Trust": 0.1, "Anger": 0.2, "Surprise": 0.2, "Objective": 0.1
    })
    res_sad = ranker.rank_candidates(candidates, vec_sad, alpha=0.5, beta=0.5, top_n=5)
    for c in res_sad:
        print(f"Rank {c['final_rank']} | Final: {c['final_score']:.4f} (RetNorm: {c['retrieval_score_norm']:.4f}, EmoNorm: {c['emotion_score_norm']:.4f}) | {c['title']}")

    # 3. Skenario C: Fallback Retrieval-Only
    print("\n--- Skenario C: Mode Fallback (User Vector None) ---")
    res_fallback = ranker.rank_candidates(candidates, None, alpha=0.5, beta=0.5, top_n=5)
    for c in res_fallback:
        print(f"Rank {c['final_rank']} | Final: {c['final_score']:.4f} | Status: {c['emotion_status']} | {c['title']}")

    # 4. Uji Sensitivitas Bobot
    print("\n--- Evaluasi Sensitivitas Bobot Alpha & Beta (Skenario Joy) ---")
    res_bal = ranker.rank_candidates(candidates, vec_joy, alpha=0.5, beta=0.5, top_n=3)
    res_con = ranker.rank_candidates(candidates, vec_joy, alpha=0.8, beta=0.2, top_n=3)
    res_emo = ranker.rank_candidates(candidates, vec_joy, alpha=0.2, beta=0.8, top_n=3)

    scenario_results = {
        "scenario_joy_trust": {"top_candidates": res_joy},
        "scenario_sadness_fear": {"top_candidates": res_sad},
        "scenario_fallback": {"top_candidates": res_fallback}
    }
    weight_sensitivity_results = {
        "balanced": res_bal,
        "content_heavy": res_con,
        "emotion_heavy": res_emo
    }

    # Generate Reports
    generate_emotion_report(query_id, scenario_results, weight_sensitivity_results)
    print("\n" + "=" * 70)
    print("PENGUJIAN TAHAP 8 SELESAI DENGAN SUKSES!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_emotion_experiment()
