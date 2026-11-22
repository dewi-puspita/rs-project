"""
Modul Reciprocal Rank Fusion (RRF).
Sesuai Rantug v2 (Bab 7.3 & Bab 8.3).
Rumus: RRF(b) = sum_r [ 1 / (k + rank_r(b)) ]
"""

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_candidates: int = 30
) -> List[Dict[str, Any]]:
    """
    Menggabungkan ranking Dense (E5) dan Sparse (BM25) menggunakan RRF.
    
    Argumen:
    - dense_results: list dict [{'book_id': ..., 'rank': int, 'dense_score': float}]
    - sparse_results: list dict [{'book_id': ..., 'rank': int, 'bm25_score': float}]
    - k: konstanta RRF (default 60, standar Cormack et al. 2009)
    - top_candidates: jumlah buku kandidat teratas yang dikembalikan
    """
    rrf_scores = {}
    record_info = {}

    # 1. Masukkan Dense Results
    for item in dense_results:
        bid = str(item["book_id"]).strip()
        rank = item["rank"]
        rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k + rank))
        if bid not in record_info:
            record_info[bid] = {
                "book_id": bid,
                "dense_rank": rank,
                "dense_score": item.get("dense_score"),
                "bm25_rank": None,
                "bm25_score": None
            }
        else:
            record_info[bid]["dense_rank"] = rank
            record_info[bid]["dense_score"] = item.get("dense_score")

    # 2. Masukkan Sparse (BM25) Results
    for item in sparse_results:
        bid = str(item["book_id"]).strip()
        rank = item["rank"]
        rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k + rank))
        if bid not in record_info:
            record_info[bid] = {
                "book_id": bid,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": rank,
                "bm25_score": item.get("bm25_score")
            }
        else:
            record_info[bid]["bm25_rank"] = rank
            record_info[bid]["bm25_score"] = item.get("bm25_score")

    # 3. Urutkan berdasarkan skor RRF tertinggi
    sorted_bids = sorted(rrf_scores.keys(), key=lambda b: rrf_scores[b], reverse=True)[:top_candidates]

    fused_candidates = []
    for final_rank, bid in enumerate(sorted_bids, start=1):
        info = record_info[bid]
        info["rrf_rank"] = final_rank
        info["rrf_score"] = round(rrf_scores[bid], 5)
        fused_candidates.append(info)

    return fused_candidates
