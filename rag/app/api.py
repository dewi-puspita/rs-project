"""
=============================================================================
BACKEND API SERVICE (FastAPI + Uvicorn)
=============================================================================
Menghubungkan 3 Engine Inti (RAG Engine + Emotion Engine + LLM Engine)
menjadi REST API untuk Frontend (Vite / React).
=============================================================================
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from rag.app.rag_engine import HybridRAGEngine
from rag.app.emotion_engine import EmotionRecommenderEngine, EXPECTED_EMOTIONS
from rag.app.llm_engine import LLMEngine

app = FastAPI(
    title="Emotion & Hybrid RAG Book Recommendation API",
    description="Backend API untuk Chatbot Rekomendasi Buku Remaja berbasis Hybrid RAG dan Emotion Vector",
    version="2.0.0"
)

# Aktifkan CORS agar frontend Vite (http://localhost:5173 dsb) dapat mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Singleton Engines
print("[API Server] Memuat modul Hybrid RAG, Emotion, dan LLM Engines...")
rag_engine = HybridRAGEngine()
emotion_engine = EmotionRecommenderEngine()
llm_engine = LLMEngine(model_name="llama3.1:8b")
print("[API Server] Semua Engine siap melayani permintaan!")


CANONICAL_GENRES = [
    {
        "name": "Romance",
        "label": "Romance",
        "keywords": ["romance", "romantic", "love", "first loves", "dating", "boyfriend", "girlfriend", "kiss", "crush", "lover", "sweetheart"]
    },
    {
        "name": "School Life",
        "label": "School Life",
        "keywords": ["school", "high school", "academy", "classmate", "student", "teacher", "elementary school", "sweet valley"]
    },
    {
        "name": "Friendship",
        "label": "Friendship",
        "keywords": ["friendship", "friend", "best friends", "peers", "buddy", "companion"]
    },
    {
        "name": "Coming-of-Age",
        "label": "Coming-of-Age",
        "keywords": ["coming-of-age", "coming of age", "adolescence", "adolescent", "growing up", "teenager", "teenagers", "youth", "identity"]
    },
    {
        "name": "Fantasy",
        "label": "Fantasy",
        "keywords": ["fantasy", "magic", "magical", "wizard", "witch", "spell", "supernatural", "mythology", "fairy"]
    },
    {
        "name": "Adventure",
        "label": "Adventure",
        "keywords": ["adventure", "adventurers", "quest", "journey", "expedition", "exploration", "pirates", "survival"]
    },
    {
        "name": "Mystery",
        "label": "Mystery",
        "keywords": ["mystery", "detective", "investigation", "crime", "secret", "clue", "sherlock", "solve"]
    },
    {
        "name": "Sci-Fi",
        "label": "Sci-Fi",
        "keywords": ["science fiction", "sci-fi", "aliens", "extraterrestrial", "space", "dystopia", "dystopian", "future", "robot"]
    },
    {
        "name": "Thriller & Horror",
        "label": "Thriller & Horror",
        "keywords": ["horror", "thriller", "suspense", "ghost", "haunted", "fear", "vampire", "dark", "chilling"]
    },
    {
        "name": "Drama",
        "label": "Drama",
        "keywords": ["drama", "family life", "family", "parent", "sister", "brother", "tragedy", "emotional", "resilience"]
    },
    {
        "name": "Comedy",
        "label": "Comedy",
        "keywords": ["comedy", "humor", "humorous", "funny", "laughter", "hilarious", "parody"]
    }
]


def extract_real_genres(genre_str: str, themes_str: str, desc: str, title: str) -> str:
    """Mengekstrak 1-2 genre buku yang sesungguhnya (bukan subjek teknis perpustakaan)."""
    text_corpus = f"{genre_str or ''} {themes_str or ''} {desc or ''} {title or ''}".lower()
    matched = []
    for g in CANONICAL_GENRES:
        score = 0
        for kw in g["keywords"]:
            if kw in text_corpus:
                if kw in (genre_str or '').lower() or kw in (themes_str or '').lower():
                    score += 2
                else:
                    score += 1
        if score > 0:
            matched.append((g["label"], score))
            
    if not matched:
        return "Young Adult (Remaja)"
        
    matched.sort(key=lambda x: x[1], reverse=True)
    top_genres = [m[0] for m in matched[:2]]
    return " & ".join(top_genres)


def build_natural_reason(
    book: Dict[str, Any],
    user_query: str,
    age_cohort: Optional[str] = None,
    active_emotions: Optional[Dict[str, float]] = None
) -> str:
    """Menyusun alasan rekomendasi yang natural, kontekstual, dan masuk akal (human-like)."""
    desc = book.get("description", "").lower()
    genre_raw = book.get("genre", "")
    themes_raw = book.get("themes", "")
    title = book.get("title", "")
    real_genre = extract_real_genres(genre_raw, themes_raw, desc, title)
    q_lower = user_query.lower()
    
    # 1. Tentukan fokus cerita yang cocok dengan kueri
    hook = ""
    if any(k in q_lower for k in ["romantis", "cinta", "manis", "senyum", "love", "pacaran"]):
        if any(k in desc for k in ["love", "romance", "fall", "kiss", "heart", "boyfriend", "relationship", "crush"]) or "Romance" in real_genre:
            hook = f"Buku bergenre {real_genre} ini menyajikan kisah percintaan masa remaja yang manis dan menghangatkan hati, pas dengan ceritamu yang ingin bacaan bikin tersenyum."
        elif "School Life" in real_genre or "Friendship" in real_genre:
            hook = f"Buku bergenre {real_genre} ini menghadirkan dinamika interaksi masa sekolah yang manis dan menyenangkan, cocok menemani waktu santai."
        else:
            hook = f"Menghadirkan kisah {real_genre} dengan alur yang ringan dan sentuhan cerita yang manis."
    elif any(k in q_lower for k in ["galau", "sedih", "coming of age", "coming-of-age", "persahabatan", "sahabat"]):
        hook = f"Buku bergenre {real_genre} ini mengangkat dinamika emosi dan hangatnya persahabatan, sangat relate bagi yang sedang mencari kisah pendewasaan diri (coming-of-age)."
    elif any(k in q_lower for k in ["fantasi", "sihir", "misteri", "petualangan", "distopia", "tegang"]):
        hook = f"Buku bergenre {real_genre} ini menyuguhkan petualangan seru yang memikat dengan teka-teki dan kejutan menarik di setiap babnya."
    else:
        hook = f"Buku bergenre {real_genre} ini memiliki kesesuaian cerita yang kuat dengan pencarianmu dan menghadirkan alur yang seru untuk diikuti."

    # 2. Kalimat usia
    age_phrase = ""
    if age_cohort:
        if age_cohort in ("12-13", "14-15"):
            age_phrase = f" Karakter dan konfliknya sangat pas untuk pembaca usia {age_cohort} tahun."
        else:
            age_phrase = f" Dinamika cerita dan kedalaman karakternya sangat cocok untuk rentang usia {age_cohort} tahun."

    # 3. Kalimat nuansa emosi
    emo_phrase = ""
    emo_norm = book.get("emotion_score_norm")
    if emo_norm is not None and active_emotions:
        primary_emo = max(active_emotions.items(), key=lambda x: x[1])[0].lower() if active_emotions else "joy"
        if primary_emo in ["joy", "ceria", "senang"]:
            emo_phrase = " Nuansa ceritanya membawa hawa positif dan keceriaan yang selaras dengan mood yang kamu cari."
        elif primary_emo in ["sadness", "sedih"]:
            emo_phrase = " Sentuhan emosionalnya mampu meluapkan perasaan dan menyentuh hati pembaca."
        elif primary_emo in ["anticipation", "petualangan"]:
            emo_phrase = " Membangkitkan rasa penasaran dan antusiasme tinggi untuk terus membaca."
        else:
            emo_phrase = f" Keselarasan nuansa emosinya mencapai {emo_norm:.1%} dengan preferensi membaca kamu."

    return f"{hook}{age_phrase}{emo_phrase}".strip()


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Kueri atau cerita keinginan membaca dari pengguna")
    model_name: str = Field(default="llama3.1:8b", description="Model Ollama: 'llama3.1:8b' atau 'qwen2.5:7b'")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Bobot komponen retrieval (0.0 s.d 1.0)")
    beta: float = Field(default=0.5, ge=0.0, le=1.0, description="Bobot komponen emosi (0.0 s.d 1.0)")
    emotions: Optional[Dict[str, float]] = Field(default=None, description="Bobot emosi eksplisit jika dipilih pengguna")
    age_range: Optional[str] = Field(default=None, description="Kohor usia pembaca: '12-13', '14-15', '16-17', '18-19' sesuai Paper RecSys '25")
    top_n: int = Field(default=5, ge=1, le=20, description="Jumlah rekomendasi buku yang dihasilkan")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Book Recommendation System API",
        "supported_models": ["llama3.1:8b", "qwen2.5:7b"],
        "available_emotions": EXPECTED_EMOTIONS
    }


@app.post("/api/recommend")
def recommend_books(req: RecommendationRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Kueri pengguna tidak boleh kosong.")

    # 1. Atur model LLM aktif
    llm_engine.set_model(req.model_name)

    # 2. Parsing Kueri Pengguna
    parsed = llm_engine.parse_user_query(req.query)
    
    # Tentukan kohor usia efektif (dari request atau deteksi teks)
    effective_age = req.age_range
    if not effective_age and parsed.get("detected_age"):
        a = parsed["detected_age"]
        if a <= 13:
            effective_age = "12-13"
        elif a <= 15:
            effective_age = "14-15"
        elif a <= 17:
            effective_age = "16-17"
        else:
            effective_age = "18-19"

    # Gabungkan emosi yang terdeteksi dari teks dengan emosi yang dipilih dari UI (jika ada)
    active_emotions = dict(parsed.get("detected_emotions", {}))
    if req.emotions:
        for em, val in req.emotions.items():
            if em in EXPECTED_EMOTIONS and val > 0.0:
                active_emotions[em] = max(active_emotions.get(em, 0.0), float(val))

    # Kalibrasi preferensi emosi sesuai temuan Paper RecSys '25 (Section 3.5)
    # Jika usia 12-13: afinitas kuat pada Joy & Surprise, minim emosi negatif intens
    if effective_age == "12-13" and not req.emotions:
        active_emotions.setdefault("joy", 0.6)
        active_emotions.setdefault("surprise", 0.5)

    # 3. Hybrid RAG Search (Dense E5 + Sparse BM25 + RRF)
    keywords_en = parsed["bm25_keywords_en"]
    if effective_age == "12-13":
        keywords_en += " middle grade early teen"
    elif effective_age in ("18-19",):
        keywords_en += " mature young adult coming of age"

    retrieval_candidates = rag_engine.search(
        query_id=parsed["query_id"],
        keywords_en=keywords_en,
        top_k=20
    )

    # 4. Emotion Re-Ranking (9 Dimensi + Fusi Alpha/Beta)
    user_vec = emotion_engine.create_user_emotion_vector(active_emotions)
    ranked_candidates = emotion_engine.re_rank(
        retrieval_candidates=retrieval_candidates,
        user_emotion_vector=user_vec,
        alpha=req.alpha,
        beta=req.beta,
        top_n=req.top_n
    )

    # 5. Narasi LLM panjang dilewati demi respons instan (~1 detik)
    # Catatan: Fungsi llm_engine.generate_recommendation tetap tersedia di llm_engine.py
    explanation_text = ""

    # 6. Format Respons terstruktur sesuai permintaan spesifik:
    # (Judul, Penulis, Rating, Sinopsis Singkat, Alasan)
    formatted_books = []
    for b in ranked_candidates:
        desc = b.get("description", "")
        sinopsis = desc[:280] + "..." if len(desc) > 280 else (desc or "Sinopsis tidak tersedia.")
        
        # Ekstrak genre buku yang sesungguhnya (bukan subjek perpustakaan)
        genre_raw = b.get("genre", "")
        themes_raw = b.get("themes", "")
        cleaned_genre = extract_real_genres(genre_raw, themes_raw, desc, b.get("title", ""))
        
        # Susun alasan rekomendasi yang natural, kontekstual, dan ramah
        rank = b.get("final_rank", 1)
        alasan = build_natural_reason(b, req.query, effective_age, active_emotions)

        formatted_books.append({
            "rank": rank,
            "judul": b.get("title", "Judul tidak tersedia"),
            "penulis": b.get("author", "Penulis tidak tersedia"),
            "rating": round(float(b.get("average_rating", 0.0)), 2),
            "sinopsis_singkat": sinopsis,
            "alasan": alasan,
            "genre": cleaned_genre,
            "themes": themes_raw,
            "scores": {
                "final_score": b.get("final_score"),
                "retrieval_score_norm": b.get("retrieval_score_norm"),
                "emotion_score_norm": b.get("emotion_score_norm"),
                "rrf_score": b.get("rrf_score")
            }
        })

    return {
        "query": req.query,
        "model_used": req.model_name,
        "alpha": req.alpha,
        "beta": req.beta,
        "detected_emotions": active_emotions,
        "age_range": effective_age,
        "detected_age": parsed.get("detected_age"),
        "total_books": len(formatted_books),
        "books": formatted_books,
        "explanation": explanation_text
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag.app.api:app", host="127.0.0.1", port=8000, reload=False)
