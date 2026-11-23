"""
=============================================================================
MODUL 3: LLM & GROUNDED RESPONSE GENERATOR (Inference Lokal Ollama)
=============================================================================
Slide PPT: "Modul Integrasi LLM & Penjelasan Rekomendasi Ter-Grounding"

Fungsi Inti:
1. Query Parser   : Mengekstrak topik, kata kunci BM25, deteksi emosi & usia.
2. Local LLM      : Terhubung ke Ollama lokal (Llama 3.1 8B / Qwen 2.5 7B).
3. Grounding Guard: Menjamin LLM TIDAK mengarang buku di luar Top-N,
                    TIDAK mengubah peringkat, dan menjelaskan keselarasan emosi
                    sebagai preferensi (bukan jaminan perasaan mutlak).
4. Offline Engine : Fallback otomatis jika Ollama belum dinyalakan.
=============================================================================
"""

import os
import re
from typing import List, Dict, Any, Optional
import requests

GROUNDING_SYSTEM_PROMPT = """You are a helpful, empathetic book recommendation assistant for teenagers and young adults.
Use ONLY the book records supplied in the context.
Rules:
1. Do NOT invent books, authors, ratings, age information, themes, content warnings, or plot details outside the context.
2. Do NOT change the ranking produced by the recommendation engine. Present books in the exact order of their Rank.
3. Explain each recommendation using the supplied Description, Genre, and Themes.
4. If a field or detail is missing, state that the information is unavailable.
5. Describe emotion scores as emotional preference alignment (keselarasan preferensi emosi), NOT as a guarantee of how the reader will actually feel.
6. Respond warmly, naturally, and supportively in Bahasa Indonesia."""

EMOTION_KEYWORDS_MAP = {
    "Joy": ["bahagia", "senang", "ceria", "gembira", "menyenangkan", "lucu", "humor", "komedi", "hangat", "happy", "joy"],
    "Sadness": ["sedih", "haru", "pilu", "duka", "menangis", "patah hati", "melankolis", "kehilangan", "galau", "sad", "sadness"],
    "Fear": ["takut", "horor", "horror", "seram", "menyeramkan", "mencekam", "tegang", "menegangkan", "menakutkan", "panik", "hantu", "thriller", "fear", "scary", "spooky", "creepy"],
    "Anger": ["marah", "emosi", "kesal", "konflik", "balas dendam", "benci", "amarah", "anger"],
    "Surprise": ["kejutan", "twist", "plot twist", "tak terduga", "misterius", "kaget", "surprise"],
    "Trust": ["persahabatan", "sahabat", "percaya", "kesetiaan", "loyalitas", "keluarga", "teman", "trust"],
    "Anticipation": ["petualangan", "penasaran", "antusias", "eksplorasi", "tantangan", "misi", "perjuangan", "adventure", "anticipation"],
    "Disgust": ["jijik", "muak", "disgust"],
    "Objective": ["fakta", "sejarah", "biografi", "dokumenter", "nonfiksi", "objective"]
}

INDONESIAN_TO_ENGLISH_TERMS = {
    "persahabatan": "friendship",
    "sahabat": "friends friendship",
    "teman": "friends friendship",
    "petualangan": "adventure",
    "perjuangan": "struggle struggles",
    "remaja": "teenager teen young adult",
    "sekolah": "high school school",
    "cinta": "love romance",
    "romantis": "romantic romance",
    "misteri": "mystery detective",
    "detektif": "detective crime mystery",
    "kasus": "case investigation mystery",
    "kriminal": "crime criminal investigation",
    "pembunuhan": "murder crime mystery",
    "sihir": "magic wizard fantasy",
    "keluarga": "family",
    "fantasi": "fantasy magical",
    "sedih": "sad emotional",
    "galau": "heartbreak sad melancholic",
    "patah hati": "heartbreak breakup sad",
    "bahagia": "happy uplifting",
    "keberanian": "courage bravery",
    "horor": "horror scary",
    "horror": "horror scary",
    "seram": "scary horror creepy",
    "menyeramkan": "scary horror creepy",
    "tegang": "suspense thriller suspenseful",
    "menegangkan": "suspense thriller suspenseful",
    "mencekam": "eerie chilling suspense",
    "hantu": "ghost haunted horror",
    "thriller": "thriller suspense",
    "takut": "fear scary horror",
    "distopia": "dystopia dystopian",
    "fiksi ilmiah": "sci-fi science fiction",
    "alien": "alien extraterrestrial",
    "komedi": "comedy humorous funny",
    "lucu": "funny humor comedy"
}


class LLMEngine:
    def __init__(self, model_name: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def set_model(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def parse_user_query(text: str) -> Dict[str, Any]:
        """Menganalisis kueri pengguna untuk mengekstrak kueri ID, kata kunci BM25, emosi, dan usia."""
        clean_text = text.strip()
        lower = clean_text.lower()

        # Deteksi Usia
        age = None
        m_age = re.search(r'\b(?:umur|usia|berumur)?\s*(\d{1,2})\s*(?:tahun|thn|yo)?\b', lower)
        if m_age:
            val = int(m_age.group(1))
            if 10 <= val <= 25:
                age = val

        # Deteksi Emosi Eksplisit
        detected_emotions: Dict[str, float] = {}
        for dim, kws in EMOTION_KEYWORDS_MAP.items():
            for kw in kws:
                if re.search(rf'\b{re.escape(kw)}\b', lower):
                    detected_emotions[dim] = max(detected_emotions.get(dim, 0.0), 0.75)

        # Kata Kunci BM25 Bahasa Inggris
        en_words = []
        for id_t, en_t in INDONESIAN_TO_ENGLISH_TERMS.items():
            if id_t in lower:
                en_words.append(en_t)

        # Cek kata kunci pinjaman / bahasa Inggris langsung
        direct_genres = ["horror", "thriller", "suspense", "mystery", "fantasy", "romance", "sci-fi", "dystopia", "detective", "adventure", "ghost", "vampire", "crime", "magic"]
        for term in direct_genres:
            if re.search(rf'\b{re.escape(term)}\b', lower):
                en_words.append(term)

        bm25_keywords = " ".join(dict.fromkeys(" ".join(en_words).split())) if en_words else "young adult teenage book story"

        return {
            "query_id": clean_text,
            "bm25_keywords_en": bm25_keywords,
            "detected_emotions": detected_emotions,
            "detected_age": age
        }

    def _call_ollama(self, messages: List[Dict[str, str]], temperature: float = 0.5, max_tokens: int = 150) -> Optional[str]:
        """Menghubungi endpoint lokal Ollama."""
        try:
            url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                return data["message"]["content"]
        except Exception:
            pass
        return None

    @staticmethod
    def _generate_offline_grounded_response(user_query: str, candidates: List[Dict[str, Any]]) -> str:
        """Generator respons grounded offline deterministik jika Ollama belum dinyalakan."""
        lines = [
            f"Halo! Berdasarkan kueri kamu: *\"{user_query}\"*, berikut adalah {len(candidates)} buku rekomendasi terbaik yang paling sesuai:\n"
        ]

        for c in candidates:
            rank = c.get("final_rank", 1)
            title = c.get("title", "Judul tidak tersedia")
            author = c.get("author", "Penulis tidak tersedia")
            rating = c.get("average_rating", 0.0)
            genre = c.get("genre", "Tidak tersedia")
            themes = c.get("themes", "Tidak tersedia")
            final_score = c.get("final_score", 0.0)
            emo_score = c.get("emotion_score_norm")
            emo_label = f"{emo_score:.1%}" if emo_score is not None else "Mode Netral"

            desc = c.get("description", "")
            snippet = desc[:220] + "..." if len(desc) > 220 else desc

            lines.append(f"### {rank}. **{title}**")
            lines.append(f"- **Penulis**: {author} | ⭐ **Rating**: {rating:.2f}/5.0")
            lines.append(f"- **Genre & Tema**: {genre} | *{themes}*")
            lines.append(f"- **Skor Rekomendasi**: `{final_score:.4f}` (Keselarasan Emosi: `{emo_label}`)")
            if snippet:
                lines.append(f"- **Sinopsis Singkat**: *\"{snippet}\"*")
            lines.append(
                f"- **Alasan**: Buku ini menempati peringkat ke-{rank} karena memiliki kecocokan topik pencarian yang sangat kuat dan profil emosi yang selaras dengan suasana hatimu.\n"
            )

        lines.append(
            "> 💡 *Catatan: Skor emosi menunjukkan tingkat keselarasan preferensi profil emosi dengan nuansa buku, bukan jaminan perasaan mutlak saat membaca.*"
        )
        return "\n".join(lines)

    def generate_recommendation(
        self,
        user_query: str,
        candidates: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Menghasilkan teks rekomendasi yang ter-grounding."""
        if not candidates:
            return "Maaf, sistem tidak menemukan buku yang relevan dengan kriteria pencarian tersebut."

        # Format konteks buku untuk LLM
        context_items = []
        for c in candidates:
            context_items.append(
                f"[Peringkat {c.get('final_rank')}]\n"
                f"- Judul: {c.get('title')}\n"
                f"- Penulis: {c.get('author')}\n"
                f"- Rating: {c.get('average_rating')}\n"
                f"- Genre/Tema: {c.get('genre')} / {c.get('themes')}\n"
                f"- Skor Akhir: {c.get('final_score'):.4f} (Keselarasan Emosi: {c.get('emotion_score_norm')})\n"
                f"- Sinopsis: {c.get('description')}\n"
            )
        context_str = "\n".join(context_items)

        prompt = (
            f"Pertanyaan Pengguna:\n\"{user_query}\"\n\n"
            f"Daftar Buku Rekomendasi:\n{context_str}\n"
            f"Tugasmu: Jelaskan buku-buku di atas secara ramah dalam Bahasa Indonesia, "
            f"pertahankan urutan peringkatnya, sebutkan penulis, rating, dan alasan kecocokannya."
        )

        messages = [{"role": "system", "content": GROUNDING_SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-4:])
        messages.append({"role": "user", "content": prompt})

        # Coba panggil Ollama
        reply = self._call_ollama(messages)
        if reply and len(reply.strip()) > 50:
            return reply

        # Jika Ollama belum siap/download belum selesai, gunakan Offline Fallback
        return self._generate_offline_grounded_response(user_query, candidates)

    def generate_reasons_for_books(
        self,
        user_query: str,
        candidates: List[Dict[str, Any]],
        age_cohort: Optional[str] = None,
        active_emotions: Optional[Dict[str, float]] = None
    ) -> Dict[int, str]:
        """
        Menghasilkan alasan rekomendasi per buku langsung dari model LLM terpilih (Llama/Qwen).
        Mengembalikan dictionary: {rank: 'alasan dari LLM'}
        """
        if not candidates:
            return {}

        book_summaries = []
        for c in candidates:
            rank = c.get("final_rank", 1)
            title = c.get("title", "")
            author = c.get("author", "")
            genre = c.get("genre", "")
            desc = (c.get("description", "") or "")[:180]
            book_summaries.append(
                f"Buku #{rank}: '{title}' karya {author}\n"
                f"- Genre: {genre}\n"
                f"- Sinopsis: {desc}\n"
            )

        age_info = f"Rentang usia pembaca: {age_cohort} tahun." if age_cohort else ""
        emo_str = ", ".join([f"{k} ({v})" for k, v in active_emotions.items()]) if active_emotions else "Netral"
        emo_info = f"Mood/emosi pembaca: {emo_str}."

        prompt = (
            f"Kebutuhan Pembaca: \"{user_query}\"\n{age_info} {emo_info}\n\n"
            f"Daftar Buku Rekomendasi:\n{''.join(book_summaries)}\n"
            f"Instruksi: Tuliskan tepat 1 kalimat alasan rekomendasi per nomor buku dalam Bahasa Indonesia. "
            f"Langsung jelaskan kesesuaian cerita dan daya tariknya (JANGAN hanya mengulang judul atau nama penulis).\n\n"
            f"Contoh format yang WAJIB diikuti:\n"
            f"#1: Buku ini sangat pas karena menyajikan ketegangan misteri yang mencekam dan alur cerita yang tak terduga.\n"
            f"#2: Buku ini cocok untukmu karena menghadirkan kisah horor yang intens dengan misteri mendalam.\n\n"
            f"Tuliskan jawaban untuk setiap nomor buku sekarang:\n"
        )

        messages = [
            {"role": "system", "content": "Kamu adalah kurator buku remaja yang ramah. Berikan alasan rekomendasi 1 kalimat yang hangat per buku."},
            {"role": "user", "content": prompt}
        ]

        reply = self._call_ollama(messages, temperature=0.6, max_tokens=400)
        reasons_map: Dict[int, str] = {}
        if reply:
            for line in reply.strip().split("\n"):
                line = line.strip()
                m = re.match(r'^#?(\d+)[\s:\.\-]+(.*)$', line)
                if m:
                    r_idx = int(m.group(1))
                    r_text = m.group(2).strip()
                    # Bersihkan jika ada tanda kurung siku atau awalan berulang
                    r_text = re.sub(r'^\[|\]$', '', r_text).strip()
                    # Pastikan bukan sekadar nama buku/penulis dan tidak terpotong menggantung
                    if len(r_text) > 25 and not r_text.endswith(('karya', 'yang', 'dan', 'dengan', 'di', 'pada')):
                        reasons_map[r_idx] = r_text

        return reasons_map
