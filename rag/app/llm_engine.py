"""
MODUL 3: LLM & GROUNDED RESPONSE GENERATOR (Inference Lokal Ollama)
Fungsi Inti:
1. Query Parser   : Mengekstrak topik, kata kunci BM25, deteksi emosi & usia.
2. Local LLM      : Terhubung ke Ollama lokal (Llama 3.1 8B / Qwen 2.5 7B).
3. Grounding Guard: Menjamin LLM TIDAK mengarang buku di luar Top-N, TIDAK mengubah peringkat, dan menjelaskan keselarasan emosi sebagai preferensi (bukan jaminan perasaan mutlak).
4. Offline Engine : Fallback otomatis jika Ollama belum dinyalakan.
"""

import re
from typing import List, Dict, Any, Optional
import requests

GROUNDING_SYSTEM_PROMPT = """You are a helpful, empathetic book recommendation assistant for teenagers and young adults.

Use ONLY the book records supplied in the context as the source of information for your recommendations.

GROUNDING RULES:
1. Do NOT invent or assume any books, authors, ratings, age information, themes, content warnings, characters, settings, historical periods, or plot details that are not explicitly stated in the supplied book records.
2. Do NOT use your pretrained knowledge or external information to fill in missing details about a book, its author, or its story.
3. Explain each recommendation using ONLY the supplied Description, Genre, and Themes.
4. Do NOT force a book to match the user's request if the supplied information does not support that connection. Explain only the aspects that are genuinely relevant.
5. If a detail is missing or insufficiently supported by the supplied records, do NOT guess or make assumptions. Simply avoid mentioning that detail.
6. Preserve every book title exactly as provided. Do NOT translate, paraphrase, correct, or modify book titles.
7. Make each recommendation explanation specific to its book. Avoid generic statements such as "this book is suitable for you" without explaining why.
8. Keep explanations concise, clear, and factually grounded in the supplied records.
9. Do NOT change the ranking produced by the recommendation engine. Present books in the exact order of their Rank.
10. Describe emotion scores as emotional preference alignment (keselarasan preferensi emosi), NOT as a guarantee of how the reader will actually feel.
11. Preserve the original meaning of the supplied Description. Do NOT reverse, exaggerate, misinterpret, or change the meaning of any statement.
12. Each recommendation explanation must mention at least one concrete and relevant detail from the supplied Description, Genre, or Themes. Avoid vague statements that could apply to any book.
13. When referring to a book by title, reproduce its complete title exactly as provided. Never translate, shorten, paraphrase, or partially omit a book title.
14. Do NOT introduce personal, historical, cultural, or biographical assumptions that are not explicitly supported by the supplied book records.
15. Explain the actual connection between the user's request and the book. If the connection is weak or indirect, do not exaggerate its relevance.

OUTPUT RULES:
- Write recommendation explanations in Bahasa Indonesia.
- Use the pronoun "kamu" naturally when addressing the reader.
- Do NOT include English phrases unnecessarily.
- Do NOT add information that is not supported by the supplied book records.
- Follow the output format requested by the user or the recommendation system.

Respond warmly, naturally, and supportively in Bahasa Indonesia."""

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

    def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 150,
        extra_options: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Menghubungi endpoint lokal Ollama."""
        url = f"{self.ollama_url}/api/chat"
        try:
            options: Dict[str, Any] = {
                "temperature": temperature,
                "num_predict": max_tokens
            }
            if extra_options:
                options.update(extra_options)
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": options
            }
            print(f"[LLM Engine] Mengirim request ke Ollama ({url}) | model='{self.model_name}' | max_tokens={max_tokens}")
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                content = data["message"]["content"]
                print(f"[LLM Engine] Ollama merespons HTTP 200 | model='{self.model_name}' | panjang={len(content)} karakter")
                return content
            err_detail = (resp.text or "").strip()[:200]
            print(f"[LLM Engine] Ollama gagal | HTTP {resp.status_code} | model='{self.model_name}' | detail='{err_detail}'")
        except Exception as e:
            print(f"[LLM Engine] Error request Ollama | {type(e).__name__}: {e}")
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

        reply = self._call_ollama(messages)
        if reply and len(reply.strip()) > 50:
            print(f"[LLM Engine] Menggunakan respons Ollama ('{self.model_name}') untuk generate_recommendation ({len(reply)} karakter).")
            return reply

        # Jika Ollama belum siap/download belum selesai, gunakan Offline Fallback
        print(f"[LLM Engine] Beralih ke offline fallback untuk generate_recommendation (model='{self.model_name}').")
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
            genre = (c.get("genre", "") or "").strip()[:45]
            themes = (c.get("themes", "") or "").strip()[:35]
            desc = (c.get("description", "") or "").strip()[:120]
            parts = [f"#{rank} {title}"]
            if genre:
                parts.append(f"G:{genre}")
            if themes:
                parts.append(f"T:{themes}")
            if desc:
                parts.append(f"S:{desc}")
            book_summaries.append(" | ".join(parts))

        total_books = len(candidates)
        ctx_parts = []
        if age_cohort:
            ctx_parts.append(f"Usia:{age_cohort}")
        if active_emotions:
            ctx_parts.append("Mood:" + ",".join(active_emotions.keys()))
        meta_ctx = f" | {' | '.join(ctx_parts)}" if ctx_parts else ""
        rank_tags = ", ".join(f"#{c.get('final_rank', idx)}:" for idx, c in enumerate(candidates, start=1))

        prompt = (
            f"Q:{user_query}{meta_ctx}\n"
            + "\n".join(book_summaries)
            + f"\nTulis {total_books} baris ({rank_tags}), masing-masing 1 kalimat singkat (10-15 kata) Bahasa Indonesia sapaan 'kamu' berbasis fakta S/G/T; jangan mengarang atau terjemahkan judul."
        )

        system_prompt = (
            f"Kurator buku remaja akurat. Jawab {total_books} baris {rank_tags} saja, 1 kalimat singkat per buku pakai sapaan 'kamu'. "
            f"Gunakan hanya fakta sinopsis, genre, dan tema sebagai referensi; dilarang pakai info eksternal atau mengarang detail. "
            f"Judul dan metadata hanya referensi: jangan tulis judul, jangan terjemahkan judul, dan jangan keluarkan label G:, T:, atau S:. "
            f"Sebutkan detail spesifik tanpa alasan generik. "
            f"Respond warmly, naturally, and supportively in Bahasa Indonesia."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        num_predict = 140 if total_books <= 3 else max(140, total_books * 50)
        num_ctx = 512 if total_books <= 3 else 1024
        reply = self._call_ollama(
            messages,
            temperature=0.3,
            max_tokens=num_predict,
            extra_options={"num_ctx": num_ctx, "num_batch": 512, "num_thread": 8}
        )
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

            parsed_count = len(reasons_map)
            if parsed_count == 0:
                print(f"[LLM Engine] Ollama merespons ({len(reply)} karakter), tetapi parsing alasan gagal (0/{total_books} buku). Semua buku beralih ke offline fallback.")
            elif parsed_count < total_books:
                missing_ranks = [c.get("final_rank", 1) for c in candidates if c.get("final_rank", 1) not in reasons_map]
                print(f"[LLM Engine] Parsing alasan parsial: {parsed_count}/{total_books} buku dari Ollama ('{self.model_name}'); rank {missing_ranks} beralih ke offline fallback.")
            else:
                print(f"[LLM Engine] Parsing alasan berhasil: {parsed_count}/{total_books} buku menggunakan respons Ollama ('{self.model_name}').")
        else:
            print(f"[LLM Engine] Ollama tidak memberikan respons (model='{self.model_name}'). Semua {total_books} buku beralih ke offline fallback.")

        return reasons_map
