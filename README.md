# Sistem Rekomendasi Buku Remaja (Hybrid RAG + 9-D Emotion Vector)

Aplikasi Chatbot Rekomendasi Buku Remaja yang menggabungkan:
1. **Hybrid RAG**: Pencarian padat semantik (*Multilingual E5 Large*) + pencarian kata kunci leksikal (*BM25Okapi*) dengan *Reciprocal Rank Fusion (RRF)*.
2. **Emotion Re-Ranking**: Penilaian keselarasan 9 dimensi emosi psikologis (*NRC-EIL*: Joy, Trust, Anticipation, Surprise, Sadness, Fear, Anger, Disgust, Valence) menggunakan *Cosine Similarity* dan fusi bobot linear $\alpha$ (konten) & $\beta$ (emosi).
3. **LLM Grounding**: Menggunakan model lokal Ollama (`llama3.1:8b` atau `qwen2.5:7b`) untuk analisis konteks, kueri parsing, dan penjelasan rekomendasi yang akurat.

---

## 🏛️ Arsitektur Sederhana 3 Engine (Mudah Dijelaskan di PPT)

Semua logika inti sistem telah disederhanakan ke dalam **3 file engine utama** di dalam folder `rag/app/`:

```
PROJECT RS/
├── rag/
│   └── app/
│       ├── rag_engine.py        # [Engine 1] Document builder, E5 Dense, BM25 Sparse, RRF
│       ├── emotion_engine.py    # [Engine 2] 9-D NRC Emotion Vector, Cosine Similarity, Fusi Alpha/Beta
│       ├── llm_engine.py        # [Engine 3] Query parser, Ollama connector (Llama 3.1 & Qwen 2.5), Grounding
│       └── api.py               # REST API FastAPI penghubung Frontend & 3 Engine
├── frontend/                    # Web Interface berbasis React 18 + Vite + Lucide Icons
└── README.md                    # Dokumentasi lengkap sistem
```

### 1. `rag_engine.py` (Hybrid RAG Engine)
* **Input**: Kueri teks (Bahasa Indonesia) + Kata kunci terjemahan (Inggris).
* **Proses**:
  - Dense retrieval via ChromaDB (`intfloat/multilingual-e5-large`).
  - Sparse retrieval via BM25Okapi (`rank_bm25`).
  - Fusi peringkat timbal-balik: $RRF(d) = \sum \frac{1}{k + r(d)}$ dengan konstanta $k = 60$.
* **Output**: Top candidate books beserta teks korpus.

### 2. `emotion_engine.py` (Emotion Engine)
* **Input**: Kandidat buku dari RAG + Vektor emosi pengguna (9 dimensi).
* **Proses**:
  - Ambil vektor emosi buku dari companion store (`O(1)`).
  - Hitung *Cosine Similarity* antara emosi pengguna dan emosi buku.
  - Normalisasi Min-Max terhadap skor RRF dan skor emosi.
  - Fusi skor linear:
    $$S_{final} = \alpha \cdot S_{retrieval\_norm} + \beta \cdot S_{emotion\_norm}$$
* **Output**: Daftar buku terurut berdasarkan keselarasan konten dan emosi.

### 3. `llm_engine.py` (LLM Engine)
* **Input**: Kueri awal pengguna + Top buku hasil re-ranking.
* **Proses**:
  - Mengekstrak maksud bacaan, kelompok usia, dan emosi pengguna.
  - Menghubungkan ke Ollama lokal (`llama3.1:8b` atau `qwen2.5:7b`).
  - Menghasilkan narasi penjelasan rekomendasi berbasis fakta buku (grounded).
* **Output**: Penjelasan natural bahasa Indonesia dan verifikasi konteks.

---

## 📋 Format Output Kartu Buku di Frontend

Setiap kartu rekomendasi buku di frontend menyajikan 5 informasi wajib sesuai permintaan:
1. **📖 Judul**: Nama judul buku.
2. **✍️ Penulis**: Pengarang/penulis buku.
3. **⭐ Rating**: Nilai rata-rata ulasan pembaca (skala 1 - 5).
4. **📝 Sinopsis Singkat**: Ringkasan cerita / deskripsi buku.
5. **💡 Alasan Rekomendasi**: Penjelasan grounded mengapa buku tersebut cocok dengan kueri dan mood emosi pembaca.
*(Tersedia juga tombol collapsible untuk melihat detail skor transparansi RRF & Cosine Emosi untuk bahan presentasi).*

---

## 🚀 Cara Menjalankan Aplikasi

### Langkah 1: Pastikan Ollama Aktif
Ollama berjalan di latar belakang dengan model yang sudah terpasang:
```bash
ollama list
# Memastikan llama3.1:8b dan/atau qwen2.5:7b tersedia
```

### Langkah 2: Jalankan Backend API (Terminal 1)
Jalankan perintah berikut di terminal:
```bash
python -m uvicorn rag.app.api:app --host 127.0.0.1 --port 8000
```
Backend siap di `http://127.0.0.1:8000`. Dokumentasi Swagger interaktif di `http://127.0.0.1:8000/docs`.

### Langkah 3: Jalankan Frontend Vite (Terminal 2)
Jalankan perintah berikut di terminal:
```bash
cd frontend
npm run dev
```
Buka browser di: **`http://localhost:5173`**.
