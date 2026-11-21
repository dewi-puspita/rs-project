# Spesifikasi Tugas Implementasi untuk Antigravity
## Chatbot Rekomendasi Buku Berbasis Hybrid RAG dan Emotion Vector

**Versi:** 2.0  
**Status:** Rancangan teknis yang telah diselaraskan dengan dataset enriched dan keputusan diskusi terbaru  
**Tujuan:** Menjadi dokumen acuan dan instruksi implementasi untuk Antigravity sebagai coding agent.

---

## 1. Ringkasan proyek

Bangun prototype **chatbot rekomendasi buku** yang mengembangkan pendekatan *Emotion Vector-Based Fine-Tuning of Large Language Models for Age-Aware Teenage Book Recommendations* dengan menambahkan kemampuan percakapan, Hybrid Retrieval-Augmented Generation (Hybrid RAG), dan pemeringkatan kandidat berdasarkan Emotion Vector yang sudah tersedia.

Sistem terdiri dari beberapa komponen utama:

1. dataset buku enriched yang sudah tersedia sebagai sumber pengetahuan;
2. Knowledge Base dengan dense retrieval dan sparse retrieval;
3. penggabungan hasil retrieval menggunakan Reciprocal Rank Fusion (RRF);
4. Emotion Companion Store untuk mengambil Emotion Vector buku berdasarkan ID;
5. penghitungan kesesuaian emosi menggunakan cosine similarity, setelah metode pembentukan preferensi emosi pengguna ditetapkan;
6. Llama 3.1 8B Instruct dan Qwen 2.5 7B Instruct sebagai LLM inference, tanpa fine-tuning;
7. chatbot yang menyampaikan rekomendasi berdasarkan buku dan metadata yang benar-benar ditemukan.

### Prinsip utama

**Emotion Vector sudah tersedia di dataset. Jangan membangun ulang proses pembentukannya.**

Dilarang melakukan hal-hal berikut tanpa keputusan baru yang eksplisit:

- menghitung ulang NRC-EIL;
- mengulang synonym matching atau WordNet untuk membentuk Emotion Vector;
- menghitung ulang agregasi atau normalisasi Emotion Vector yang sudah ada;
- melakukan fine-tuning terhadap Llama atau Qwen;
- melakukan enrichment Genre/Themes ulang;
- mengubah atau menimpa dataset sumber;
- membuat atau mengarang metadata yang tidak didukung sumber.

Tugas sistem adalah memvalidasi format dan nilai, menyimpan, mengambil, serta menggunakan Emotion Vector yang sudah tersedia.

---

## 2. Dataset final dan kondisi aktual

### 2.1 File yang digunakan

Dataset final untuk Knowledge Base adalah:

```text
Fixed_Genre_Themes_GoodReads_Emotion.csv
```

Dalam workspace, Antigravity harus menemukan file tersebut dan memeriksa lokasi aktualnya. Salah satu lokasi yang pernah dilaporkan adalah:

```text
data_pipeline/enrichment/Fixed_Genre_Themes_GoodReads_Emotion.csv
```

Jangan menganggap lokasi tersebut pasti benar sebelum memeriksa struktur repository.

Dataset ini **sudah melalui enrichment Genre dan Themes**. Tahap enrichment dianggap selesai. Jangan menjalankan enrichment ulang atau mengganti isi kolom yang sudah ada.

### 2.2 Struktur kolom aktual

Dataset yang diperiksa memiliki 19.784 baris dan 12 kolom:

| Kolom | Fungsi | Pemakaian dalam sistem |
|---|---|---|
| `ISBN` | Identitas eksternal buku | Metadata |
| `ID` | Identitas record buku | Kunci utama penghubung antarkomponen |
| `Title` | Judul buku | Document text dan tampilan |
| `Author` | Penulis | Document text dan tampilan |
| `Description` | Deskripsi/sinopsis | Sumber teks utama untuk retrieval |
| `Average_Rating` | Rata-rata rating | Metadata tampilan; bukan skor utama ranking |
| `Age` | Atribut usia dari dataset awal | Dipertahankan persis; maknanya harus mengikuti dokumentasi sumber |
| `EmotionVector` | Vektor emosi yang telah tersedia | Emotion Companion Store |
| `Genre` | Metadata kategori/subjek hasil enrichment | Metadata opsional untuk tampilan/filter/eksperimen |
| `genre_source` | Asal data Genre | Audit/provenance |
| `Themes` | Metadata tema hasil enrichment | Metadata opsional untuk tampilan/filter/eksperimen |
| `themes_source` | Asal data Themes | Audit/provenance |

**Perhatikan nama kolom aktual `EmotionVector`** (tanpa underscore). Jangan mengasumsikan nama `Emotion_Vector` dari dokumen atau rancangan lama. Gunakan konfigurasi pemetaan kolom agar perubahan kapitalisasi/nama dapat ditangani tanpa mengubah file sumber.

### 2.3 Hasil pemeriksaan dataset

Hasil pemeriksaan yang sudah dilakukan:

| Pemeriksaan | Hasil |
|---|---:|
| Total baris | 19.784 |
| Jumlah kolom | 12 |
| `Genre` terisi | 18.120 (sekitar 91,6%) |
| `Themes` terisi | 4.564 (sekitar 23,1%) |
| `Title` kosong | 38 |
| `Author` kosong | 5 |
| `Description` kosong | 8 |
| `Title` dan `Description` sama-sama kosong | 6 |
| ID kosong/duplikat | 0 |
| ISBN duplikat | 0 |
| EmotionVector berhasil diparsing dan memiliki 9 dimensi | 19.784 |

Gunakan pemeriksaan ulang terhadap file aktual saat pipeline dijalankan. Angka di atas adalah baseline hasil inspeksi saat rancangan ini dibuat; jika file berubah, laporkan selisihnya dan jangan diam-diam mengubah spesifikasi.

### 2.4 Perlakuan kolom `Age`

Kolom `Age` berasal dari dataset awal dan **tidak diubah saat enrichment**.

Aturan:

- pertahankan nilai `Age` persis seperti sumber;
- jangan melakukan normalisasi, pengelompokan ulang, atau penggantian nilai;
- jangan langsung menganggapnya sebagai minimum reading age atau jaminan kesesuaian usia;
- jangan menggunakan `Age` sebagai filter keamanan usia sampai definisinya diverifikasi dari paper/dokumentasi dataset;
- jika ditampilkan pada UI, gunakan label yang sesuai dengan arti sumber. Jika artinya belum dipastikan, jangan memberi label interpretatif.

### 2.5 Interpretasi Genre dan Themes

Isi `Genre` tidak seluruhnya merupakan genre baku yang terstandardisasi. Isinya dapat mencampur genre, kategori katalog, subjek, nama tokoh, dan istilah dari berbagai bahasa.

Karena itu:

- perlakukan `Genre` sebagai **metadata kategori/subjek katalog**, bukan selalu sebagai genre resmi;
- pertahankan `genre_source` dan `themes_source` untuk audit;
- Themes hanya tersedia pada sebagian kecil data, sehingga jangan menjadikannya metadata wajib;
- jangan menganggap Themes yang kosong berarti buku tidak memiliki tema tertentu;
- jangan menjalankan enrichment ulang.

---

## 3. Aturan preprocessing dan kelayakan indexing

### 3.1 Dataset sumber tidak diubah

File CSV sumber harus tetap utuh. Seluruh hasil preprocessing, indeks, laporan, dan file turunan disimpan pada folder output terpisah.

Jangan menghapus baris dari CSV asli hanya karena sebagian kolom teks kosong.

### 3.2 Kriteria buku yang masuk indeks retrieval

Buku memenuhi syarat masuk indeks jika setidaknya salah satu dari dua field berikut tersedia dan berisi teks yang layak:

- `Title`, atau
- `Description`.

`Author` dapat menjadi informasi tambahan, tetapi **Author saja tidak cukup** untuk menyatakan bahwa buku memiliki representasi konten yang layak untuk retrieval.

Hasil pemeriksaan menunjukkan 6 buku tidak mempunyai `Title` dan `Description`. Buku tersebut dikecualikan dari ChromaDB dan BM25.

Dengan demikian, jumlah target awal:

- dataset asli: 19.784 buku;
- buku yang masuk dense dan sparse retrieval index: 19.778 buku;
- buku yang dikecualikan dari indeks: 6 buku;
- Emotion Companion Store: tetap berisi 19.784 buku.

Antigravity harus memverifikasi kembali jumlah ini saat indexing. Jangan hard-code jumlah baris tanpa melakukan pemeriksaan.

### 3.3 Penanganan nilai kosong

Jangan memasukkan placeholder buatan ke dalam teks embedding atau korpus BM25.

Contoh aturan:

- Title kosong, Description tersedia: gunakan Description dan Author jika tersedia.
- Description kosong, Title tersedia: gunakan Title dan Author jika tersedia.
- Author kosong: abaikan bagian Author.
- Title dan Description kosong: jangan masukkan ke dense/BM25 index; tetap pertahankan record pada dataset dan Emotion Store.

Untuk tampilan chatbot, field kosong boleh ditampilkan dengan label seperti `Judul tidak tersedia` atau `Sinopsis tidak tersedia`. Label tampilan tersebut tidak boleh dimasukkan ke teks embedding.

### 3.4 Document Builder baseline

Untuk baseline awal, teks retrieval **hanya menggunakan field utama**:

```text
Title: {title_if_available}
Author: {author_if_available}
Description: {description_if_available}
```

Aturan:

- tambahkan hanya field yang memiliki isi;
- bersihkan whitespace berulang dan nilai kosong/NaN;
- jangan masukkan `Genre`, `Themes`, `genre_source`, `themes_source`, `Age`, `Average_Rating`, atau `EmotionVector` ke teks baseline embedding/BM25;
- Genre dan Themes tetap disimpan sebagai metadata terpisah;
- jangan menggabungkan metadata yang tidak tersedia menjadi string `None`, `nan`, atau placeholder;
- gunakan satu fungsi pembangun teks dasar agar teks untuk E5, BGE, dan BM25 berasal dari konten buku yang sama.

Simpan tiga representasi logis:

1. `document_text`: teks dasar Title + Author + Description;
2. `e5_passage_text`: prefix E5 + document_text;
3. `bge_passage_text`: document_text tanpa prefix/instruction.

Untuk BM25, gunakan `document_text` tanpa prefix model embedding.

---

## 4. Arsitektur sistem

Sistem mempunyai dua pipeline: **offline Knowledge Base construction** dan **online recommendation**.

### 4.1 Arsitektur tingkat sistem

```text
USER
 |
 v
CHATBOT / USER INTERFACE
 |
 v
LLM QUERY UNDERSTANDING
 - mempertahankan maksud dan konteks percakapan
 - mengekstrak topik, kata kunci, preferensi, dan batasan
 - menghasilkan query asli dan versi Inggris untuk BM25
 |
 v
QUERY ROUTING
 |
 +------------------------------------------+
 |                    |                     |
 v                    v                     v
E5 DENSE             BGE DENSE              BM25 SPARSE
Query asli           Query Inggris          Query Inggris
Bahasa pengguna      hasil terjemahan       hasil terjemahan
+ "query: "          + query instruction    tanpa prefix model
 |                    |                     |
 v                    v                     v
ChromaDB E5          ChromaDB BGE           BM25 Index
 |                    |                     |
 +--------------------+---------------------+
                      |
                      v
             RRF FUSION (per eksperimen)
                      |
                      v
             RETRIEVED BOOK CANDIDATES
                      |
                      v
             LOOKUP EMOTION VECTOR
             berdasarkan canonical book_id
                      |
                      v
             EMOTION ALIGNMENT
             cosine similarity
                      |
                      v
             FINAL RE-RANKING
             retrieval score + emotion score
                      |
                      v
             TOP-N BOOKS
                      |
                      v
             LLM GROUNDED RESPONSE
                      |
                      v
             CHATBOT RECOMMENDATIONS
```

### 4.2 Offline Knowledge Base construction

```text
Fixed_Genre_Themes_GoodReads_Emotion.csv
(19.784 baris; sumber tidak diubah)
                |
                v
      INSPECTION & VALIDATION
                |
       +--------+-----------------------+
       |                                |
       v                                v
  DOCUMENT BUILDER                EMOTION VALIDATOR
  Title + Author +                ast.literal_eval()
  Description                    validasi 9 dimensi
       |                                |
       |                                v
       |                       Emotion Companion Store
       |                       19.784 records keyed by ID
       |
       +-------------------------------+
       |
       +-----------------------+
       |                       |
       v                       v
 E5 REPRESENTATION        BGE REPRESENTATION
 passage: document_text   document_text
       |                       |
       v                       v
 E5 EMBEDDINGS            BGE EMBEDDINGS
       |                       |
       v                       v
 ChromaDB E5              ChromaDB BGE

 document_text
       |
       v
 SAME TOKENIZER FUNCTION
       |
       v
 BM25Okapi INDEX
 19.778 indexed documents
 + book_id mapping
```

### 4.3 Prinsip modular

Pisahkan minimal komponen berikut:

1. dataset loader dan validator;
2. document builder;
3. embedding/index builder;
4. BM25 index builder;
5. Emotion Companion Store;
6. query understanding dan translation;
7. dense retrieval;
8. sparse retrieval;
9. RRF fusion;
10. emotion ranker dan final ranking;
11. LLM response generator;
12. chatbot UI.

Jangan membuat seluruh alur menjadi satu fungsi besar yang sulit diuji.

---

## 5. Pemilihan embedding model: dua jalur komparasi

Dua model berikut menjadi kandidat utama eksperimen. Keduanya dipisahkan ke dalam indeks ChromaDB masing-masing.

### 5.1 Jalur A — Multilingual E5 Large

```text
Model ID: intfloat/multilingual-e5-large
Embedding dimension: 1024
Maximum input length: 512 tokens
Language: Multilingual
```

Tujuan jalur ini adalah menguji retrieval lintas bahasa secara langsung: query pengguna dapat berbahasa Indonesia, sementara deskripsi buku mayoritas berbahasa Inggris.

**Aturan prefix E5:**

- setiap teks dokumen wajib diawali `passage: `;
- setiap query dense wajib diawali `query: `;
- prefix tidak digunakan dalam BM25.

Contoh dokumen:

```text
passage: Title: Harry Potter
Author: J.K. Rowling
Description: A young wizard discovers...
```

Contoh query:

```text
query: Saya ingin membaca cerita tentang persahabatan dan petualangan remaja.
```

Jangan lupa prefix pada dokumen maupun query. Model card E5 menyatakan bahwa format prefix tersebut sesuai dengan pelatihan model dan ketiadaannya dapat menurunkan performa retrieval.

### 5.2 Jalur B — BGE Large English v1.5

```text
Model ID: BAAI/bge-large-en-v1.5
Embedding dimension: 1024
Maximum input length: 512 tokens
Language: English
```

Tujuan jalur ini adalah menguji strategi **translate-then-search**. Query pengguna diterjemahkan ke bahasa Inggris sebelum dense retrieval.

Untuk query retrieval yang pendek dan dokumen yang lebih panjang, model card BGE v1.5 merekomendasikan query instruction berikut:

```text
Represent this sentence for searching relevant passages: 
```

Contoh query dense BGE:

```text
Represent this sentence for searching relevant passages: 
I want to read a story about friendship and teenage adventure.
```

Dokumen BGE tidak menggunakan query instruction atau prefix E5.

Model card BGE v1.5 menyatakan instruction untuk query dapat digunakan untuk retrieval. Penggunaan instruction sebaiknya dicatat sebagai konfigurasi eksperimen dan dipertahankan konsisten selama pengujian.

### 5.3 Perbedaan kedua jalur

| Komponen | Jalur A: E5 | Jalur B: BGE |
|---|---|---|
| Model | `intfloat/multilingual-e5-large` | `BAAI/bge-large-en-v1.5` |
| Bahasa query dense | Bahasa asli pengguna | Bahasa Inggris hasil terjemahan |
| Bahasa dokumen | Mayoritas Inggris | Mayoritas Inggris |
| Awalan dokumen | `passage: ` | Tidak ada |
| Awalan/instruction query | `query: ` | `Represent this sentence for searching relevant passages: ` |
| Dimensi | 1.024 | 1.024 |
| Batas input | 512 token | 512 token |
| ChromaDB | Indeks tersendiri | Indeks tersendiri |

Jangan menggunakan embedding E5 untuk query pada indeks BGE atau sebaliknya. Dokumen dan query harus selalu di-embed menggunakan model serta konfigurasi yang sama.

### 5.4 Perangkat dan konfigurasi awal

Pemeriksaan Antigravity melaporkan spesifikasi perangkat sekitar:

- RAM 16 GB;
- CPU 12 cores.

Konfigurasi ini dapat digunakan untuk uji embedding lokal berbasis CPU, tetapi kecepatan dan konsumsi memori harus diukur langsung.

Konfigurasi uji awal yang disarankan:

| Parameter | Nilai awal |
|---|---|
| Device | CPU, jika GPU tidak tersedia |
| Batch size awal | 4 atau 8 |
| Max sequence length | 512 |
| Normalize embeddings | True |
| E5 dimension | 1.024 |
| BGE dimension | 1.024 |

Sebelum membangun seluruh indeks:

1. jalankan benchmark pada 500–1.000 dokumen;
2. catat waktu proses, penggunaan RAM, dan error;
3. periksa distribusi panjang token dokumen;
4. periksa jumlah dokumen yang melebihi 512 token;
5. sesuaikan batch size jika terjadi memory pressure.

Jangan menyimpulkan model pasti berjalan lancar hanya dari ukuran file bobot model. Penggunaan memori aktual juga dipengaruhi tokenizer, runtime, batch, dan proses aplikasi lain.

### 5.5 Kebijakan teks panjang

Kedua model mempunyai batas input 512 token. Karena itu:

- hitung dan laporkan jumlah dokumen yang melebihi batas;
- jangan mengubah atau memotong kolom Description pada CSV sumber;
- biarkan preprocessing embedding menangani truncation secara eksplisit dan catat konfigurasi;
- untuk baseline awal, gunakan satu representasi dokumen per buku dengan truncation yang konsisten;
- evaluasi strategi chunking secara terpisah jika truncation terbukti menghilangkan informasi penting.

Jika chunking diterapkan kemudian, hasil per chunk harus dipetakan kembali ke satu `book_id`, dan ranking beberapa chunk milik buku yang sama perlu diagregasi.

---

## 6. Offline indexing: ChromaDB, BM25, dan Emotion Companion Store

### 6.1 Canonical book ID

Gunakan `ID` sebagai kunci utama penghubung antarkomponen.

Aturan:

- normalisasi representasi ID menjadi string kanonis, misalnya `str(ID)` setelah memastikan nilai tidak kosong;
- jangan mengubah identitas ID atau membuat ID baru secara acak;
- simpan mapping ID yang sama pada ChromaDB, BM25, dan Emotion Store;
- jangan memakai nomor urut DataFrame sebagai pengganti ID;
- validasi tidak ada ID duplikat sebelum indexing.

### 6.2 Dense index dengan ChromaDB

Bangun dua indeks persistent yang terpisah:

```text
rag/
└── vector_store/
    ├── chroma_e5_large/
    └── chroma_bge_large/
```

Nama collection yang disarankan:

```text
books_multilingual_e5_large
books_bge_large_en_v1_5
```

Metadata setiap record minimal memuat:

- canonical `book_id`;
- `ISBN`;
- `Title`;
- `Author`;
- `Average_Rating`;
- `Age`;
- `Genre`;
- `genre_source`;
- `Themes`;
- `themes_source`.

Simpan `document_text` sebagai konten dokumen sesuai kebutuhan ChromaDB. `EmotionVector` tidak perlu dimasukkan ke metadata ChromaDB karena disimpan terpisah.

Jangan menggunakan satu collection untuk dua embedding model. Dimensi dan ruang representasi harus tetap terpisah.

### 6.3 Sparse index dengan BM25Okapi

Gunakan `rank_bm25` (`BM25Okapi`) sebagai baseline sparse retrieval.

- korpus BM25 memakai `document_text` tanpa prefix E5/BGE;
- korpus berasal dari 19.778 buku eligible yang sama;
- tokenisasi korpus dan query wajib menggunakan fungsi preprocessing yang konsisten;
- karena korpus deskripsi mayoritas berbahasa Inggris, gunakan konfigurasi tokenisasi yang sesuai untuk teks Inggris;
- simpan urutan `book_ids` yang sama persis dengan urutan dokumen saat BM25 index dibuat.

File output yang disarankan:

```text
rag/vector_store/bm25_index.pkl
rag/vector_store/bm25_book_ids.json
```

Jangan hanya menyimpan objek BM25 tanpa mapping ID. Indeks BM25 menghasilkan indeks posisi dokumen; posisi itu harus bisa dipetakan kembali ke canonical `book_id`.

BM25 index dapat digunakan bersama jalur dense E5 dan BGE karena korpus buku dasarnya sama.

### 6.4 Emotion Companion Store

EmotionVector tersimpan dalam kolom `EmotionVector` sebagai representasi string dictionary Python dengan tanda petik tunggal. Karena itu, gunakan:

```python
import ast

emotion_vector = ast.literal_eval(raw_emotion_vector)
```

Jangan gunakan `eval()`. Jangan mengasumsikan `json.loads()` dapat membaca string dictionary tersebut.

Urutan sembilan dimensi:

```text
Anger
Anticipation
Disgust
Fear
Joy
Sadness
Surprise
Trust
Objective
```

Validasi:

1. setiap record memiliki tepat sembilan key yang diharapkan;
2. semua nilai bertipe numerik (`int` atau `float` diterima);
3. semua nilai finite, tidak NaN dan tidak Infinity;
4. setelah validasi, konversikan nilai ke `float` untuk komputasi;
5. jangan menghitung ulang, menormalisasi ulang, atau mengubah nilai EmotionVector sumber;
6. setiap vektor harus dipasangkan dengan canonical `book_id`.

Simpan sebagai dictionary keyed by `book_id` agar lookup dari ID ke EmotionVector efisien.

File output yang disarankan:

```text
rag/vector_store/emotion_companion_store.json
```

**Store wajib mempertahankan seluruh 19.784 buku**, termasuk enam buku yang tidak masuk indeks retrieval.

Contoh struktur:

```json
{
  "12345": {
    "Anger": 0.01,
    "Anticipation": 0.12,
    "Disgust": 0.00,
    "Fear": 0.02,
    "Joy": 0.31,
    "Sadness": 0.08,
    "Surprise": 0.10,
    "Trust": 0.25,
    "Objective": 0.11
  }
}
```

Angka hanya ilustrasi format, bukan data buku aktual.

### 6.5 Laporan indexing

Setiap proses indexing harus menghasilkan laporan, minimal berisi:

- jumlah baris dataset sumber;
- jumlah record lolos validasi ID;
- jumlah buku eligible;
- jumlah buku yang dikecualikan beserta ID dan alasannya;
- jumlah record dense E5;
- jumlah record dense BGE;
- jumlah record BM25;
- jumlah EmotionVector yang berhasil disimpan;
- jumlah ID yang cocok antara indeks dan Emotion Store;
- jumlah record yang gagal diproses;
- model, versi, dimensi, device, batch size, dan max sequence length;
- waktu proses dan lokasi output.

Syarat konsistensi:

```text
E5 indexed IDs = BGE indexed IDs = BM25 indexed IDs
Emotion Store IDs = all valid dataset IDs
```

Dengan kondisi awal saat ini, targetnya 19.778 ID untuk masing-masing retrieval index dan 19.784 ID pada Emotion Store.

---

## 7. Online query processing dan query routing

### 7.1 Tujuan

Input pengguna dapat berupa bahasa Indonesia atau Inggris, ditulis secara bebas, dan dapat disampaikan dalam beberapa giliran percakapan.

LLM query understanding bertugas menghasilkan representasi query yang terstruktur, bukan memilih buku atau menentukan skor ranking.

### 7.2 Structured Query

Skema output yang disarankan:

```python
class StructuredQuery(BaseModel):
    raw_user_query: str
    semantic_query_original: str
    semantic_query_en: str | None = None
    bm25_query_en: str
    topics: list[str] = []
    genre_preferences: list[str] = []
    emotion_preferences: list[str] = []
    content_constraints: list[str] = []
    age: int | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
```

Catatan:

- `semantic_query_original` mempertahankan bahasa dan maksud pengguna untuk jalur E5;
- `semantic_query_en` merupakan versi Inggris untuk dense BGE;
- `bm25_query_en` merupakan versi Inggris untuk sparse BM25 pada kedua jalur;
- jika query awal sudah berbahasa Inggris, proses penerjemahan dapat mempertahankan teks aslinya dengan normalisasi seperlunya;
- jangan menambahkan topik, genre, emosi, atau batasan yang tidak disebutkan atau tidak didukung konteks percakapan;
- field tidak diketahui tetap `null` atau list kosong.

### 7.3 Jalur query untuk setiap retriever

Satu input pengguna menghasilkan query yang diarahkan ke tiga komponen:

| Komponen | Query yang dipakai | Aturan |
|---|---|---|
| E5 Dense | `query: {semantic_query_original}` | Bahasa asli pengguna; E5 bersifat multilingual |
| BGE Dense | `Represent this sentence for searching relevant passages: {semantic_query_en}` | Bahasa Inggris hasil terjemahan |
| BM25 | `bm25_query_en` | Bahasa Inggris; tanpa prefix model |

Contoh input:

```text
Aku ingin membaca cerita tentang persahabatan, petualangan,
dan perjuangan remaja.
```

Dense E5:

```text
query: Aku ingin membaca cerita tentang persahabatan, petualangan, dan perjuangan remaja.
```

Dense BGE:

```text
Represent this sentence for searching relevant passages:
I want to read a story about friendship, adventure, and teenage struggles.
```

BM25:

```text
friendship adventure teenage struggles
```

Query BM25 pada contoh tersebut hanya ilustrasi. Sistem harus menjaga maksud asli pengguna, bukan sekadar menerjemahkan kata demi kata atau menambah keyword yang tidak relevan.

### 7.4 Konsistensi penerjemahan untuk eksperimen

Untuk membandingkan E5 dan BGE secara adil:

- gunakan query test yang sama;
- gunakan satu konfigurasi translation/query understanding yang sama untuk membuat versi Inggris;
- simpan query asli dan hasil terjemahannya;
- gunakan versi `bm25_query_en` yang sama untuk BM25 pada kedua eksperimen;
- jangan membandingkan E5 dengan query hasil satu LLM dan BGE dengan query hasil LLM lain pada eksperimen embedding;
- perbandingan Llama versus Qwen dilakukan sebagai eksperimen terpisah dengan pipeline retrieval yang sudah dikunci.

Dengan begitu, perbedaan hasil retrieval lebih mungkin berasal dari model embedding/strategi retrieval, bukan dari perbedaan kualitas penerjemahan query.

### 7.5 Error handling

Jika output LLM tidak sesuai schema:

1. lakukan satu kali perbaikan atau parsing ulang;
2. jika tetap gagal, gunakan fallback yang aman;
3. jangan menghentikan seluruh aplikasi karena satu output JSON tidak valid;
4. jangan mengarang `bm25_query_en` ketika terjemahan gagal; tampilkan error atau gunakan fallback yang jelas dan tercatat.

Simpan konteks multi-turn yang relevan. Preferensi baru harus memperbarui konteks sebelumnya, bukan otomatis mengganti seluruh query.

---

## 8. Hybrid Retrieval dan RRF Fusion

### 8.1 Dense retrieval

Pada tiap eksperimen, jalankan salah satu jalur dense berikut:

**Dense E5**

1. ambil `semantic_query_original`;
2. tambahkan prefix `query: `;
3. buat embedding menggunakan `intfloat/multilingual-e5-large`;
4. cari Top-K pada ChromaDB E5.

**Dense BGE**

1. ambil `semantic_query_en`;
2. tambahkan query instruction BGE;
3. buat embedding menggunakan `BAAI/bge-large-en-v1.5`;
4. cari Top-K pada ChromaDB BGE.

Contoh parameter awal:

```text
dense_top_k = 20
```

Nilai tersebut adalah parameter awal eksperimen, bukan nilai optimal yang sudah terbukti.

### 8.2 Sparse retrieval

1. ambil `bm25_query_en`;
2. lakukan normalisasi dan tokenisasi dengan fungsi yang sama seperti korpus;
3. cari Top-K pada BM25Okapi;
4. kembalikan hasil sebagai canonical `book_id`, rank, dan BM25 score.

Contoh parameter awal:

```text
bm25_top_k = 20
```

### 8.3 Reciprocal Rank Fusion

Gabungkan ranking dense dan BM25 menggunakan RRF:

\[
RRF(b) = \sum_{r \in R} \frac{1}{k + rank_r(b)}
\]

Keterangan:

- \(b\): buku;
- \(R\): daftar retriever;
- \(rank_r(b)\): posisi buku dalam hasil retriever ke-\(r\);
- \(k\): konstanta RRF.

Parameter awal:

```text
rrf_k = 60
dense_top_k = 20
bm25_top_k = 20
candidate_top_k = 30
```

Gunakan ranking/posisi yang konsisten (rank dimulai dari 1 atau 0, tetapi rumus dan implementasinya harus diselaraskan). Simpan rank asli masing-masing retriever.

### 8.4 Eksperimen retrieval yang dibandingkan

Lakukan minimal pengujian berikut:

| Eksperimen | Dense | Sparse | Fusion |
|---|---|---|---|
| R1 | E5 | Tidak | Tidak |
| R2 | BGE | Tidak | Tidak |
| R3 | Tidak | BM25 | Tidak |
| R4 | E5 | BM25 | RRF |
| R5 | BGE | BM25 | RRF |

BM25 yang digunakan pada R3, R4, dan R5 harus berasal dari indeks korpus yang sama. Query Inggris untuk BM25 juga harus sama pada perbandingan R4 dan R5.

### 8.5 Output retrieval

Setiap kandidat retrieval minimal membawa:

- canonical `book_id`;
- rank dense jika tersedia;
- dense similarity/relevance score jika tersedia;
- rank BM25 jika tersedia;
- BM25 score jika tersedia;
- RRF score jika fusion dijalankan;
- metadata buku yang diperlukan.

Retrieval hanya menghasilkan kandidat. Keputusan rekomendasi final dilakukan setelah emotion ranking dan final re-ranking.

Jangan menjumlahkan raw dense score dan raw BM25 score secara langsung karena skala keduanya berbeda. Gunakan RRF untuk menggabungkan peringkat.

---

## 9. Emotion Vector Recommender

### 9.1 Mengambil EmotionVector

Untuk setiap kandidat hasil retrieval:

1. ambil canonical `book_id`;
2. cari EmotionVector pada `emotion_companion_store.json`;
3. pastikan urutan sembilan dimensinya konsisten;
4. validasi vektor tidak kosong dan bukan vektor nol;
5. gunakan nilai yang telah diparsing dari dataset tanpa menghitung ulang.

### 9.2 User Emotion Preference belum boleh diasumsikan

Metode untuk membentuk User Emotion Preference harus ditetapkan secara eksplisit sebelum emotion ranking diaktifkan pada sistem end-to-end.

Pilihan yang masih perlu diputuskan:

1. **Age-based preference:** hanya jika mekanisme pada paper sumber dapat direplikasi dari definisi dan prosedur yang jelas;
2. **Explicit emotion preference:** pengguna memilih atau menyebutkan preferensi emosi, lalu sistem mengonversinya menggunakan pemetaan yang terdokumentasi;
3. **Hybrid approach:** menggabungkan preferensi berbasis usia dan preferensi eksplisit, hanya jika aturan penggabungannya didefinisikan dan diuji.

Jangan membiarkan LLM mengeluarkan sembilan angka bobot emosi secara bebas tanpa aturan. Jangan menganggap daftar label emosi otomatis setara dengan User Emotion Vector dalam paper sumber.

Jika metode belum ditetapkan:

- implementasikan interface/modul `user_emotion_profile`;
- nonaktifkan emotion ranking atau gunakan mode eksplisit yang sudah didefinisikan;
- jangan mengarang formula age-based;
- catat keterbatasan tersebut dalam laporan.

### 9.3 Cosine Similarity

Jika User Emotion Preference telah tersedia sebagai vektor \(P_u\), dan vektor buku adalah \(V_b\), hitung:

\[
S_{emotion}(u,b) =
\frac{P_u \cdot V_b}{\|P_u\|\|V_b\|}
\]

Tangani kondisi:

- vektor pengguna kosong;
- vektor pengguna nol;
- vektor buku tidak valid;
- vektor buku nol;
- key atau urutan dimensi tidak sesuai.

Untuk kondisi invalid, tandai skor sebagai unavailable dan gunakan fallback ranking yang terdokumentasi. Jangan membuat skor palsu.

### 9.4 Interpretasi skor

Cosine similarity mengukur keselarasan arah dua vektor, bukan memastikan dampak psikologis buku terhadap pembaca.

Gunakan istilah seperti:

- kesesuaian profil emosi;
- emotional alignment score;
- kemiripan preferensi emosi.

Jangan menyatakan sistem dapat memastikan pembaca akan merasa bahagia, sedih, atau mengalami emosi tertentu setelah membaca buku.

---

## 10. Final re-ranking

Setelah kandidat retrieval tersedia dan EmotionVector berhasil diambil, hitung skor akhir.

Contoh rancangan:

\[
S_{final}(b) =
\alpha \cdot S_{retrieval\_norm}(b)
+
\beta \cdot S_{emotion\_norm}(b)
\]

Aturan:

- `S_retrieval_norm` harus berasal dari skor RRF yang ditransformasikan ke skala yang sebanding;
- `S_emotion_norm` harus ditransformasikan secara konsisten jika diperlukan;
- jangan menggabungkan raw RRF dan cosine similarity tanpa normalisasi/penjelasan skala;
- simpan RRF score, normalized retrieval score, emotion score, dan final score secara terpisah;
- `alpha` dan `beta` adalah parameter eksperimen, bukan nilai optimal;
- nilai awal 0,5 dan 0,5 boleh dipakai hanya sebagai baseline uji;
- jika EmotionVector atau User Emotion Preference tidak valid, gunakan fallback retrieval-only dan tandai bahwa emotion ranking tidak aktif;
- jangan menggunakan `Average_Rating` sebagai komponen skor utama kecuali ditetapkan sebagai eksperimen terpisah.

Contoh struktur output:

```json
{
  "book_id": "12345",
  "title": "Example Book",
  "rrf_score": 0.03,
  "retrieval_score_norm": 0.85,
  "emotion_score": 0.91,
  "final_score": 0.88,
  "rank": 1
}
```

Angka tersebut hanya ilustrasi format.

Pilih Top-N setelah final ranking. Nilai awal yang dapat digunakan adalah Top-5, tetapi harus menjadi parameter konfigurasi.

---

## 11. LLM integration dan response generation

### 11.1 LLM yang digunakan

Sediakan pilihan:

- Llama 3.1 8B Instruct;
- Qwen 2.5 7B Instruct.

Keduanya digunakan dalam mode inference dan **tanpa fine-tuning**.

Model embedding adalah komponen terpisah. Jangan menggunakan nama model Llama/Qwen sebagai embedding model.

Provider/runtime harus disesuaikan dengan layanan atau perangkat yang benar-benar tersedia. Nama model dan endpoint harus dikonfigurasi, bukan diasumsikan.

Simpan API key pada environment variable atau secret manager. Jangan menaruh API key di source code atau commit ke repository.

### 11.2 Tugas LLM

LLM dapat digunakan untuk:

1. memahami pesan pengguna;
2. mempertahankan konteks percakapan;
3. menyusun query terstruktur;
4. membantu penerjemahan query untuk BM25 dan BGE;
5. menjelaskan hasil rekomendasi final.

LLM bukan komponen yang menentukan ranking buku secara intuitif. Ranking final ditentukan oleh modul retrieval dan recommender.

### 11.3 Grounded response generation

Kirimkan hanya Top-N hasil final beserta metadata yang tersedia ke response generator.

LLM harus:

- menggunakan judul dan penulis sesuai record;
- menjelaskan buku berdasarkan Description;
- menjelaskan alasan rekomendasi berdasarkan hasil retrieval dan skor emosi secara hati-hati;
- tidak menambahkan buku di luar Top-N;
- tidak mengubah ranking;
- tidak mengarang rating, usia, genre, themes, content warning, atau detail cerita;
- mengakui informasi yang tidak tersedia.

Instruksi grounding yang dapat digunakan:

```text
You are a book recommendation assistant.
Use only the book records supplied in the context.
Do not invent books, authors, ratings, age information, themes,
content warnings, publication years, or plot details.
Do not change the ranking produced by the recommendation engine.
Explain each recommendation using the supplied description and metadata.
If a field is missing, state that the information is unavailable.
Describe emotion scores as preference alignment, not as a guarantee
of how the reader will feel.
Respond in Indonesian unless the user uses another language.
```

Lakukan pemeriksaan pasca-generasi. Jika judul/penulis yang disebutkan tidak cocok dengan record Top-N, gunakan template jawaban deterministik dari hasil ranking.

---

## 12. Evaluasi dan rancangan eksperimen

Evaluasi harus memisahkan kualitas embedding, hybrid retrieval, emotion ranking, dan LLM response generation.

### 12.1 Dataset query evaluasi

Siapkan kumpulan query uji yang mewakili:

- query berbahasa Indonesia;
- query berbahasa Inggris;
- query dengan topik spesifik;
- query yang lebih abstrak atau berbasis suasana/emosi;
- query dengan batasan genre/kategori jika metadata mendukung;
- query multi-turn.

Untuk menghitung metrik retrieval, perlu ada relevansi acuan (ground truth) yang dinilai secara konsisten. Jika belum tersedia, buat anotasi relevansi query-buku terlebih dahulu. Jangan mengklaim Precision/Recall/nDCG tanpa label relevansi yang memadai.

### 12.2 Metrik retrieval

Gunakan metrik yang sesuai, misalnya:

- Precision@K;
- Recall@K;
- Mean Reciprocal Rank (MRR@K);
- Normalized Discounted Cumulative Gain (nDCG@K).

Bandingkan R1 sampai R5 pada bagian 8.4 dengan query dan label relevansi yang sama.

### 12.3 Eksperimen embedding dan bahasa

Eksperimen utama:

1. E5 multilingual: query dense bahasa asli + BM25 query Inggris;
2. BGE English: query dense Inggris hasil terjemahan + BM25 query Inggris;
3. Dense-only versus Hybrid (dense + BM25 + RRF).

Jangan mengubah dataset, BM25 index, query set, dan ground truth di antara eksperimen. Simpan hasil query terjemahan agar perbandingan dapat direproduksi.

### 12.4 Evaluasi Emotion Ranking

Periksa:

- ketepatan lookup EmotionVector berdasarkan ID;
- kesesuaian urutan sembilan dimensi;
- hasil cosine similarity dengan unit test;
- perubahan ranking ketika preferensi emosi berubah;
- fallback saat vektor kosong/tidak valid;
- perbedaan hasil sebelum dan sesudah emotion re-ranking.

Lakukan pengujian retrieval-only versus retrieval + emotion ranking. Jangan mengklaim emotion ranking meningkatkan kualitas sebelum ada evaluasi.

### 12.5 Evaluasi LLM

Untuk membandingkan Llama dan Qwen, gunakan:

- query yang sama;
- structured query yang sama jika yang dievaluasi adalah response generation;
- kandidat Top-N dan urutan ranking yang sama;
- prompt yang sama;
- parameter inference yang dicatat.

Aspek evaluasi:

- ketepatan memahami query;
- relevansi penjelasan;
- kesesuaian dengan metadata dan Description;
- kepatuhan terhadap ranking;
- hallucination/factual consistency;
- kualitas bahasa dan keterbacaan.

Pisahkan pengujian query understanding dari response generation agar sumber perbedaan dapat diidentifikasi.

---

## 13. Chatbot UI

Gunakan Streamlit atau framework yang sudah ada di repository. Jangan mengganti framework tanpa alasan dan tanpa memeriksa struktur proyek.

Elemen UI minimal:

1. judul aplikasi dan deskripsi singkat;
2. pemilih LLM (Llama 3.1 8B atau Qwen 2.5 7B);
3. area percakapan multi-turn;
4. input pesan pengguna;
5. daftar rekomendasi Top-N;
6. card buku yang menampilkan Title, Author, Description ringkas, Average_Rating jika tersedia, Genre/Themes jika tersedia, dan skor;
7. label Age hanya jika definisi sumber sudah jelas;
8. expandable section untuk retrieval score, emotion score, final score, dan metadata;
9. tombol percakapan baru/reset;
10. pesan error yang mudah dipahami.

Jangan menampilkan `None`, `NaN`, atau string kosong pada UI. Gunakan label `Informasi tidak tersedia` jika field perlu ditampilkan.

Jangan menyebut kolom Genre sebagai genre baku jika nilai sumbernya merupakan metadata kategori/subjek campuran.

---

## 14. Struktur folder yang disarankan

Struktur aktual repository harus diperiksa terlebih dahulu. Contoh struktur modular:

```text
PROJECT RS/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas/
│   │   ├── query_schema.py
│   │   └── book_schema.py
│   └── services/
│       ├── llm_service.py
│       ├── query_parser.py
│       ├── query_translation.py
│       ├── document_builder.py
│       ├── embedding_service.py
│       ├── indexing_service.py
│       ├── bm25_service.py
│       ├── retrieval_service.py
│       ├── rrf_fusion.py
│       ├── emotion_store.py
│       ├── emotion_ranker.py
│       ├── response_generator.py
│       └── conversation_service.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── validate_dataset.py
│   ├── build_knowledge_base.py
│   ├── benchmark_embedding.py
│   └── test_rag_pipeline.py
├── data/
│   ├── source/
│   └── processed/
├── rag/
│   └── vector_store/
│       ├── chroma_e5_large/
│       ├── chroma_bge_large/
│       ├── bm25_index.pkl
│       ├── bm25_book_ids.json
│       └── emotion_companion_store.json
├── reports/
│   ├── dataset_validation_report.md
│   ├── indexing_report.md
│   └── retrieval_evaluation_report.md
├── tests/
│   ├── test_dataset.py
│   ├── test_document_builder.py
│   ├── test_emotion_store.py
│   ├── test_retrieval.py
│   └── test_ranking.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Struktur ini hanya usulan. Sesuaikan dengan repository yang sudah ada dan hindari refactor besar yang tidak diperlukan.

---

## 15. Rencana implementasi bertahap

### Tahap 0 — Inspect before coding

1. periksa struktur repository;
2. temukan `Fixed_Genre_Themes_GoodReads_Emotion.csv`;
3. baca nama kolom dan beberapa baris;
4. verifikasi jumlah baris, missing values, dan ID;
5. uji parsing `EmotionVector`;
6. periksa folder output dan dependency yang sudah tersedia;
7. laporkan rencana implementasi.

**Output:** laporan inspeksi dan asumsi yang masih perlu dikonfirmasi.

### Tahap 1 — Dataset loader dan validator

Implementasikan:

- loader CSV;
- konfigurasi nama kolom;
- canonical book ID;
- preprocessing whitespace dan missing value;
- kelayakan indexing;
- validasi EmotionVector;
- laporan dataset.

**Syarat selesai:** file sumber tidak berubah, 19.784 EmotionVector tervalidasi (atau semua penyimpangan tercatat), dan 19.778 buku eligible sesuai kondisi dataset aktual.

### Tahap 2 — Document Builder

Implementasikan `document_builder.py` dengan baseline Title + Author + Description.

Output minimum:

- `document_text`;
- `e5_passage_text`;
- `bge_passage_text`;
- `book_id`;
- metadata buku.

**Syarat selesai:** enam buku yang tidak mempunyai Title maupun Description tidak masuk dokumen retrieval, tidak ada placeholder dalam teks embedding, dan kolom Genre/Themes tetap metadata saja.

### Tahap 3 — Emotion Companion Store

Implementasikan parser berbasis `ast.literal_eval()`, validasi sembilan dimensi, konversi angka ke float setelah validasi, dan penyimpanan JSON keyed by ID.

**Syarat selesai:** seluruh ID dataset yang valid terwakili di Emotion Store, termasuk enam buku yang tidak masuk retrieval index.

### Tahap 4 — Embedding benchmark dan dense indexing

1. benchmark 500–1.000 dokumen untuk E5 Large dan BGE Large;
2. catat latency, RAM, batch size, dan jumlah dokumen yang terpotong;
3. tetapkan konfigurasi eksperimen;
4. bangun ChromaDB E5 dan BGE secara terpisah.

**Syarat selesai:** dua collection/indeks persistent tersedia, jumlah record sesuai, dimensi benar, serta query dan dokumen menggunakan konfigurasi model yang sama.

### Tahap 5 — BM25 indexing

Implementasikan tokenizer dan BM25Okapi dari `document_text` tanpa prefix model.

Simpan objek BM25 dan mapping `book_id` yang persis mengikuti urutan dokumen index.

**Syarat selesai:** BM25 memuat seluruh buku eligible dan hasil retrieval dapat dipetakan ke ID yang benar.

### Tahap 6 — Query understanding dan translation

Implementasikan structured query yang memuat:

- raw query;
- semantic query dalam bahasa asli;
- semantic query Inggris;
- query Inggris untuk BM25;
- topik, preferensi, dan constraint yang benar-benar diekspresikan pengguna.

**Syarat selesai:** satu query dapat menghasilkan versi yang benar untuk E5, BGE, dan BM25; terjemahan konsisten; error JSON memiliki fallback.

### Tahap 7 — Dense, BM25, dan Hybrid Retrieval

Implementasikan R1–R5, RRF, konfigurasi Top-K, serta pengembalian canonical book ID.

**Syarat selesai:** semua retriever dapat dijalankan terpisah dan bersama, hasil RRF dapat diaudit, dan query kosong/hasil kosong ditangani.

### Tahap 8 — Emotion ranking dan final ranking

Implementasikan User Emotion Preference hanya setelah metode pembentukannya diputuskan. Sambungkan Emotion Store, cosine similarity, normalisasi skor, dan fallback retrieval-only.

**Syarat selesai:** unit test cosine similarity lulus, urutan dimensi konsisten, dan final score dapat dilacak dari komponen skornya.

### Tahap 9 — LLM integration dan UI

Integrasikan Llama/Qwen tanpa fine-tuning, query understanding, grounded response generation, dan UI.

**Syarat selesai:** model dapat dipilih, rekomendasi berasal dari Top-N final, LLM tidak mengubah ranking, dan field kosong ditangani dengan benar.

### Tahap 10 — Evaluasi dan dokumentasi

Jalankan retrieval evaluation, emotion-ranking evaluation, dan LLM evaluation. Dokumentasikan konfigurasi, dataset, hasil, serta keterbatasan.

**Syarat selesai:** hasil pengujian dapat direproduksi dan tidak ada klaim performa tanpa hasil evaluasi.

---

## 16. Kriteria penerimaan akhir

- [ ] Dataset final `Fixed_Genre_Themes_GoodReads_Emotion.csv` digunakan.
- [ ] Dataset sumber tidak diubah atau ditimpa.
- [ ] Enrichment Genre/Themes tidak dijalankan ulang.
- [ ] `Age` dipertahankan dan tidak diinterpretasikan tanpa sumber.
- [ ] Document text baseline hanya memakai Title, Author, Description.
- [ ] Genre dan Themes hanya disimpan sebagai metadata untuk baseline.
- [ ] Enam buku tanpa Title dan Description tidak masuk retrieval index.
- [ ] Enam buku tersebut tetap ada pada dataset sumber dan Emotion Store.
- [ ] Dense E5 dan BGE mempunyai ChromaDB terpisah.
- [ ] E5 menggunakan prefix `passage:` untuk dokumen dan `query:` untuk query.
- [ ] BGE menggunakan English query dan query instruction yang dikonfigurasi.
- [ ] BM25 memakai `document_text` tanpa prefix model.
- [ ] Query BM25 menggunakan bahasa Inggris yang konsisten dengan korpus.
- [ ] Query Inggris yang digunakan untuk BM25 sama pada eksperimen E5 dan BGE.
- [ ] BM25 menyimpan mapping posisi dokumen ke canonical `book_id`.
- [ ] EmotionVector diparsing dengan `ast.literal_eval()`.
- [ ] Urutan sembilan dimensi tervalidasi.
- [ ] Emotion Store mencakup seluruh dataset yang valid.
- [ ] RRF menggabungkan ranking, bukan menjumlahkan raw score berbeda skala.
- [ ] Metode User Emotion Preference terdokumentasi dan tidak dikarang.
- [ ] Cosine similarity memiliki unit test dan penanganan vektor nol.
- [ ] Final ranking menyimpan skor setiap komponen.
- [ ] Llama dan Qwen berjalan tanpa fine-tuning.
- [ ] LLM hanya menjelaskan buku Top-N yang diberikan.
- [ ] UI tidak menampilkan None/NaN.
- [ ] Evaluasi dilakukan pada query dan relevansi acuan yang konsisten.
- [ ] README menjelaskan setup, indexing, menjalankan sistem, dan evaluasi.

---

## 17. Aturan kerja untuk Antigravity

1. **Inspect first, then implement.** Jangan menebak lokasi dataset, struktur repository, header CSV, atau konfigurasi runtime.
2. Jangan menghapus atau menimpa dataset sumber.
3. Jangan menghitung ulang EmotionVector atau mengulang NRC-EIL.
4. Jangan melakukan fine-tuning Llama/Qwen.
5. Jangan melakukan enrichment Genre/Themes ulang.
6. Jangan memasukkan Genre/Themes ke teks baseline embedding/BM25.
7. Jangan memasukkan placeholder buatan ke teks embedding.
8. Jangan memasukkan enam buku yang tidak mempunyai Title dan Description ke dense/BM25 index.
9. Jangan menghapus enam buku tersebut dari dataset atau Emotion Store.
10. Jangan menganggap Age sebagai minimum reading age sebelum ada bukti dari sumber.
11. Jangan mengarang User Emotion Preference atau angka bobot emosi.
12. Pisahkan query asli, query Inggris untuk BGE, dan query Inggris untuk BM25.
13. Gunakan query Inggris BM25 yang sama ketika membandingkan E5 dan BGE.
14. Jangan mencampur embedding dua model dalam satu Chroma collection.
15. Jangan menjumlahkan raw cosine/dense score dan raw BM25 score tanpa skema fusion yang tepat.
16. Buat perubahan kecil dan jalankan test setelah setiap tahap.
17. Catat semua jumlah record, ID yang dikecualikan, error, parameter, dan waktu proses.
18. Jangan menyimpulkan model atau metode lebih baik sebelum evaluasi.
19. Jika ada keputusan yang belum ditetapkan, buat interface/configuration dan laporkan kebutuhan keputusan; jangan mengarang asumsi.
20. Utamakan prototype end-to-end yang benar, dapat dijalankan, dan dapat diuji daripada fitur tambahan yang tidak selesai.

---

## 18. Prompt pembuka untuk Antigravity

```text
Baca seluruh dokumen spesifikasi versi 2.0 ini sebelum mengubah repository.

Tugasmu adalah membangun prototype chatbot rekomendasi buku berbasis Hybrid RAG dan Emotion Vector.

KONTEKS DATASET:
Gunakan file Fixed_Genre_Themes_GoodReads_Emotion.csv yang sudah memiliki Genre, genre_source, Themes, themes_source, dan EmotionVector. Dataset ini adalah hasil enrichment final. Jangan melakukan enrichment ulang dan jangan mengubah file sumber.

KONDISI DATA YANG SUDAH DIPERIKSA:
- 19.784 baris dan 12 kolom.
- 19.778 buku eligible untuk retrieval karena memiliki Title atau Description.
- 6 buku tanpa Title dan Description dikecualikan dari ChromaDB dan BM25.
- Emotion Companion Store tetap berisi semua 19.784 record valid.
- EmotionVector memiliki sembilan dimensi dan harus diparsing menggunakan ast.literal_eval().
- Age tidak diubah dari dataset awal dan tidak boleh ditafsirkan sebagai minimum reading age tanpa dokumentasi.

RAG:
Bangun dua jalur embedding yang terpisah:
A. intfloat/multilingual-e5-large:
   - dokumen: "passage: " + document_text
   - query dense: "query: " + semantic_query_original
   - query dapat tetap dalam bahasa pengguna.
B. BAAI/bge-large-en-v1.5:
   - dokumen: document_text tanpa prefix
   - query dense: English translation dengan instruction
     "Represent this sentence for searching relevant passages: "
BM25:
   - gunakan rank_bm25/BM25Okapi;
   - korpus: document_text tanpa prefix model;
   - query: bm25_query_en;
   - gunakan query Inggris BM25 yang sama untuk eksperimen E5 dan BGE;
   - simpan mapping BM25 document position ke canonical book_id.
ChromaDB:
   - simpan indeks E5 dan BGE secara terpisah.
Fusion:
   - gunakan RRF, jangan menambahkan raw score yang skala nilainya berbeda.

EMOTION:
- Jangan hitung ulang EmotionVector.
- Validasi sembilan key dan semua nilai numerik finite.
- Gunakan canonical book_id untuk lookup O(1).
- Jangan mengarang formula User Emotion Preference. Jika metode belum diputuskan, sediakan interface dan laporkan.

LLM:
- Llama 3.1 8B Instruct dan Qwen 2.5 7B Instruct hanya digunakan dalam mode inference.
- Tidak ada fine-tuning.
- LLM tidak boleh mengubah ranking atau menambahkan buku yang tidak ada dalam Top-N.

CARA KERJA:
1. Inspeksi repository dan dataset aktual terlebih dahulu.
2. Laporkan temuan sebelum perubahan besar.
3. Implementasikan secara bertahap sesuai dokumen.
4. Setelah tiap tahap, jalankan test yang sesuai.
5. Laporkan file yang dibuat/diubah, jumlah record, hasil test, konfigurasi model, error, dan kendala.
6. Jangan menghapus atau menimpa sumber data.
7. Jangan mengklaim performa meningkat sebelum evaluasi retrieval dilakukan.

Mulai dari Tahap 0: inspect repository dan validasi dataset. Jangan langsung membuat seluruh aplikasi sekaligus.
```

---

## 19. Referensi teknis model embedding

Gunakan dokumentasi model card sebagai acuan implementasi prefix, instruction, dimensi, dan batas token. Periksa kembali versi model/dependency saat implementasi karena detail runtime dapat berubah.

1. **Multilingual E5 Large**  
   Model card: https://huggingface.co/intfloat/multilingual-e5-large  
   Paper/technical report: https://arxiv.org/abs/2402.05672  
   Catatan: prefix `query:` dan `passage:` digunakan untuk asymmetric retrieval; input panjang dibatasi hingga 512 token.

2. **BGE Large English v1.5**  
   Model card: https://huggingface.co/BAAI/bge-large-en-v1.5  
   Catatan: dimensi embedding 1.024 dan sequence length 512. Untuk short-query-to-passage retrieval, model card merekomendasikan query instruction `Represent this sentence for searching relevant passages: `. Instruction tidak ditambahkan ke passage.

3. **BGE model usage and guidance**  
   https://github.com/FlagOpen/FlagEmbedding  
   Catatan: gunakan dokumentasi resmi untuk implementasi dan konfigurasi model BGE.

---

## 20. Keputusan yang masih perlu ditetapkan

Beberapa keputusan belum boleh diisi oleh coding agent secara sepihak:

1. **User Emotion Preference:** apakah menggunakan pendekatan age-based dari paper, explicit user emotion, atau hybrid.
2. **Definisi kolom Age:** perlu bukti dokumentasi sebelum digunakan sebagai filter.
3. **Bobot final ranking:** `alpha` dan `beta` harus dievaluasi, bukan dianggap optimal.
4. **Provider/runtime LLM:** tentukan berdasarkan akses aktual.
5. **Pilihan model embedding final:** jalankan benchmark E5 dan BGE sebelum menentukan model terbaik.
6. **Ground-truth retrieval:** siapkan query dan label relevansi untuk menghitung metrik evaluasi.

Keputusan yang sudah ditetapkan dan tidak perlu dipertanyakan ulang oleh Antigravity adalah penggunaan dataset enriched final, tidak mengulang enrichment, tidak menghitung ulang EmotionVector, mengecualikan enam buku tanpa Title dan Description dari retrieval index, mempertahankan seluruh EmotionVector di companion store, serta memisahkan indeks E5 dan BGE.

**Prioritas kerja:** validasi dataset → Document Builder → Emotion Store → benchmark dan dense indexing E5/BGE → BM25 indexing → query routing bilingual → RRF → emotion ranking → Llama/Qwen response generation → UI → evaluasi.
