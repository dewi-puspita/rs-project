# Spesifikasi Tugas Implementasi untuk Antigravity

## Chatbot Rekomendasi Buku Berbasis Hybrid RAG dan Emotion Vector

**Versi:** 1.0
**Tujuan dokumen:** Menjadi brief teknis dan instruksi implementasi
untuk Antigravity sebagai coding agent.

---

## 1. Ringkasan proyek

Bangun sebuah prototype **chatbot rekomendasi buku** yang mengembangkan
pendekatan *Emotion Vector-Based Fine-Tuning of Large Language Models
for Age-Aware Teenage Book Recommendations* dengan menambahkan kemampuan
percakapan dan Retrieval-Augmented Generation (RAG).

Fokus pengembangan kelompok adalah:

1. membangun knowledge base dari dataset buku yang sudah dimiliki;
2. melakukan enrichment metadata buku secara terbatas jika diperlukan
   dan memungkinkan;
3. mengimplementasikan Hybrid RAG untuk mencari kandidat buku;
4. menggunakan **Emotion Vector yang sudah tersedia di dataset** untuk
   menghitung kesesuaian emosional dan melakukan ranking;
5. mengintegrasikan Llama 3.1 8B dan Qwen 2.5 7B dalam mode inference
   tanpa fine-tuning;
6. membuat chatbot yang menjelaskan rekomendasi berdasarkan data buku
   yang benar-benar ditemukan.

### Prinsip paling penting

**Dataset yang digunakan sudah memiliki Emotion Vector. Jangan membangun
ulang proses Emotion Vector dari awal.**

Jangan: - menghitung ulang NRC-EIL; - mengulang synonym
matching/WordNet; - menghitung ulang agregasi dan normalisasi Emotion
Vector; - melakukan fine-tuning terhadap Llama atau Qwen; - mengganti
dataset sumber dengan dataset buku lain tanpa persetujuan.

Sistem hanya perlu memvalidasi, menyimpan, mengambil, dan menggunakan
Emotion Vector yang sudah ada.

---

## 2. Konteks dataset

Dataset utama berasal dari penelitian sumber dan saat ini memiliki kolom
berikut:

---

  Kolom                   Keterangan              Penggunaan dalam sistem

---

  `ISBN`                  ISBN buku               Identitas eksternal
                                                  buku jika tersedia

  `ID`                    ID buku pada dataset    Kunci utama untuk
                                                  menghubungkan record

  `Title`                 Judul buku              Metadata dan tampilan
                                                  hasil

  `Author`                Penulis                 Metadata dan tampilan
                                                  hasil

  `Description`           Deskripsi/sinopsis buku Teks utama untuk
                                                  embedding dan semantic
                                                  retrieval

  `Average_Rating`        Rata-rata rating        Informasi tambahan,
                                                  bukan dasar utama
                                                  kesesuaian emosi

  `Age`                   Atribut usia dari       Digunakan hanya setelah
                          dataset                 definisinya
                                                  diverifikasi

`Emotion_Vector`        Vektor emosi yang telah Input utama untuk
                          tersedia                emotion-based ranking
-----------------------------------------------------------------------

### 2.1 Hal yang harus diperiksa sebelum coding

Jangan mengasumsikan format file atau struktur kolom. Pada tahap
pertama:

1. cari file dataset di workspace;
2. identifikasi format file (`csv`, `xlsx`, atau format lain);
3. baca nama kolom aktual;
4. periksa beberapa baris sampel;
5. periksa tipe data dan nilai kosong;
6. periksa format `Emotion_Vector`;
7. laporkan temuan kepada pengguna sebelum melakukan transformasi yang
   berisiko.

Nama kolom aktual dapat berbeda dalam kapitalisasi atau penggunaan
spasi. Buat konfigurasi pemetaan kolom agar nama kolom dapat disesuaikan
tanpa mengubah data sumber.

### 2.2 Validasi Emotion Vector

Emotion Vector mungkin tersimpan sebagai: - satu kolom berisi
list/string seperti `[0.1, 0.2, ...]`; - satu kolom berisi JSON
string; - beberapa kolom terpisah untuk setiap dimensi emosi.

Implementasikan parser sesuai format aktual. Jangan menebak urutan
dimensi. Pastikan urutan dan jumlah dimensi mengikuti dokumentasi
dataset/paper sumber. Jika dataset mengikuti representasi sembilan
dimensi, validasi bahwa setiap record valid memiliki sembilan nilai
numerik.

Pemeriksaan minimal: - jumlah dimensi konsisten; - semua nilai dapat
dikonversi menjadi angka; - tidak ada `NaN` atau `Infinity`; - vektor
tidak kosong; - ID buku tetap dapat dipasangkan dengan vektornya.

Jangan mengubah nilai Emotion Vector yang valid. Jika ada record yang
gagal diparse, catat di laporan validasi dan jangan diam-diam mengganti
nilainya.

### 2.3 Perlakuan terhadap kolom `Age`

Jangan langsung mengartikan `Age` sebagai batas usia minimum atau
rekomendasi usia pembaca. Definisi kolom harus diperiksa pada
dokumentasi dataset/paper. Sampai definisinya jelas, tampilkan atau
gunakan sebagai metadata sesuai arti sumbernya, bukan sebagai filter
keamanan usia.

---

## 3. Strategi metadata enrichment

Tujuan enrichment adalah melengkapi metadata yang berguna untuk
retrieval dan penyajian hasil, **bukan membangun ulang dataset**.

### 3.1 Prioritas atribut

---

  Atribut                 Prioritas               Strategi

---

  `Genre`                 Disarankan jika mudah   Ambil dari metadata
                          diperoleh               katalog/API yang dapat
                                                  diverifikasi

  `Themes`                Opsional                Tidak wajib; jangan
                                                  menganggap sinopsis
                                                  mewakili seluruh tema
                                                  buku

  `Publication_Year`      Opsional                Ambil dari sumber
                                                  metadata jika tersedia

`Content_Warning`       Tidak wajib untuk versi Jangan menyimpulkan
                          awal                    "tidak ada" hanya
                                                  karena tidak ditemukan
------------------------------------------------------------------------

Dengan waktu pengerjaan yang terbatas, jangan menghabiskan sebagian
besar waktu untuk enrichment. Sistem harus tetap berfungsi dengan
delapan kolom dataset asli.

### 3.2 Sumber enrichment

Jika enrichment dilakukan, gunakan sumber metadata buku yang dapat
diperiksa, misalnya: - Open Library API:
https://openlibrary.org/developers/api - Google Books API:
https://developers.google.com/books/docs/v1/using

Prioritaskan pencocokan ISBN. Jika ISBN tidak tersedia, pencocokan judul
dan penulis harus diperlakukan lebih hati-hati karena berisiko salah
edisi atau salah buku.

### 3.3 Aturan khusus untuk Themes

Description tidak selalu mengungkap seluruh tema buku. Karena itu: -
Themes bukan kolom wajib; - jangan mengisi Themes secara otomatis
seolah-olah merupakan metadata resmi; - jika memakai LLM untuk
ekstraksi, beri label `inferred` atau `llm_extracted`; - ekstrak hanya
tema yang didukung oleh teks deskripsi; - jika bukti tidak cukup,
hasilkan nilai kosong; - jangan menggunakan Themes sebagai satu-satunya
dasar retrieval.

Semantic retrieval tetap dapat berjalan menggunakan `Title`, `Author`,
dan `Description`.

### 3.4 Aturan khusus untuk Content Warning

Content Warning bukan kebutuhan wajib untuk prototype awal. Jika tidak
ada sumber yang dapat diverifikasi, biarkan kosong/tidak diketahui.
Nilai kosong tidak berarti buku bebas dari konten sensitif. Jangan
membuat klaim bahwa buku aman untuk usia tertentu hanya berdasarkan
sinopsis atau Emotion Vector.

### 3.5 Bentuk output enrichment

Jangan menimpa dataset asli. Buat file hasil baru, misalnya:

- `data/processed/books_clean.csv`
- `data/processed/books_enriched.csv`
- `reports/enrichment_report.csv`

Sertakan kolom sumber/status jika memungkinkan, misalnya: -
`genre_source` - `publication_year_source` - `themes_source` -
`content_warning_status`

Jika atribut tidak berhasil diperoleh, biarkan kosong dan catat jumlah
kegagalannya. Jangan mengisi nilai palsu atau placeholder yang terlihat
seperti fakta.

---

## 4. Arsitektur sistem

Sistem memiliki dua pipeline: **offline knowledge-base construction**
dan **online recommendation**.

### 4.1 Diagram 1 --- Proposed System Architecture

Diagram ini menjelaskan alur sistem dari sudut pandang pengguna.

```text
USER INTERFACE / CHATBOT
        |
        v
LLM PROCESSING
  - Llama 3.1 8B Instruct (non-fine-tuned)
  - Qwen 2.5 7B Instruct (non-fine-tuned)
        |
        v
QUERY PROCESSING
  - Extract age
  - Extract reading needs/topics
  - Extract genre preference (if available)
  - Extract emotion preference
  - Extract constraints
        |
        v
HYBRID RAG RETRIEVAL
  - Dense retrieval: embedding + vector search
  - Sparse retrieval: BM25
  - Fusion: RRF
        |
        v
RETRIEVED BOOK CANDIDATES
  - Book metadata
  - Existing Emotion Vector
        |
        v
EMOTION VECTOR RECOMMENDER
  - User Emotion Preference
  - Existing Book Emotion Vector
  - Cosine Similarity
        |
        v
RE-RANKING AND FILTERING
  - Semantic relevance
  - Emotional alignment
  - Available and verified metadata filters
        |
        v
LLM RESPONSE GENERATION
        |
        v
FINAL TOP-N BOOK RECOMMENDATIONS
```

### 4.2 Diagram 2 --- RAG and Recommendation Pipeline

Diagram ini menjelaskan proses teknis pembangunan knowledge base dan
proses rekomendasi.

```text
OFFLINE: KNOWLEDGE BASE CONSTRUCTION

Original Dataset (already contains Emotion Vector)
        |
        v
Dataset Inspection, Validation & Preprocessing
        |
        +------------------------------+
        |                              |
        v                              v
Book Text Preparation          Existing Emotion Vector
        |                              |
        v                              |
Embedding Model                       |
        |                              |
        v                              |
Book Embeddings                       |
        |                              |
        +---------------+--------------+
                        |
                        v
                 KNOWLEDGE BASE
       Embeddings + Book Metadata + Emotion Vector


ONLINE: RECOMMENDATION

User Message
        |
        v
LLM Query Understanding
        |
        v
Structured Query
        |
        +----------------------+----------------------+
        |                                             |
        v                                             v
Query Embedding                               Keyword Query
        |                                             |
        v                                             v
Dense / Vector Search                            BM25 Search
        |                                             |
        +----------------------+----------------------+
                               |
                               v
                         RRF Fusion
                               |
                               v
                    Retrieved Book Candidates
                               |
                 +-------------+--------------+
                 |                            |
                 v                            v
        Candidate Emotion Vectors      Candidate Metadata
                 |                            |
                 v                            |
        Cosine Similarity                       |
                 |                            |
                 +-------------+--------------+
                               |
                               v
                       Final Re-ranking
                               |
                               v
                         Top-N Books
                               |
                               v
                   LLM Explanation Generation
                               |
                               v
                       Chatbot Response
```

**Jangan memasukkan NRC-EIL sebagai tahap preprocessing baru.** Emotion
Vector telah disediakan dalam dataset dan digunakan kembali.

---

## 5. Offline pipeline: membangun knowledge base

### 5.1 Data preprocessing

Buat proses preprocessing yang: - membaca file dataset; - memetakan nama
kolom; - membersihkan whitespace pada teks; - memeriksa ID dan ISBN; -
memeriksa duplikasi; - memvalidasi Description; - memvalidasi
Average_Rating; - memvalidasi Emotion Vector; - mempertahankan kolom
asli; - menghasilkan laporan validasi.

Jangan menghapus edisi buku yang berbeda secara otomatis hanya karena
judulnya sama. Gunakan ID/ISBN dan aturan deduplikasi yang transparan.

### 5.2 Menyusun teks dokumen buku

Buat satu field `document_text` untuk embedding. Format yang disarankan:

```text
Title: {title}
Author: {author}
Description: {description}
Genre: {genre_if_available}
```

Aturan: - Description menjadi sumber teks utama; - Genre hanya
ditambahkan jika tersedia dan sumbernya jelas; - jangan menambahkan
Themes hasil tebakan; - jangan menambahkan Content Warning yang tidak
terverifikasi; - jangan memasukkan Emotion Vector sebagai teks embedding
secara default.

### 5.3 Embedding model

Gunakan satu embedding model terpisah dari Llama/Qwen.

Persyaratan: - model yang sama dipakai untuk embedding dokumen dan
query; - model yang sama dipakai ketika membandingkan Llama dengan
Qwen; - catat nama model, versi, dimensi, dan konfigurasi; - jangan
mengganti model setelah index dibuat kecuali seluruh embedding dibangun
ulang.

Pemilihan embedding model harus mempertimbangkan bahasa dataset dan
bahasa query. Jika mayoritas deskripsi dan query berbahasa Inggris,
pilih model yang sesuai untuk teks Inggris. Jika query dapat berbahasa
Indonesia dan dokumen berbahasa Inggris, pertimbangkan model
multilingual.

Jangan menganggap Llama 3.1 8B atau Qwen 2.5 7B sebagai embedding model
hanya karena keduanya merupakan LLM yang dipakai untuk chatbot.

### 5.4 Vector database / knowledge base

Untuk prototype, gunakan ChromaDB apabila tidak ada infrastruktur
database yang sudah ditentukan.

Satu record buku minimal harus dapat dihubungkan melalui `book_id` dan
menyimpan/mengakses: - embedding; - `Title`; - `Author`; -
`Description`; - `ISBN`; - `Age`; - `Average_Rating`; - metadata
enrichment yang tersedia; - `Emotion_Vector`.

Jika vector database tidak mendukung penyimpanan list Emotion Vector
sebagai metadata, simpan Emotion Vector pada tabel/JSON/Parquet
pendamping dengan `book_id` sebagai foreign key. Jangan kehilangan
keterkaitan antara embedding dan Emotion Vector.

### 5.5 Indexing

Sediakan script indexing yang: - dapat dijalankan ulang; - tidak membuat
record duplikat; - memiliki opsi rebuild/reset yang eksplisit; -
menampilkan jumlah record yang berhasil di-index; - mencatat record yang
gagal; - menyimpan konfigurasi model embedding dan database.

---

## 6. Online pipeline: query understanding

### 6.1 LLM yang digunakan

Sediakan dua konfigurasi model:

- **Llama 3.1 8B Instruct**, tanpa fine-tuning;
- **Qwen 2.5 7B Instruct**, tanpa fine-tuning.

Model yang dipakai harus dapat dipilih melalui konfigurasi aplikasi.
Jangan mengasumsikan model tersedia secara lokal. Periksa apakah runtime
lokal/perangkat keras memadai atau gunakan provider/API yang memang
tersedia.

Gunakan environment variable untuk API key. Jangan menulis API key
langsung di source code atau commit ke repository.

### 6.2 Peran LLM pertama: query understanding

LLM mengubah pesan bebas pengguna menjadi struktur data yang dapat
dipakai retrieval dan recommender.

Contoh pesan pengguna:

> Aku berusia 17 tahun. Aku ingin membaca cerita tentang persahabatan
> dan menemukan harapan. Aku suka cerita yang emosinya positif dan tidak
> ingin cerita dengan kekerasan berlebihan.

Contoh structured query:

```json
{
  "age": 17,
  "topics": ["friendship", "hope"],
  "genres": [],
  "emotion_preferences": ["joy", "trust", "anticipation"],
  "content_constraints": ["avoid excessive violence"],
  "semantic_query": "A story about friendship and finding hope with a positive emotional tone",
  "keyword_query": ["friendship", "hope"],
  "needs_clarification": false,
  "clarification_question": null
}
```

Contoh tersebut hanya ilustrasi. Jangan memaksa model mengisi genre atau
emosi jika pengguna tidak menyebutkannya. Field yang tidak diketahui
harus bernilai `null` atau list kosong, bukan ditebak.

### 6.3 Skema output query

Gunakan schema tervalidasi (misalnya Pydantic) agar keluaran LLM tidak
langsung dipakai sebagai struktur bebas.

Field minimal: - `age`: integer atau null; - `topics`: list string; -
`genres`: list string; - `emotion_preferences`: list label emosi yang
valid atau kosong; - `content_constraints`: list string; -
`semantic_query`: string; - `keyword_query`: list string; -
`needs_clarification`: boolean; - `clarification_question`: string atau
null.

Jika output LLM bukan JSON valid: 1. coba satu kali perbaikan/parse
ulang; 2. jika tetap gagal, gunakan fallback yang aman; 3. jangan
hentikan seluruh aplikasi karena satu keluaran tidak valid.

### 6.4 Percakapan multi-turn

Simpan konteks percakapan yang diperlukan agar pengguna dapat
menyampaikan preferensi secara bertahap.

Contoh: - User: "Aku ingin buku tentang persahabatan." - User: "Yang
emosinya lebih positif." - Sistem harus mempertahankan topik
persahabatan dan memperbarui preferensi emosi, bukan menganggap pesan
kedua sebagai query yang sepenuhnya terpisah.

Sediakan tombol atau perintah untuk memulai percakapan baru/menghapus
konteks sesi.

---

## 7. Hybrid RAG retrieval

### 7.1 Dense retrieval

1. Ambil `semantic_query`.
2. Buat embedding query menggunakan embedding model yang sama dengan
   embedding dokumen.
3. Cari kemiripan query terhadap embedding buku di vector database.
4. Ambil sejumlah kandidat awal, misalnya `dense_top_k = 20`.

Nilai Top-K harus dapat diatur melalui konfigurasi.

### 7.2 Sparse retrieval dengan BM25

1. Gunakan `keyword_query` atau teks query yang telah dibersihkan.
2. Jalankan BM25 pada korpus `document_text`.
3. Ambil kandidat awal, misalnya `bm25_top_k = 20`.
4. Pastikan hasil memiliki `book_id` yang sama dengan dataset utama.

Gunakan implementasi BM25 yang stabil dan terdokumentasi. Untuk
prototype, implementasi library seperti `rank_bm25` dapat
dipertimbangkan.

### 7.3 Fusion dengan RRF

Gabungkan ranking dense retrieval dan BM25 menggunakan Reciprocal Rank
Fusion.

Rumus umum:

\[ RRF(d)=`\sum`{=tex}\_{r `\in `{=tex}R}`\frac{1}{k+rank_r(d)}`{=tex}
\]

Keterangan: - (d): dokumen/buku; - (R): daftar ranking dari retriever; -
(rank_r(d)): posisi buku pada hasil retriever ke-(r); - (k): konstanta
RRF.

Sediakan parameter `rrf_k` dan `candidate_top_k` di konfigurasi.

Jangan menjumlahkan skor mentah cosine/vector distance dengan skor BM25
secara langsung karena skala skor dapat berbeda. RRF menggunakan posisi
ranking agar penggabungan lebih terkontrol.

### 7.4 Output retrieval

Setiap kandidat minimal membawa: - `book_id`; - `Title`; - `Author`; -
`Description`; - `Age` jika tersedia; - `Average_Rating` jika
tersedia; - `Emotion_Vector`; - skor/posisi dense retrieval jika
tersedia; - skor/posisi BM25 jika tersedia; - skor RRF.

Retrieval menghasilkan kandidat, bukan rekomendasi final.

### 7.5 Metadata filtering

Filter hanya metadata yang tersedia dan definisinya jelas. Jangan
menerapkan filter Genre, Age, atau Content Warning jika datanya kosong
atau tidak dapat ditafsirkan.

Jika batasan pengguna tidak dapat diverifikasi dari metadata, jangan
menyatakan bahwa sistem telah menjamin batasan tersebut. Sistem dapat
menyebutkan keterbatasan informasi dengan jujur.

---

## 8. Emotion Vector recommender

### 8.1 Peran Emotion Vector

Emotion Vector buku diambil dari dataset asli berdasarkan `book_id`.
Nilainya tidak dihitung ulang.

Komponen ini membandingkan: - User Emotion Preference; - Book Emotion
Vector.

### 8.2 User Emotion Preference

Sistem perlu mempunyai cara membentuk preferensi emosi pengguna.
Implementasikan sebagai modul terpisah agar strategi dapat diuji dan
diganti.

Prioritas desain: 1. jika paper sumber menyediakan mekanisme Age-Based
Preference yang dapat direplikasi dari informasi yang tersedia,
implementasikan mekanisme tersebut sesuai definisi sumber; 2. jika
pengguna secara eksplisit memilih emosi, sistem boleh membentuk
preferensi dari pilihan tersebut sebagai mode tambahan; 3. jika tidak
ada preferensi eksplisit dan mekanisme age-based belum dapat direplikasi
secara valid, jangan mengarang formula. Minta pengguna memilih
preferensi emosi atau nonaktifkan emotional ranking untuk sesi itu
dengan penjelasan yang jelas.

Jangan mengasumsikan bahwa daftar emosi pilihan pengguna otomatis setara
dengan User Emotion Vector dalam paper. Dokumentasikan metode konversi
pilihan emosi menjadi vektor, jika digunakan.

### 8.3 Cosine Similarity

Untuk setiap kandidat buku, hitung:

\[ S\_{emotion}(u,b)= `\frac{P_u \cdot V_b}`{=tex} {\|P_u\|\|V_b\|} \]

Keterangan: - (P_u): User Emotion Preference; - (V_b): Emotion Vector
buku dari dataset.

Periksa kasus vektor nol agar tidak terjadi pembagian dengan nol. Jika
salah satu vektor tidak valid, tandai skor sebagai tidak tersedia dan
jangan membuat skor palsu.

### 8.4 Jangan salah mengartikan skor

Cosine Similarity menunjukkan keselarasan arah dua vektor, bukan bukti
bahwa buku pasti membuat pengguna merasa emosi tertentu. Gunakan istilah
seperti: - "kesesuaian profil emosi"; - "emotional alignment score"; -
"kemiripan preferensi emosi".

Hindari klaim psikologis atau klaim bahwa sistem dapat memastikan dampak
emosional buku terhadap pembaca.

---

## 9. Final re-ranking

Gabungkan relevansi retrieval dan emotional alignment setelah keduanya
tersedia.

Contoh rancangan:

\[
S\_{final}(b)=`\alpha `{=tex}S\_{retrieval}(b)+`\beta `{=tex}S\_{emotion}(b)
\]

Namun, skor retrieval perlu dinormalisasi/ditransformasikan terlebih
dahulu agar skala sebanding dengan cosine similarity. Jangan langsung
menggabungkan skor RRF mentah dengan cosine similarity tanpa pemeriksaan
skala.

Persyaratan: - `alpha` dan `beta` dapat dikonfigurasi; - nilai awal
harus dinyatakan sebagai baseline eksperimen, bukan nilai optimal; -
ranking harus dapat dijalankan meskipun skor emosi tidak tersedia,
dengan fallback yang terdokumentasi; - urutkan kandidat berdasarkan
`final_score`; - simpan komponen skor secara terpisah untuk audit.

Contoh struktur hasil:

```json
{
  "book_id": "B001",
  "title": "Example Book",
  "retrieval_score": 0.72,
  "emotion_score": 0.91,
  "final_score": 0.83,
  "rank": 1
}
```

Angka di atas hanya ilustrasi.

Jangan menggunakan `Average_Rating` sebagai komponen skor final kecuali
kelompok secara eksplisit memutuskan untuk meneliti pengaruh rating dan
menambahkan eksperimen terpisah.

---

## 10. LLM response generation

Setelah Top-N ditentukan oleh kode recommender, kirimkan hanya informasi
kandidat yang sudah ditemukan dan di-ranking ke LLM.

LLM bertugas: - menyapa/merespons pengguna secara natural; - menjelaskan
mengapa buku direkomendasikan; - menyajikan judul dan penulis dengan
benar; - menyampaikan sinopsis berdasarkan Description; - menjelaskan
kesesuaian emosi dengan hati-hati; - menyebutkan informasi usia/rating
hanya jika tersedia; - mengakui jika metadata tertentu tidak tersedia.

LLM **tidak** bertugas: - memilih ranking berdasarkan intuisi sendiri; -
mengubah skor hasil recommender; - menambahkan buku yang tidak ada di
daftar kandidat; - mengarang tema, content warning, tahun terbit, atau
detail cerita; - mengklaim content warning tidak ada ketika datanya
unknown.

### 10.1 Prompt grounding

Gunakan instruksi yang setara dengan:

```text
You are a book recommendation assistant.
Use only the book records supplied in the context.
Do not invent books, authors, ratings, age information, publication years,
themes, content warnings, or plot details.
Do not change the ranking or scores produced by the recommendation engine.
Explain recommendations using the supplied description and metadata.
If a field is missing, say that the information is unavailable.
Describe emotion scores as preference alignment, not as a guarantee
of how the reader will feel.
Return a clear and concise response in Indonesian.
```

Lakukan pemeriksaan pasca-generasi: - judul yang disebut LLM harus
berasal dari Top-N; - penulis harus sesuai record; - skor yang
ditampilkan harus sama dengan skor recommender; - jika validasi gagal,
gunakan template jawaban deterministik dari hasil ranking.

---

## 11. Llama dan Qwen: konfigurasi eksperimen

Buat konfigurasi agar model dapat dipilih, misalnya:

```env
LLM_PROVIDER=local_or_api
LLM_MODEL=llama-3.1-8b-instruct
```

atau:

```env
LLM_PROVIDER=local_or_api
LLM_MODEL=qwen2.5-7b-instruct
```

Nama model aktual dan provider harus disesuaikan dengan runtime yang
benar-benar tersedia.

### 11.1 Perbandingan yang adil

Saat membandingkan kedua model, samakan: - dataset; - embedding model; -
vector database/index; - prompt template; - parameter retrieval; - RRF
configuration; - emotion ranking method; - alpha/beta; - Top-N; - test
queries.

Yang diganti hanya model LLM. Catat versi model/provider, temperature,
max tokens, dan parameter inference lainnya.

Jangan menyimpulkan model yang lebih baik hanya dari satu atau dua
contoh percakapan. Siapkan kumpulan query uji yang sama untuk kedua
model.

---

## 12. Antarmuka chatbot

Buat antarmuka sederhana, jelas, dan dapat digunakan untuk demo.

### 12.1 Elemen UI minimal

1. Judul aplikasi dan penjelasan singkat.
2. Pemilih model: Llama 3.1 8B atau Qwen 2.5 7B.
3. Area percakapan.
4. Input teks pengguna.
5. Informasi preferensi yang berhasil diekstrak, bila relevan.
6. Daftar rekomendasi Top-N dalam card.
7. Setiap card menampilkan:
   - judul;
   - penulis;
   - sinopsis singkat;
   - rating jika tersedia;
   - informasi Age sesuai definisi dataset;
   - emotional alignment score;
   - alasan rekomendasi.
8. Indikator ketika sistem sedang memproses.
9. Pesan error yang dapat dipahami.
10. Tombol percakapan baru/reset.

Jangan menampilkan field kosong sebagai "None" atau "NaN". Sembunyikan
atau tampilkan "Informasi tidak tersedia".

### 12.2 Transparansi

Sediakan detail/expandable section yang dapat menampilkan: - retrieval
score; - emotion score; - final score; - metadata yang digunakan; -
model LLM yang sedang dipakai.

Tujuannya agar hasil rekomendasi dapat diperiksa saat demo dan evaluasi.

---

## 13. Evaluasi dan pengujian

Buat evaluasi terpisah untuk retrieval, ranking, dan kualitas jawaban
LLM.

### 13.1 Retrieval evaluation

Gunakan query uji yang sama dan penilaian relevansi kandidat yang
konsisten.

Metrik yang dapat dipakai: - Recall@K; - Precision@K; - MRR@K; - NDCG@K.

Bandingkan sekurang-kurangnya: 1. Dense Retrieval saja; 2. Hybrid
Retrieval (Dense + BM25 + RRF).

Jangan mengklaim Hybrid RAG lebih baik sebelum hasil pengujian tersedia.

### 13.2 Emotion ranking evaluation

Periksa: - apakah Emotion Vector kandidat berhasil diambil dengan
benar; - apakah cosine similarity dihitung dengan benar; - apakah
ranking berubah secara konsisten ketika preferensi emosi berubah; -
apakah invalid/missing vector ditangani dengan benar.

Buat unit test untuk cosine similarity menggunakan vektor contoh yang
hasil matematisnya diketahui.

### 13.3 LLM evaluation

Gunakan query dan konteks hasil retrieval yang sama untuk Llama dan
Qwen.

Aspek penilaian: - ketepatan ekstraksi query; - kepatuhan pada
konteks; - ketepatan judul dan penulis; - relevansi penjelasan; - tidak
mengarang fakta; - konsistensi format jawaban.

Pisahkan kualitas LLM dari kualitas retrieval/ranking. Jika buku yang
ditemukan salah, jangan langsung menyalahkan response generation.

### 13.4 Ablation/comparison yang disarankan

Jika waktu memungkinkan, bandingkan: - Baseline A: Dense Retrieval +
Emotion Vector ranking; - Model B: Hybrid Retrieval + Emotion Vector
ranking.

Untuk membandingkan LLM: - jalankan pipeline yang sama dengan Llama; -
jalankan pipeline yang sama dengan Qwen.

Jangan mengubah beberapa komponen sekaligus saat ingin mengukur pengaruh
satu komponen.

---

## 14. Struktur folder yang disarankan

Sesuaikan dengan repository yang sudah ada. Jangan membuat ulang proyek
atau mengganti framework tanpa memeriksa struktur saat ini.

Contoh struktur:

```text
project/
├── app/
│   ├── main.py
│   ├── ui/
│   │   └── chatbot.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── query_parser.py
│   │   ├── retrieval_service.py
│   │   ├── emotion_ranker.py
│   │   └── response_generator.py
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── preprocessing.py
│   │   └── enrichment.py
│   └── schemas/
│       ├── query_schema.py
│       └── book_schema.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── preprocess_dataset.py
│   ├── enrich_metadata.py
│   └── build_index.py
├── data/
│   ├── raw/
│   └── processed/
├── vector_store/
├── reports/
├── tests/
│   ├── test_dataset.py
│   ├── test_emotion_vector.py
│   ├── test_retrieval.py
│   └── test_ranking.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Folder dan nama file boleh disesuaikan setelah Antigravity memeriksa
repository. Struktur ini merupakan usulan, bukan kewajiban untuk
memaksakan refactor besar.

---

## 15. Rekomendasi teknologi

Pilih teknologi berdasarkan kondisi repository dan kemudahan
implementasi.

  Komponen            Pilihan awal

---

  Bahasa              Python
  Data processing     pandas
  Validation/schema   Pydantic
  Embedding           Model embedding terpisah, multilingual bila diperlukan
  Vector database     ChromaDB
  Sparse retrieval    BM25 / `rank_bm25`
  Fusion              Reciprocal Rank Fusion
  LLM                 Llama 3.1 8B Instruct dan Qwen 2.5 7B Instruct
  UI                  Streamlit atau framework yang sudah digunakan repository
  Testing             pytest

Jangan menambahkan framework yang tidak diperlukan. Untuk prototype
dengan waktu terbatas, utamakan implementasi yang sederhana, modular,
dan dapat dijalankan.

---

## 16. Rencana kerja tiga hari

### Hari 1 --- Dataset, validasi, enrichment seperlunya

Target: - memeriksa dataset aktual; - memastikan format Emotion
Vector; - membersihkan data tanpa mengubah nilai Emotion Vector; -
mencoba mengambil Genre dan Publication Year jika mudah; - menyusun
`document_text`; - menghasilkan laporan dataset.

Batas waktu enrichment: jangan sampai menghambat indexing. Themes dan
Content Warning tidak wajib.

### Hari 2 --- Knowledge base dan retrieval

Target: - memilih embedding model; - membangun ChromaDB/index; -
mengimplementasikan dense retrieval; - mengimplementasikan BM25; -
menggabungkan hasil dengan RRF; - menguji retrieval menggunakan beberapa
query contoh.

### Hari 3 --- Emotion ranking, LLM, UI, dan pengujian

Target: - menghubungkan kandidat dengan Emotion Vector; -
mengimplementasikan User Emotion Preference sesuai metode yang telah
dipastikan; - menghitung cosine similarity; - mengimplementasikan final
ranking; - menghubungkan Llama dan Qwen tanpa fine-tuning; - membuat UI
chatbot; - menjalankan pengujian end-to-end dan memperbaiki error utama.

Jika ada fitur yang tidak selesai, prioritaskan alur inti yang berjalan
dan dokumentasikan fitur yang ditunda.

---

## 17. Urutan instruksi untuk Antigravity

Kerjakan secara bertahap. Jangan langsung membuat seluruh aplikasi dalam
satu perubahan besar.

### Tahap 0 --- Inspect before coding

1. Periksa struktur repository.
2. Temukan dataset yang sudah diberikan.
3. Baca header dan sampel dataset.
4. Identifikasi format Emotion Vector.
5. Periksa dependency dan aplikasi yang sudah ada.
6. Laporkan ringkasan temuan dan rencana implementasi.
7. Jangan menghapus, memindahkan, atau menimpa file sumber.

**Output tahap 0:** laporan inspeksi dataset/repository dan daftar
asumsi yang masih perlu dikonfirmasi.

### Tahap 1 --- Dataset loader dan validator

Implementasikan loader, pemetaan kolom, parser Emotion Vector,
preprocessing, dan validation report.

**Syarat selesai:** - dataset dapat dibaca; - kolom aktual terpetakan; -
Emotion Vector berhasil diparse tanpa mengubah nilai valid; - data
invalid tercatat; - file asli tidak berubah.

### Tahap 2 --- Metadata enrichment terbatas

Implementasikan enrichment Genre dan/atau Publication Year jika waktu
dan API memungkinkan.

**Syarat selesai:** - enrichment dapat dijalankan secara terpisah; -
ISBN diprioritaskan sebagai kunci pencarian; - sumber data dicatat; -
kegagalan tidak menghentikan seluruh pipeline; - tidak ada nilai yang
dikarang.

Themes dan Content Warning tidak perlu dikerjakan jika berisiko
menghabiskan waktu atau tidak ada sumber yang dapat diverifikasi.

### Tahap 3 --- Knowledge base dan indexing

Implementasikan document builder, embedding generation, dan vector
database.

**Syarat selesai:** - jumlah record ter-index dilaporkan; - record dapat
dicari; - metadata dan Emotion Vector dapat diambil kembali berdasarkan
`book_id`; - indexing dapat diulang tanpa duplikasi.

### Tahap 4 --- Hybrid retrieval

Implementasikan dense retrieval, BM25, dan RRF.

**Syarat selesai:** - kedua retriever mengembalikan `book_id`; - fusion
menghasilkan ranking kandidat; - metadata buku dan Emotion Vector
kandidat dapat diambil; - sistem menangani query kosong atau hasil
retrieval kosong.

### Tahap 5 --- Emotion ranking

Implementasikan modul terpisah untuk: - membentuk User Emotion
Preference; - mengambil Emotion Vector buku; - menghitung cosine
similarity; - menangani vektor invalid/nol; - menghasilkan emotion
score.

**Syarat selesai:** - hasil cosine similarity lolos unit test; - urutan
dimensi konsisten; - tidak ada perhitungan ulang Emotion Vector buku.

Jika metode Age-Based Preference dari paper belum dapat dipastikan,
jangan membuat formula sendiri. Buat interface/modul yang dapat diisi
setelah definisinya dikonfirmasi.

### Tahap 6 --- Final ranking

Implementasikan penggabungan skor retrieval dan emotion.

**Syarat selesai:** - bobot dapat dikonfigurasi; - skala skor diperiksa
sebelum digabung; - setiap skor komponen disimpan; - fallback berjalan
saat emotion score tidak tersedia.

### Tahap 7 --- LLM integration

Implementasikan: - provider/model configuration; - Llama 3.1 8B
Instruct; - Qwen 2.5 7B Instruct; - query understanding; - response
generation; - JSON validation; - fallback jika model gagal atau
mengembalikan format salah.

**Syarat selesai:** - model dapat dipilih; - keduanya memakai prompt dan
pipeline yang sama; - tidak ada fine-tuning; - LLM tidak dapat
menambah/mengubah ranking final.

### Tahap 8 --- Chatbot UI dan end-to-end test

Implementasikan UI, sesi percakapan, pilihan model, kartu rekomendasi,
dan transparansi skor.

**Syarat selesai:** - pengguna dapat mengirim query; - sistem
menghasilkan kandidat; - emotion ranking dijalankan; - Top-N
ditampilkan; - penjelasan sesuai record; - error ditampilkan dengan
jelas.

---

## 18. Kriteria penerimaan akhir

Implementasi dianggap memenuhi tugas jika:

- [ ] Dataset asli digunakan sebagai basis.
- [ ] Dataset asli tidak ditimpa.
- [ ] Emotion Vector yang sudah tersedia tidak dihitung ulang.
- [ ] Format dan urutan Emotion Vector divalidasi.
- [ ] Metadata enrichment tidak menghasilkan fakta rekaan.
- [ ] Sistem tetap berjalan tanpa Themes dan Content Warning.
- [ ] Embedding dokumen dan query menggunakan model yang sama.
- [ ] Dense retrieval berfungsi.
- [ ] BM25 berfungsi.
- [ ] RRF menggabungkan hasil retrieval.
- [ ] Emotion Vector buku diambil berdasarkan ID buku.
- [ ] Cosine Similarity dihitung dengan benar.
- [ ] Final ranking dapat diaudit melalui skor komponennya.
- [ ] Llama dan Qwen dapat dipilih tanpa fine-tuning.
- [ ] Kedua LLM diuji dengan kondisi yang sama.
- [ ] Jawaban LLM hanya menggunakan buku dan metadata yang ditemukan.
- [ ] UI menampilkan rekomendasi Top-N.
- [ ] README menjelaskan setup, konfigurasi, indexing, menjalankan
  aplikasi, dan pengujian.

---

## 19. Instruksi penting untuk coding agent

Gunakan instruksi berikut sebagai aturan kerja:

1. **Inspect first, then implement.** Jangan menebak nama file, header
   dataset, format Emotion Vector, framework, atau provider LLM.
2. Jangan menghapus atau menimpa dataset asli.
3. Jangan menghitung ulang Emotion Vector atau mengimplementasikan
   ulang NRC-EIL.
4. Jangan melakukan fine-tuning Llama/Qwen.
5. Jangan membuat metadata buku yang tidak didukung sumber.
6. Jangan menganggap Themes wajib.
7. Jangan menganggap Description mencakup seluruh tema buku.
8. Jangan menyatakan Content Warning tidak ada hanya karena datanya
   kosong.
9. Jangan menganggap kolom Age sebagai minimum reading age sebelum
   definisinya diverifikasi.
10. Pisahkan retrieval, emotion ranking, final ranking, dan response
    generation menjadi modul yang dapat diuji.
11. Jangan mengklaim performa meningkat sebelum evaluasi dilakukan.
12. Buat perubahan kecil dan jalankan test setelah setiap tahap.
13. Jika ada informasi yang tidak tersedia, gunakan konfigurasi atau
    minta konfirmasi; jangan mengarang.
14. Utamakan prototype end-to-end yang benar dan dapat dijalankan
    daripada fitur tambahan yang tidak selesai.

---

## 20. Prompt pembuka yang dapat ditempel ke Antigravity

```text
Baca seluruh dokumen spesifikasi ini sebelum melakukan perubahan kode.

Tugasmu adalah membangun prototype chatbot rekomendasi buku berbasis Hybrid RAG dan Emotion Vector.

KONTEKS KRITIS:
Dataset buku sudah tersedia dan sudah memiliki kolom Emotion_Vector dari penelitian sumber. Jangan menghitung ulang NRC-EIL, jangan membuat ulang Emotion Vector, dan jangan melakukan fine-tuning terhadap Llama atau Qwen. Pengembangan berfokus pada metadata enrichment seperlunya, knowledge base/embedding, Hybrid RAG retrieval, Emotion Vector ranking, integrasi Llama 3.1 8B Instruct dan Qwen 2.5 7B Instruct tanpa fine-tuning, serta UI chatbot.

Sebelum coding:
1. Inspeksi repository dan dataset aktual.
2. Tampilkan nama kolom, beberapa sampel, tipe data, missing values, dan format Emotion_Vector.
3. Periksa definisi kolom Age jika dokumentasinya tersedia.
4. Periksa framework dan dependency yang sudah ada.
5. Buat rencana implementasi berdasarkan kondisi aktual.
6. Jangan menghapus atau menimpa file sumber.

Implementasikan secara bertahap sesuai Tahap 0 sampai Tahap 8 dalam dokumen. Setelah setiap tahap, jalankan pemeriksaan atau test yang relevan dan laporkan file yang dibuat/diubah, hasil test, serta kendala.

Jangan mengarang metadata. Genre dan Publication Year hanya di-enrich jika ada sumber yang dapat diverifikasi. Themes bersifat opsional karena Description belum tentu mencakup seluruh tema. Content Warning tidak wajib dan nilai kosong tidak boleh dianggap berarti tidak ada konten sensitif.

Pastikan Emotion Vector selalu terhubung dengan buku melalui ID yang stabil. Pisahkan retrieval score, emotion score, dan final score. LLM hanya menjelaskan hasil ranking; LLM tidak boleh mengubah ranking atau menambahkan buku yang tidak ditemukan.

Jangan langsung melakukan refactor besar atau membuat seluruh aplikasi sekaligus. Mulai dari inspeksi, laporkan temuan, lalu implementasikan tahap demi tahap.
```

---

## 21. Keputusan yang masih harus dikonfirmasi sebelum hasil dianggap final

Beberapa detail tidak boleh ditebak oleh coding agent:

1. **Lokasi dan format dataset aktual:** file CSV/XLSX dan nama kolom
   sebenarnya.
2. **Definisi kolom `Age`:** apakah usia rata-rata pengguna, kelompok
   usia, atau atribut lain.
3. **Urutan sembilan dimensi Emotion Vector:** gunakan dokumentasi
   dataset/paper.
4. **Metode User Emotion Preference:** apakah akan mereplikasi
   mekanisme age-based dari paper, menggunakan pilihan emosi eksplisit,
   atau menguji keduanya.
5. **Provider/runtime LLM:** lokal atau API, sesuai akses dan kemampuan
   perangkat.
6. **Embedding model:** pilih satu model dan gunakan konsisten untuk
   dokumen serta query.
7. **Jumlah Top-N dan bobot ranking:** tetapkan sebagai parameter
   eksperimen, bukan klaim nilai optimal.

**Prioritas implementasi:** dataset asli yang sudah memiliki Emotion
Vector → validasi → enrichment seperlunya → embedding dan knowledge base
→ Hybrid Retrieval → Emotion Vector ranking → Llama/Qwen response
generation → UI dan evaluasi.

------------------------------------------------------------------------

## 22. Rancangan RAG (End-to-End Implementation Blueprint)

Bagian ini merupakan rancangan teknis dan panduan langkah demi langkah implementasi sistem **Hybrid RAG + Emotion Vector Recommender** dari awal hingga akhir berdasarkan kondisi dataset terkini (`Enrichment/books_enriched_with_themes.csv`).

### 22.1 Ringkasan Komponen Sistem
Sistem dibangun dalam 5 modul terpisah yang saling terhubung secara modular:

```text
[ Dataset Bersih & Enriched: 19.784 Buku ]
                     |
                     v
+--------------------+---------------------+
| MODUL 1: KNOWLEDGE BASE & INDEXING       |
| - Text Constructor: Title+Author+Desc+G+T|
| - Dense Vector Store: ChromaDB (MiniLM)  |
| - Sparse Index: BM25 (rank_bm25)         |
| - Emotion Companion Store: SQLite/Parquet|
+--------------------+---------------------+
                     |
                     v
+--------------------+---------------------+
| MODUL 2: QUERY UNDERSTANDING (LLM)       |
| - Input: Pesan Percakapan Pengguna       |
| - Engine: Llama 3.1 8B / Qwen 2.5 7B     |
| - Output: JSON Terstruktur (Pydantic)    |
|   (semantic_query, keywords, emotions)   |
+--------------------+---------------------+
                     |
                     v
+--------------------+---------------------+
| MODUL 3: HYBRID RETRIEVAL & RRF FUSION   |
| - Dense Search: Top-K (ChromaDB)         |
| - Sparse Search: Top-K (BM25)            |
| - Rank Fusion: Reciprocal Rank Fusion    |
| - Candidates Pool: Top-C Candidates      |
+--------------------+---------------------+
                     |
                     v
+--------------------+---------------------+
| MODUL 4: EMOTION RANKING & RE-RANKING    |
| - User Emotion Vector vs Book Vector     |
| - Metric: Cosine Similarity (9 dimensi)  |
| - Re-ranking: Alpha*Retrieval + Beta*Emo |
| - Final Selection: Top-N Books           |
+--------------------+---------------------+
                     |
                     v
+--------------------+---------------------+
| MODUL 5: GROUNDED RESPONSE & CHATBOT UI  |
| - Response Generation: Llama / Qwen      |
| - Prompt Grounding (Anti-Halusinasi)     |
| - Interactive UI: Streamlit              |
|   (Chat, Model Switcher, Score Inspector)|
+------------------------------------------+
```

---

### 22.2 Rincian Tahapan Pembangunan (Langkah Demi Langkah)

#### Tahap A: Penyusunan Teks Dokumen (`Document Builder`)
* **File Input:** `Enrichment/books_enriched_with_themes.csv` (19.784 baris).
* **Format Teks Representasi (`document_text`):**
  ```text
  Title: {title}
  Author: {author}
  Description: {description}
  Genre: {genre_if_available}
  Themes: {themes_if_available}
  ```
* **Aturan:**
  * Nilai metadata kosong dihilangkan dari baris (tidak menampilkan `None` atau `NaN`).
  * Kolom audit teknis (`genre_source`, `themes_source`, `EmotionVector`) **tidak dimasukkan ke dalam teks embedding** agar tidak mengotori makna semantik.

#### Tahap B: Membangun Knowledge Base (ChromaDB + BM25 + Emotion Store)
1. **Dense Vector Store (ChromaDB):**
   * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensi, cepat dan efisien di CPU).
   * **Penyimpanan:** Persistent folder `data/vector_store/chroma_db/`.
   * **Metadata per entri:** `book_id`, `ISBN`, `Title`, `Author`, `Average_Rating`, `Age`, `Genre`, `Themes`.
2. **Sparse Retrieval Store (BM25):**
   * Tokenisasi korpus teks `document_text` menggunakan `rank_bm25` (BM25Okapi).
   * Index disimpan ke `data/vector_store/bm25_index.pkl`.
3. **Companion Emotion Store:**
   * Menyimpan dictionary vektor 9 dimensi: `Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust, Objective`.
   * Disimpan ke `data/vector_store/emotion_store.json` / Parquet dengan kunci utama `book_id` untuk lookup $O(1)$.

#### Tahap C: Query Understanding Service
* **Tugas:** Menerjemahkan pesan santai / percakapan pengguna menjadi parameter penelusuran.
* **Skema Pydantic (`query_schema.py`):**
  ```python
  class StructuredQuery(BaseModel):
      age: Optional[int] = None
      semantic_query: str
      keyword_query: List[str]
      emotion_preferences: List[str]  # joy, sadness, trust, etc.
      genre_preferences: List[str]
      content_constraints: List[str]
  ```
* **Fallback Rule:** Jika LLM gagal memproduksi format JSON yang valid, gunakan fallback regex / keyword parser sederhana sehingga proses retrieval tidak pernah crash.

#### Tahap D: Hybrid Retrieval Service & RRF Fusion
1. **Dense Retrieval:** Menghitung embedding query dari `semantic_query`, mencari kemiripan kosinus terhadap dokumen di ChromaDB $\rightarrow$ menghasilkan `dense_rank(b)`.
2. **BM25 Retrieval:** Mencari kecocokan kata kunci dari `keyword_query` pada korpus BM25 $\rightarrow$ menghasilkan `bm25_rank(b)`.
3. **Reciprocal Rank Fusion (RRF):**
   $$RRF(b) = \frac{1}{k + rank_{dense}(b)} + \frac{1}{k + rank_{bm25}(b)}$$
   (Parameter default: $k = 60$, mengambil kandidat awal $Top-K = 20$).

#### Tahap E: Emotion Vector Recommender & Final Re-Ranking
1. **User Emotion Profiling:**
   * Membentuk vektor 9 dimensi pengguna ($P_u$) berdasarkan preferensi emosi yang diekstrak LLM atau pilihan emosi langsung.
   * Setiap dimensi emosi diberi bobot ternormalisasi.
2. **Cosine Similarity:**
   $$S_{emotion}(u, b) = \frac{P_u \cdot V_b}{\|P_u\| \|V_b\|}$$
   (Dengan penanganan kasus vektor nol / missing value).
3. **Skor Akhir (Final Re-Ranking):**
   $$S_{final}(b) = \alpha \cdot S_{retrieval\_norm}(b) + \beta \cdot S_{emotion}(b)$$
   (Default eksperimen: $\alpha = 0.5$, $\beta = 0.5$, dapat disesuaikan di UI).
4. Memilih **Top-N Buku Terbaik** (misal $N = 5$).

#### Tahap F: LLM Response Generation (Prompt Grounding)
* **Model Inference:** Llama 3.1 8B Instruct atau Qwen 2.5 7B Instruct (via Groq API / Provider lokal).
* **Grounding Rule:** LLM hanya menerima data Top-N buku yang ditemukan beserta sinopsis, skor relevansi, dan skor emosinya.
* **Larangan:** LLM tidak boleh merekomendasikan buku di luar daftar konteks yang diberikan dan tidak boleh mengarang tema / fakta cerita.

#### Tahap G: Interactive Chatbot UI (Streamlit)
* **Sidebar Controls:**
  * Pilihan Model LLM: `Llama 3.1 8B Instruct` vs `Qwen 2.5 7B Instruct`.
  * Parameter Slider: Bobot $\alpha$ (Retrieval) vs $\beta$ (Emotion), Jumlah Rekomendasi Top-N.
* **Main Chat Area:** Percakapan natural multi-turn dengan histori sesi.
* **Recommendation Cards:**
  * Menampilkan Judul, Penulis, Rating, Usia Pembaca (Goodreads Reader Age), Genre, Themes.
  * Kartu Skor Transparan: Retrieval Score, Emotion Alignment Score, Final Score.
  * Alasan rekomendasi yang dihasilkan oleh LLM.

---

### 22.3 Rencana Struktur Folder Kode RAG
```text
PROJECT RS/
├── app/
│   ├── main.py                      # Entrypoint aplikasi Streamlit
│   ├── config.py                    # Konfigurasi model, parameter alpha/beta, API Key
│   ├── schemas/
│   │   ├── query_schema.py          # Pydantic schema untuk query understanding
│   │   └── book_schema.py           # Pydantic schema untuk data buku & skor
│   └── services/
│       ├── llm_service.py           # Integrasi Llama 3.1 & Qwen 2.5 (Groq/API)
│       ├── query_parser.py          # Ekstraksi preferensi pengguna via LLM
│       ├── indexing_service.py      # Builder dokumen & ChromaDB + BM25 indexer
│       ├── retrieval_service.py     # Dense search + BM25 search + RRF Fusion
│       ├── emotion_ranker.py        # Komputasi Cosine Similarity 9 dimensi emosi
│       └── response_generator.py    # Grounded generator penjelasan rekomendasi
├── scripts/
│   ├── build_knowledge_base.py     # Skrip mandiri membangun ChromaDB & BM25
│   └── test_rag_pipeline.py        # Unit testing penelusuran & ranking
├── data/
│   └── vector_store/                # Folder penyimpanan index ChromaDB & BM25
└── reports/
    └── rag_evaluation_report.md     # Evaluasi retrieval (Dense vs Hybrid RAG)
```
