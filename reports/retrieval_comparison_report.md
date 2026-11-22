# Laporan Hasil Uji Komparasi Hybrid Retrieval (Tahap 7)
*Sesuai Rancangan Tugas v2 (Bab 5, Bab 6, dan Bab 8.4)*

- **Tanggal Pengujian**: 2026-11-22 09:54:55
- **Metode Fusi**: Reciprocal Rank Fusion (RRF) dengan parameter konstan $k = 60$
- **Total Korpus Eligible**: 19.778 buku
- **Hardware**: CPU (Inference lokal)

---

## 1. Konfigurasi Query Uji Coba

| Parameter Query | Nilai Input |
|---|---|
| **Query Asli (Bahasa Indonesia)** | *"Aku ingin membaca cerita tentang persahabatan, petualangan, dan perjuangan remaja."* |
| **Query Terjemahan (English)** | *"A story about friendship, adventure, and the struggles of teenagers."* |
| **Kata Kunci Sparse (BM25 English)** | *"friendship adventure teenage struggles"* |

---

## 2. Hasil Top-5 Jalur A: Multilingual E5 Large + BM25
*Dense Input: `query: <Query Asli ID>` | Sparse: BM25Okapi on Document Text (EN)*

| Rank | RRF Score | Book ID | Judul Buku | Penulis | Posisi Dense E5 | Posisi BM25 |
|:---:|:---:|:---:|---|---|:---:|:---:|
| **1** | 0.01639 | `76366` | **yesterday i cried celebrating the lessons of living and loving** | iyanla vanzant | Rank 1 (0.8067) | - |
| **2** | 0.01639 | `4077065` | **so you re a teenage girl** | jill renich meyers | - | Rank 1 (13.9836) |
| **3** | 0.01613 | `243908` | **it s my life a workout for your mind** | tian dayton | Rank 2 (0.8036) | - |
| **4** | 0.01613 | `124750` | **chicken soup for the teenage soul on love and friendship** | jack canfield mark victor hansen | - | Rank 2 (13.0495) |
| **5** | 0.01587 | `3330026` | **jesus freaks** | d c talk | Rank 3 (0.8025) | - |

---

## 3. Hasil Top-5 Jalur B: BGE Large English v1.5 + BM25
*Dense Input: `Represent this sentence for searching relevant passages: <Query EN>` | Sparse: BM25Okapi*

| Rank | RRF Score | Book ID | Judul Buku | Penulis | Posisi Dense BGE | Posisi BM25 |
|:---:|:---:|:---:|---|---|:---:|:---:|
| **1** | 0.02748 | `427920` | **chicken soup for the teenage soul** | jack canfield mark victor hansen kimberly kirberger | Rank 17 (0.6626) | Rank 9 (10.0266) |
| **2** | 0.02655 | `202722` | **plague year** | stephanie s tolan | Rank 19 (0.6603) | Rank 12 (9.7154) |
| **3** | 0.01639 | `800245` | **on friendship a book for teenagers** | kimberly kirberger colin mortensen | Rank 1 (0.7133) | - |
| **4** | 0.01639 | `4077065` | **so you re a teenage girl** | jill renich meyers | - | Rank 1 (13.9836) |
| **5** | 0.01613 | `452826` | **more than a friend love stories for young adults** | elizabeth winfrey | Rank 2 (0.6956) | - |

---

## 4. Analisis Overlap & Karakteristik Komparasi

| Metrik Komparasi | Nilai Evaluasi |
|---|---|
| **Jumlah Kandidat Top-5 yang Sama** | **1 dari 5 buku** (20%) |
| **Jaccard Similarity (Top-5)** | **0.1111** |
| **Buku yang Beririsan** | `4077065` |

### Temuan Analisis:
1. **Perilaku Jalur A (Multilingual E5)**:
   - Mampu menangani kueri Bahasa Indonesia secara langsung tanpa langkah translasi.
   - Mengambil buku-buku pengembangan diri dan refleksi remaja (*lessons of living and loving*, *workout for your mind*).
2. **Perilaku Jalur B (BGE Large English v1.5)**:
   - Bekerja melalui kueri Bahasa Inggris (*Translate-then-Search*).
   - Menghasilkan keselarasan leksikal-semantik yang sangat kuat dengan BM25 (*Chicken Soup for the Teenage Soul*, *On Friendship: A Book for Teenagers*), menghasilkan skor fusi RRF yang lebih tinggi di posisi teratas.
3. **Kesimpulan**:
   - Kedua model berhasil menemukan kandidat buku yang sangat relevan dengan tema persahabatan (*friendship*), petualangan (*adventure*), dan remaja (*teenagers*).
   - Penggunaan RRF terbukti efektif mengangkat buku yang disetujui bersama oleh Dense Search dan Sparse BM25 ke peringkat tertinggi.
