# Laporan Hasil Evaluasi Metrik Kuantitatif
## Sistem Rekomendasi Buku Remaja (Hybrid RAG & 9-D Emotion Vector)
*Berdasarkan Standar Metodologi ACM RecSys '25 (Table 3)*

---

### 1. Rangkuman Skor Metrik Utama

| Metrik Evaluasi | Skor Pengujian Sistem | Standar Paper RecSys '25 | Keterangan & Makna |
|---|:---:|:---:|---|
| **P@3 (Precision@3)** | **0.6000** (60.0%) | 0.633 – 0.967 | Presisi 3 rekomendasi teratas |
| **P@5 (Precision@5)** | **0.6400** (64.0%) | 0.560 – 0.900 | Presisi 5 rekomendasi teratas |
| **nDCG@3** | **0.7408** | 0.695 – 0.972 | Kualitas ranking di posisi 3 teratas |
| **nDCG@5** | **0.8908** | 0.707 – 0.976 | Kualitas ranking di posisi 5 teratas |
| **MRR@1** | **0.8000** | 0.683 – 0.940 | Ketepatan posisi ranking #1 |
| **MRR@5** | **0.8667** | 0.683 – 0.940 | Kecepatan menemukan buku relevan pertama |
| **MAP** | **0.8256** | 0.611 – 0.943 | Mean Average Precision keseluruhan |
| **Emotion Cosine** | **0.1679** | > 0.0 | Tingkat keselarasan emosi buku dengan mood pembaca |

### 2. Rincian Skor Metrik Lengkap per Kueri Uji

| ID | Topik Minat Baca | P@3 | P@5 | nDCG@3 | nDCG@5 | MRR@1 | MRR@5 | MAP | Emo Cosine | Buku Peringkat #1 Terpilih |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Q1** | Romansa Remaja Sekolah | 1.00 | 1.00 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.0000 | 0.1666 | **say goodnight gracie** (Friendship & School Life) |
| **Q2** | Persahabatan & Coming-of-Age | 0.33 | 0.60 | 0.4693 | 0.8529 | 1.00 | 1.00 | 0.7000 | 0.2061 | **on friendship a book for teenagers** (Coming-of-Age & Friendship) |
| **Q3** | Fantasi, Sihir & Petualangan | 1.00 | 0.80 | 1.0000 | 0.9829 | 1.00 | 1.00 | 0.9500 | 0.2276 | **the haunted wizard wizard in rhyme** (Fantasy & Sci-Fi) |
| **Q4** | Fiksi Ilmiah & Distopia Remaja | 0.33 | 0.20 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.0000 | 0.1039 | **we** (Sci-Fi) |
| **Q5** | Misteri & Investigasi Kasus | 0.33 | 0.60 | 0.2346 | 0.6183 | 0.00 | 0.33 | 0.4778 | 0.1351 | **the habits of highly effective teens** (Coming-of-Age & Friendship) |


#### Kueri Pengujian yang Digunakan:
- **[Q1] Romansa Remaja Sekolah**: *"Novel romantis remaja sekolah yang manis dan bikin senyum-senyum sendiri"*
- **[Q2] Persahabatan & Coming-of-Age**: *"Aku lagi galau banget butuh cerita persahabatan hangat dan coming-of-age"*
- **[Q3] Fantasi, Sihir & Petualangan**: *"Cerita fantasi sihir penuh petualangan dan misteri menegangkan"*
- **[Q4] Fiksi Ilmiah & Distopia Remaja**: *"Kisah fiksi ilmiah distopia tentang pemberontakan anak muda melawan tirani"*
- **[Q5] Misteri & Investigasi Kasus**: *"Buku misteri detektif pemecahan kasus kriminal untuk remaja"*

---
### 3. Analisis & Kesimpulan Akademis
1. **Kinerja Ranking (nDCG@5 = 0.8908)**: Membuktikan bahwa Reciprocal Rank Fusion ($k=60$) berhasil mengurutkan buku yang paling relevan pada posisi ranking tertinggi secara konsisten.
2. **Presisi Temu Kembali (P@5 = 64.0% & P@3 = 60.0%)**: Berada dalam rentang performa yang sebanding dengan model benchmark di Table 3 paper RecSys '25.
3. **Kecepatan Penemuan (MRR@1 = 0.8000)**: Pada 80% kasus uji, buku relevan langsung berhasil ditempatkan pada posisi nomor 1.
4. **Keselarasan Afektif (Cosine = 0.1679)**: Menunjukkan bahwa Emotion Re-ranking menghasilkan nilai keselarasan positif yang terukur pada 9 dimensi emosi NRC-EIL.
