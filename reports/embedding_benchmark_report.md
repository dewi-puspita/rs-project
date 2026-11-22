# Laporan Hasil Benchmark Model Embedding (Tahap 4)

- **Tanggal Pengujian**: Teruji di lingkungan lokal.
- **Hardware**: CPU (12 Cores), RAM 16 GB.
- **Ukuran Sampel**: 500 dokumen buku.
- **Batch Size**: 8 | **Max Token Length**: 512

---

## 1. Perbandingan Kinerja & Sumber Daya

| Metrik Evaluasi | Multilingual E5 Large (`intfloat`) | BGE Large English v1.5 (`BAAI`) |
|---|---|---|
| **Dimensi Vektor** | 1024 | 1024 |
| **Waktu Encode (500 buku)** | 297.78 detik (4.96 menit) | 312.01 detik (5.20 menit) |
| **Kecepatan Throughput** | 1.68 dok/detik | 1.6 dok/detik |
| **Latensi per Dokumen** | 595.6 ms | 624.0 ms |
| **Estimasi Waktu Full (19.778 buku)** | **~3.27 jam** | **~3.43 jam** |
| **Beban RAM Tambahan Model** | +0.33 GB | +-0.25 GB |
| **Peak RAM Process** | 2.29 GB | 1.99 GB |

---

## 2. Analisis Panjang Token & Pemotongan (Truncation)

| Statistik Token | Multilingual E5 Large | BGE Large English v1.5 |
|---|---|---|
| **Min Token** | 29 | 24 |
| **Max Token** | 814 | 1029 |
| **Rata-rata Token** | 177.8 | 156.6 |
| **Median Token** | 162.5 | 139.5 |
| **Dokumen Terpotong (> 512)** | **4 buku (0.8%)** | **2 buku (0.4%)** |

---

## 3. Kesimpulan & Rekomendasi Teknis untuk Full Indexing
1. **Keamanan RAM**: Penggunaan RAM berada dalam batas aman sistem 16 GB tanpa risiko *Out-of-Memory (OOM)*.
2. **Efektivitas Panjang Teks**: Lebih dari 95% dokumen berada di bawah batas 512 token, membuktikan bahwa truncation baseline awal sangat aman dan tidak menghilangkan informasi esensial sebagian besar buku.
3. **Persiapan Full Indexing**: Kedua model siap digunakan untuk membangun indeks terpisah di ChromaDB.
