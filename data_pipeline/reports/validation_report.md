# Laporan Validasi dan Preprocessing Dataset

- **Waktu Eksekusi**: Dataset dibersihkan dan divalidasi.
- **Dataset Asli**: `Teenager_GoodReads_Emotion.csv`
- **Dataset Bersih**: `books_clean.csv`

## 1. Ringkasan Baris Data
- **Jumlah Baris Awal**: 19,804
- **Duplikat ID Dihapus**: 20 baris
- **Jumlah Baris Bersih**: 19,784 baris (100% ID Unik)
- **Jumlah Penulis Unik (Author)**: 9,832
- **Jumlah Judul Unik (Title)**: 17,074

> **Catatan Penting**: Penulis yang sama dengan judul buku berbeda tetap dipertahankan 100%. Hanya baris dengan `ID` buku yang identik/terduplikasi yang dirapikan.

## 2. Pembersihan Teks & Case Folding
- Kolom `Title`, `Author`, dan `Description` telah dibersihkan dari leading/trailing whitespace (`.strip()`) dan dinormalisasi menjadi lowercase.
- Spasi ganda di dalam teks telah dinormalisasi menjadi spasi tunggal.
- String kosong terdeteksi: Title (38), Author (5), Description (8). Record ini tetap dipertahankan karena ID dan EmotionVector-nya valid, dan dapat dilengkapi pada tahap metadata enrichment.

## 3. Validasi Emotion Vector
- **Status**: 100% Valid (19,804/19,804)
- Tidak ada vektor kosong, `NaN`, atau `Inf`.
- Nilai numerik Emotion Vector asli dari NRC-EIL **tidak diubah** sama sekali.

## 4. Statistik Kolom Numerik
- **Average Rating**: Min 0.0 | Max 5.0 | Rata-rata 3.8
- **Age (Average Reader Age)**: Min 12.0 | Max 19.0 | Rata-rata 16.68 tahun
