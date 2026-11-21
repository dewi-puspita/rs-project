# Laporan Validasi Dataset Final (Tahap 0 & 1)

- **Tanggal Eksekusi**: Terverifikasi secara terpadu.
- **Dataset Sumber**: `Fixed_Genre_Themes_GoodReads_Emotion.csv`
- **Status Spesifikasi**: Sesuai dengan Rantug v2.0 (Bab 2, 3, dan 15).

---

## 1. Ringkasan Verifikasi Kolom & ID
- **Total Baris**: 19,784 baris.
- **Total Kolom**: 12 kolom (Semua kolom standar terpenuhi).
- **Integritas Canonical ID**: 100% Unik (19,784 buku unik, 0 duplikat, 0 kosong).

## 2. Kelayakan Indexing Retrieval (Dense & BM25)
Sesuai Bab 3.2 Rantug v2, buku layak masuk indeks retrieval jika setidaknya memiliki **Title** atau **Description**:
- **Buku Layak Index (Eligible)**: **19,778 buku (99.97%)**
- **Buku Dikecualikan dari Index**: **6 buku (0.03%)**

### Daftar 6 Buku yang Dikecualikan dari Index:
| Book ID | ISBN | Author | Alasan Pengecualian |
|---|---|---|---|
| `35600350` | `35600350` | ivan alekseyevich bunin | Title dan Description sama-sama kosong |
| `13543545` | `9789600509885` | milan kundera yannis haris | Title dan Description sama-sama kosong |
| `10292017` | `9789602936900` | voula mastori | Title dan Description sama-sama kosong |
| `20438752` | `9789602740507` | robert arthur | Title dan Description sama-sama kosong |
| `43904399` | `043904399` | *(kosong)* | Title dan Description sama-sama kosong |
| `44022840` | `044022840` | leo tolstoy | Title dan Description sama-sama kosong |

> **Catatan Penting**: 6 buku di atas **TIDAK dihapus** dari dataset sumber dan **TETAP disimpan di Emotion Companion Store**.

## 3. Validasi Emotion Vector
- **Parser**: `ast.literal_eval()` (safe evaluation).
- **Hasil**: **19,784 / 19,784 (100% Valid)**.
- **Struktur**: Tepat 9 dimensi (`Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust, Objective`).
- **Tipe Data**: Semua numerik finite, tanpa `NaN`, tanpa `Infinity`. Nilai asli tidak diubah.

## 4. Status Metadata Enrichment
- **Genre**: 18,120 buku (91.59%)
- **Themes**: 4,564 buku (23.07%)
*(Genre dan Themes disimpan sebagai metadata terpisah di ChromaDB untuk baseline awal).*
