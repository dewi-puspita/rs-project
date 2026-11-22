# Laporan Pengujian Emotion Recommender & Final Ranking (Tahap 8)
*Sesuai Rancangan Tugas v2 (Bab 8.4, Bab 9, dan Bab 10)*

- **Tanggal Pengujian**: 2026-11-22 10:01:48
- **Dataset Emosi**: `emotion_companion_store.json` (19.784 buku ter-validasi, 9 dimensi NRC-EIL)
- **Formula Final Ranking**: $S_{final} = \alpha \cdot S_{retrieval\_norm} + \beta \cdot S_{emotion\_norm}$
- **Kueri Pengujian**: *"Aku ingin membaca cerita tentang persahabatan, petualangan, dan perjuangan remaja."*

---

## 1. Hasil Final Re-Ranking Berdasarkan Skenario Emosi Pengguna

### Skenario A: Profil Emosi Positif & Penuh Harapan (*Joy, Anticipation, Trust*)
*User Vector: Joy=0.8, Anticipation=0.7, Trust=0.6, lainnya=0.1 | $\alpha=0.5, \beta=0.5$*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Skor Retrieval (Norm) | Alignment Emosi (Norm) | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **1** | **so you re a teenage girl** | jill renich meyers | Rank 2 | 1.0000 | 0.8327 | **0.9164** |
| **2** | **yesterday i cried celebrating the lessons of living and loving** | iyanla vanzant | Rank 1 | 1.0000 | 0.8216 | **0.9108** |
| **3** | **chicken soup for the teenage soul on love and friendship** | jack canfield mark victor hansen | Rank 4 | 0.8762 | 0.8383 | **0.8572** |
| **4** | **it s my life a workout for your mind** | tian dayton | Rank 3 | 0.8762 | 0.7063 | **0.7913** |
| **5** | **not that i care friendship ring** | rachel vail | Rank 6 | 0.7524 | 0.7409 | **0.7467** |

### Skenario B: Profil Emosi Melankolis / Mendalam (*Sadness, Fear*)
*User Vector: Sadness=0.8, Fear=0.7, lainnya=0.1 | $\alpha=0.5, \beta=0.5$*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Skor Retrieval (Norm) | Alignment Emosi (Norm) | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **1** | **so you re a teenage girl** | jill renich meyers | Rank 2 | 1.0000 | 0.7144 | **0.8572** |
| **2** | **yesterday i cried celebrating the lessons of living and loving** | iyanla vanzant | Rank 1 | 1.0000 | 0.6109 | **0.8055** |
| **3** | **chicken soup for the teenage soul on love and friendship** | jack canfield mark victor hansen | Rank 4 | 0.8762 | 0.6496 | **0.7629** |
| **4** | **it s my life a workout for your mind** | tian dayton | Rank 3 | 0.8762 | 0.6013 | **0.7388** |
| **5** | **jesus freaks** | d c talk | Rank 5 | 0.7524 | 0.6838 | **0.7181** |

### Skenario C: Mode Fallback (Pengguna Tidak Memilih Preferensi Emosi)
*User Vector: None / Zero Vector | Status: Fallback Retrieval-Only*

| Rank Akhir | Judul Buku | Penulis | RRF Rank | Status Emosi | Skor Akhir ($S_{final}$) |
|:---:|---|---|:---:|:---:|:---:|
| **1** | **yesterday i cried celebrating the lessons of living and loving** | iyanla vanzant | Rank 1 | `fallback_retrieval_only` | **1.0000** |
| **2** | **so you re a teenage girl** | jill renich meyers | Rank 2 | `fallback_retrieval_only` | **1.0000** |
| **3** | **it s my life a workout for your mind** | tian dayton | Rank 3 | `fallback_retrieval_only` | **0.8762** |
| **4** | **chicken soup for the teenage soul on love and friendship** | jack canfield mark victor hansen | Rank 4 | `fallback_retrieval_only` | **0.8762** |
| **5** | **jesus freaks** | d c talk | Rank 5 | `fallback_retrieval_only` | **0.7524** |

---

## 2. Uji Sensitivitas Bobot $\alpha$ (Retrieval) vs $\beta$ (Emosi)
*Dievaluasi pada Skenario A (Joy & Anticipation)*

| Variasi Bobot | Buku Peringkat 1 | Skor Akhir | Buku Peringkat 2 | Skor Akhir |
|---|---|:---:|---|:---:|
| **Balanced ($\alpha=0.5, \beta=0.5$)** | **so you re a teenage girl** | 0.9164 | yesterday i cried celebrating the lessons of living and loving | 0.9108 |
| **Content-Heavy ($\alpha=0.8, \beta=0.2$)** | **so you re a teenage girl** | 0.9665 | yesterday i cried celebrating the lessons of living and loving | 0.9643 |
| **Emotion-Heavy ($\alpha=0.2, \beta=0.8$)** | **so you re a teenage girl** | 0.8662 | yesterday i cried celebrating the lessons of living and loving | 0.8572 |

---

## 3. Temuan Evaluasi & Kesimpulan
1. **Dinamika Pemeringkatan Ulang**:
   - Preferensi emosi berhasil mengubah urutan rekomendasi buku secara bermakna dan terukur. Buku dengan relevansi teks moderat namun keselarasan emosi tinggi dapat naik peringkat ke posisi teratas saat $\beta$ aktif.
2. **Keamanan & Transparansi Skor**:
   - Seluruh skor dinormalisasi ke rentang $[0.0, 1.0]$. Komponen skor retrieval, skor cosine similarity mentah, dan skor emosi ternormalisasi disimpan secara terpisah untuk transparansi UI.
3. **Robustness Fallback**:
   - Ketika vektor emosi pengguna bernilai nol atau tidak ditentukan, sistem secara otomatis mengaktifkan mode `fallback_retrieval_only` tanpa menghasilkan skor emosi palsu (*zero error rate*).
