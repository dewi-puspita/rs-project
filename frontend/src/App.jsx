import React, { useState } from 'react';
import {
  BookOpen,
  Loader2,
  Star,
  User,
  FileText,
  HelpCircle,
  Cpu,
  AlertCircle,
  Compass,
  Heart,
  Calendar,
  ListOrdered
} from 'lucide-react';

const AGE_RANGES = [
  { id: '12-13', label: '12 – 13 Tahun' },
  { id: '14-15', label: '14 – 15 Tahun' },
  { id: '16-17', label: '16 – 17 Tahun' },
  { id: '18-19', label: '18 – 19 Tahun' }
];

const EMOTION_LIST = [
  'joy',
  'trust',
  'anticipation',
  'surprise',
  'sadness',
  'fear',
  'anger',
  'disgust',
  'objective'
];

const PRESET_QUERIES = [
  "Lagi galau dan butuh cerita coming-of-age yang penuh persahabatan hangat",
  "Cerita fantasi sihir penuh misteri dan petualangan menegangkan",
  "Kisah sci-fi distopia tentang perjuangan remaja melawan tirani",
  "Novel romantis remaja sekolah yang manis dan bikin senyum-senyum sendiri"
];

export default function App() {
  const [query, setQuery] = useState('');
  const [modelName, setModelName] = useState('llama3.1:8b');
  const [selectedAgeRange, setSelectedAgeRange] = useState(null);
  const [selectedEmotions, setSelectedEmotions] = useState({});
  const [topN, setTopN] = useState(5);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const toggleAgeRange = (id) => {
    setSelectedAgeRange((prev) => (prev === id ? null : id));
  };

  const toggleEmotion = (em) => {
    setSelectedEmotions((prev) => {
      const next = { ...prev };
      if (next[em]) {
        delete next[em];
      } else {
        next[em] = 1.0;
      }
      return next;
    });
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          model_name: modelName,
          alpha: 0.5,
          beta: 0.5,
          age_range: selectedAgeRange,
          emotions: Object.keys(selectedEmotions).length > 0 ? selectedEmotions : null,
          top_n: topN
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (Status ${response.status})`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Gagal mendapatkan rekomendasi:", err);
      setError(err.message || "Gagal menghubungi server backend. Pastikan API FastAPI telah berjalan.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header Bersih & Rapi */}
      <header className="app-header">
        <h1 className="app-title">Sistem Rekomendasi Buku Remaja</h1>
        <p className="app-subtitle">
          Chatbot rekomendasi buku berbasis <strong>Hybrid RAG</strong> & analisis <strong>Emosi (NRC-EIL)</strong>
        </p>
      </header>

      {/* Main Single-Column Clean Layout */}
      <main className="main-content">
        {/* Form Input Terpadu */}
        <section className="query-card">
          <form onSubmit={handleSearch}>
            <label className="form-label" style={{ fontSize: '0.95rem', marginBottom: '8px' }}>
              Ceritakan buku apa yang ingin kamu baca atau suasana hatimu saat ini:
            </label>
            <textarea
              className="textarea-input"
              placeholder="Contoh: Aku lagi galau banget dan butuh cerita coming-of-age tentang persahabatan..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />

            {/* Contoh Kueri Cepat */}
            <div className="preset-container">
              <span className="preset-label">Contoh:</span>
              {PRESET_QUERIES.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="preset-btn"
                  onClick={() => setQuery(p)}
                >
                  {p.length > 36 ? p.substring(0, 36) + '...' : p}
                </button>
              ))}
            </div>

            {/* Filter Rentang Usia Pembaca (RecSys '25 Age Cohorts) */}
            <div className="filter-section">
              <div className="filter-header">
                <Calendar size={15} color="var(--primary)" />
                <span>Rentang Usia Pembaca (Opsional):</span>
              </div>
              <div className="chips-container">
                {AGE_RANGES.map((ar) => (
                  <button
                    key={ar.id}
                    type="button"
                    className={`filter-chip ${selectedAgeRange === ar.id ? 'active' : ''}`}
                    onClick={() => toggleAgeRange(ar.id)}
                  >
                    {ar.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Filter Mood / Emosi Opsional */}
            <div className="filter-section" style={{ borderTop: 'none', paddingTop: '4px' }}>
              <div className="filter-header">
                <Heart size={15} color="var(--primary)" />
                <span>Filter Mood / Suasana Hati (Opsional):</span>
              </div>
              <div className="chips-container">
                {EMOTION_LIST.map((em) => (
                  <button
                    key={em}
                    type="button"
                    className={`filter-chip ${selectedEmotions[em] ? 'active' : ''}`}
                    onClick={() => toggleEmotion(em)}
                  >
                    {em}
                  </button>
                ))}
              </div>
            </div>

            {/* Pengaturan Model LLM & Jumlah Buku (Berada persis di atas tombol kirim) */}
            <div className="control-row">
              <div className="control-item">
                <label className="form-label">
                  Model LLM:
                </label>
                <select
                  className="select-input"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                >
                  <option value="llama3.1:8b">Llama 3.1 8B</option>
                  <option value="qwen2.5:7b">Qwen 2.5 7B</option>
                </select>
              </div>

              <div className="control-item">
                <label className="form-label">
                  Jumlah Buku:
                </label>
                <select
                  className="select-input"
                  value={topN}
                  onChange={(e) => setTopN(parseInt(e.target.value, 10))}
                >
                  <option value="3">Top 3 Buku</option>
                  <option value="5">Top 5 Buku</option>
                </select>
              </div>
            </div>

            {/* Tombol Rekomendasikan */}
            <button
              type="submit"
              className="submit-btn"
              disabled={loading || !query.trim()}
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={20} /> Memproses rekomendasi...
                </>
              ) : (
                <>
                  <BookOpen size={20} /> Rekomendasikan Buku
                </>
              )}
            </button>
          </form>
        </section>

        {/* Error Notice */}
        {error && (
          <div className="error-box">
            <AlertCircle size={22} />
            <div>
              <strong>Terjadi Kendala:</strong>
              <p style={{ fontSize: '0.9rem', marginTop: '4px' }}>{error}</p>
            </div>
          </div>
        )}

        {/* Results Display */}
        {results && (
          <section>
            <div className="results-header">
              <h2 className="results-title">
                <BookOpen size={22} color="var(--primary)" /> Hasil Rekomendasi Buku
              </h2>
              <span className="results-meta">
                {results.books?.length || 0} Buku Ditemukan • Model: {results.model_used === 'qwen2.5:7b' ? 'Qwen 2.5 7B' : 'Llama 3.1 8B'}
              </span>
            </div>

            {/* Kartu Rekomendasi Buku: Hanya 5 Field Wajib */}
            <div className="book-cards-list">
              {results.books && results.books.length > 0 ? (
                results.books.map((b, idx) => (
                  <article key={idx} className="book-card">
                    {/* Header Kartu: Peringkat & Genre */}
                    <div className="book-card-header">
                      <div className="rank-wrapper">
                        <span className="rank-badge">#{b.rank || idx + 1}</span>
                        <span className="genre-label">
                          {b.genre ? b.genre : 'Young Adult / Remaja'}
                        </span>
                      </div>
                    </div>

                    {/* 5 Field Tampilan Utama */}
                    <div className="field-container">
                      {/* 1. Judul */}
                      <div className="field-item">
                        <span className="field-label label-judul">
                          <BookOpen size={14} /> Judul
                        </span>
                        <h3 className="field-value-judul">{b.judul}</h3>
                      </div>

                      {/* 2. Penulis */}
                      <div className="field-item">
                        <span className="field-label label-penulis">
                          <User size={14} /> Penulis
                        </span>
                        <p className="field-value-penulis">{b.penulis}</p>
                      </div>

                      {/* 3. Rating */}
                      <div className="field-item">
                        <span className="field-label label-rating">
                          <Star size={14} /> Rating
                        </span>
                        <p className="rating-text">
                          ⭐ {b.rating ? b.rating.toFixed(2) : 'Belum ada rating'} / 5.0
                        </p>
                      </div>

                      {/* 4. Sinopsis Singkat */}
                      <div className="field-item">
                        <span className="field-label label-sinopsis">
                          <FileText size={14} /> Sinopsis Singkat
                        </span>
                        <div className="field-value-sinopsis">
                          {b.sinopsis_singkat}
                        </div>
                      </div>

                      {/* 5. Alasan */}
                      <div className="field-item">
                        <span className="field-label label-alasan">
                          <HelpCircle size={14} /> Alasan Rekomendasi
                        </span>
                        <div className="field-value-alasan">
                          {b.alasan}
                        </div>
                      </div>
                    </div>
                  </article>
                ))
              ) : (
                <div className="empty-state">
                  Tidak ada rekomendasi buku yang sesuai dengan kueri tersebut.
                </div>
              )}
            </div>
          </section>
        )}

        {!results && !loading && (
          <div className="empty-state">
            <BookOpen size={40} style={{ marginBottom: '14px', opacity: 0.35 }} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
              Belum Ada Rekomendasi
            </h3>
            <p style={{ maxWidth: '440px', margin: '6px auto 0 auto', fontSize: '0.88rem' }}>
              Ketikkan cerita atau kueri buku di atas, tentukan rentang usia / mood jika perlu, lalu klik <strong>Rekomendasikan Buku</strong>.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
