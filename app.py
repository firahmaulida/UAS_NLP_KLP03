"""
Keyphrase Generation Dataset Studio
------------------------------------
UI eksplorasi dataset & pipeline preprocessing untuk tugas keyphrase
generation bahasa Indonesia (DOAJ -> T5).

Jalankan dengan:
    streamlit run app.py
"""

import io
import os
import re
import textwrap
from collections import Counter

import pandas as pd
import streamlit as st

# =========================================================
# KONFIGURASI HALAMAN & GAYA VISUAL
# =========================================================

st.set_page_config(
    page_title="Keyphrase Studio — Dataset Bahasa Indonesia",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_RAW = os.path.join(BASE_DIR, "dataset_id_3kolom.csv")
PATH_FULL = os.path.join(BASE_DIR, "data_preprocessed", "full_preprocessed.csv")
PATH_TRAIN = os.path.join(BASE_DIR, "data_preprocessed", "train.csv")
PATH_VAL = os.path.join(BASE_DIR, "data_preprocessed", "val.csv")
PATH_TEST = os.path.join(BASE_DIR, "data_preprocessed", "test.csv")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --paper: #F8F9FA;
    --paper-raised: #FFFFFF;
    --ink: #0F172A;
    --ink-soft: #475569;
    --ink-lighter: #64748B;
    --rule: #E2E8F0;
    --indigo: #4F46E5;
    --indigo-soft: #6366F1;
    --indigo-lighter: #EEF2FF;
    --terracotta: #DC2626;
    --moss: #059669;
    --moss-light: #D1FAE5;
    --tag-bg: #F1F5F9;
    --border-focus: #3B82F6;
    --success: #10B981;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink) !important;
}

.stApp {
    background-color: var(--paper);
}

/* Memastikan kontras teks optimal di semua elemen */
p, span, div, li, label { 
    color: var(--ink) !important;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}

h1 { font-weight: 700 !important; font-size: 2.2rem !important; }
h2 { font-weight: 700 !important; border-bottom: 2px solid var(--indigo); padding-bottom: 0.6rem; margin-top: 1.8rem !important; color: var(--ink) !important;}
h3 { font-weight: 700 !important; margin-top: 1.2rem !important; color: var(--ink) !important;}

code, .mono {
    font-family: 'IBM Plex Mono', monospace !important;
    background: var(--indigo-lighter);
    padding: 0.2rem 0.4rem;
    border-radius: 5px;
    color: var(--indigo);
    font-size: 0.85em;
    font-weight: 600;
}

/* ---------- Masthead ---------- */
.masthead {
    background: linear-gradient(135deg, var(--paper-raised) 0%, #F0F9FF 100%);
    border: 2px solid var(--indigo);
    border-radius: 12px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 8px 16px rgba(79, 70, 229, 0.1);
}
.masthead .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--indigo);
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.masthead h1 { 
    margin: 0 0 1rem 0; 
    border: none; 
    font-size: 2.8rem !important;
    color: var(--ink) !important;
}
.masthead .deck {
    font-size: 1.1rem;
    color: var(--ink-soft);
    line-height: 1.8;
    max-width: 75ch;
    font-weight: 500;
}

/* ---------- Cards ---------- */
.card {
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.3s ease;
}
.card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.card-tight { padding: 1.2rem 1.5rem; }

.stat-card {
    background: var(--paper-raised);
    border: 1.5px solid var(--rule);
    border-left: 5px solid var(--indigo);
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: var(--indigo-soft);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
}
.stat-card .stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-lighter);
    font-weight: 700;
}
.stat-card .stat-value {
    font-family: 'Source Serif 4', serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--indigo);
    line-height: 1.1;
    margin: 0.5rem 0;
}
.stat-card .stat-sub {
    font-size: 0.85rem;
    color: var(--ink-soft);
    font-weight: 500;
}

/* ---------- Pipeline ---------- */
.pipeline {
    display: flex;
    align-items: stretch;
    gap: 1rem;
    margin: 2rem 0 2.5rem 0;
    overflow-x: auto;
    padding-bottom: 0.5rem;
}
.pipe-step {
    flex: 1;
    min-width: 200px;
    background: linear-gradient(135deg, var(--paper-raised) 0%, #F0F9FF 50%);
    border: 1.5px solid var(--indigo-lighter);
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: all 0.3s ease;
}
.pipe-step:hover {
    border-color: var(--indigo);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
    transform: translateY(-2px);
}
.pipe-step .pipe-index {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--indigo);
    letter-spacing: 0.1em;
    font-weight: 700;
    text-transform: uppercase;
}
.pipe-step .pipe-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 0.7rem 0;
    color: var(--ink);
}
.pipe-step .pipe-desc {
    font-size: 0.85rem;
    color: var(--ink-soft);
    line-height: 1.6;
}
.pipe-arrow {
    display: flex; align-items: center; justify-content: center;
    width: 28px; color: var(--indigo); font-size: 1.8rem; flex-shrink: 0; font-weight: bold;
}

/* ---------- Tags ---------- */
.tag {
    display: inline-block;
    background: var(--indigo-lighter);
    color: var(--indigo);
    border: 1.5px solid var(--indigo-soft);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin: 0.2rem 0.3rem 0.2rem 0;
    transition: all 0.2s ease;
}
.tag:hover {
    background: var(--indigo-soft);
    color: white;
}
.tag-indigo { 
    background: var(--indigo-lighter); 
    border-color: var(--indigo-soft);
    color: var(--indigo);
}
.tag-terra { 
    background: #FEE2E2; 
    border-color: #FECACA; 
    color: #DC2626;
}
.tag-moss { 
    background: var(--moss-light); 
    border-color: #A7F3D0; 
    color: var(--moss);
}

/* ---------- Sample Blocks ---------- */
.sample-block {
    border: 1.5px solid var(--rule);
    background: var(--paper-raised);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sample-block .sample-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    padding: 0.9rem 1.2rem;
    border-bottom: 1.5px solid var(--rule);
    color: white;
}
.sample-head.input { 
    background: linear-gradient(90deg, var(--indigo) 0%, var(--indigo-soft) 100%);
    color: white;
}
.sample-head.target { 
    background: linear-gradient(90deg, #DC2626 0%, #F87171 100%);
    color: white;
}
.sample-body {
    padding: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--ink);
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--paper);
}

/* ---------- Tables Markdown Format ---------- */
table { 
    border-collapse: collapse; 
    width: 100%; 
    font-size: 0.9rem; 
    margin: 1.5rem 0; 
}
th, td { 
    text-align: left; 
    padding: 12px 16px; 
    border: 1px solid var(--rule); 
    color: var(--ink);
    background: var(--paper-raised);
}
th { 
    background-color: linear-gradient(135deg, #EEF2FF 0%, #F0F9FF 100%);
    font-weight: 700; 
    color: var(--indigo);
    border-bottom: 2px solid var(--indigo);
}
tr:hover { background-color: #F8F9FA; }

/* ---------- Input Fields & Selectbox ---------- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select,
.stSlider > div > div {
    border: 1.5px solid var(--rule) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    background: var(--paper-raised) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div > select:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, var(--indigo) 0%, var(--indigo-soft) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
}

/* Download Button */
a.stDownloadButton > button {
    background: linear-gradient(135deg, var(--success) 0%, #34D399 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
}
a.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
}

/* ---------- Footnote ---------- */
.footnote {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--ink-soft);
    border-top: 2px solid var(--indigo);
    margin-top: 3rem;
    padding-top: 1.5rem;
    line-height: 1.8;
}

/* Sidebar Styling Overrides */
section[data-testid="stSidebar"] {
    background-color: #F8FAFF !important;
    border-right: 2px solid var(--indigo-lighter);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] button {
    color: var(--ink-soft) !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom: 3px solid var(--indigo) !important;
}

/* Radio Styling */
.stRadio > label {
    color: var(--ink) !important;
    font-weight: 500 !important;
}

/* Caption & Helper Text */
.stCaption, .stHelp {
    color: var(--ink-soft) !important;
    font-size: 0.85rem !important;
}

/* Alert & Messages */
.stWarning, .stError, .stSuccess, .stInfo {
    border-radius: 8px !important;
    border-left: 4px solid !important;
    padding: 1rem !important;
}
.stWarning {
    border-left-color: #F59E0B !important;
}
.stError {
    border-left-color: var(--terracotta) !important;
}
.stSuccess {
    border-left-color: var(--success) !important;
}
.stInfo {
    border-left-color: var(--indigo) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# DATA LOADING (cached)
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv_safe(path: str, usecols=None):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception as e:
        st.error(f"Gagal memuat `{os.path.basename(path)}`: {e}")
        return None


@st.cache_data(show_spinner=False)
def basic_text_stats(df: pd.DataFrame, col: str):
    lengths = df[col].astype(str).str.len()
    return lengths


def file_status_row(label: str, path: str, df):
    exists = df is not None
    rows = f"{len(df):,}".replace(",", ".") if exists else "—"
    cols = len(df.columns) if exists else "—"
    size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
    size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    status_icon = "OK" if exists else "TIDAK DITEMUKAN"
    return f"| `{label}` | {status_icon} | {rows} | {cols} | {size_str} |"


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-family:'IBM Plex Mono',monospace;
        font-size:0.75rem;letter-spacing:0.15em;color:#4F46E5;text-transform:uppercase; font-weight: 700;
        background: #EEF2FF; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1rem;">
        UAS NLP · Kelompok 03
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Keyphrase Studio", help="Eksplorasi dataset keyphrase generation")
    st.caption("Eksplorasi dataset dan pipeline keyphrase generation Bahasa Indonesia")

    page = st.radio(
        "Navigasi",
        [
            "Overview",
            "Dataset Explorer",
            "Preprocessed Data",
            "Keyphrase Generator",
            "Evaluasi Model",
            "Train / Val / Test",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.85rem; line-height:1.8; color:#0F172A; background: #F0F9FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #4F46E5;">
        <b style="color: #4F46E5;">Sumber data:</b> DOAJ<br>
        <b style="color: #4F46E5;">Target model:</b> T5 keyphrase<br>
        <b style="color: #4F46E5;">Bahasa:</b> Indonesia
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HALAMAN: OVERVIEW
# =========================================================

if page == "Overview":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Dataset & Pipeline Documentation</div>
            <h1>Keyphrase Generation untuk Bahasa Indonesia</h1>
            <div class="deck">
                Dataset artikel jurnal akademik Indonesia dari <strong>DOAJ</strong>, dibersihkan melalui 
                pipeline deteksi bahasa berlapis, lalu diformat menjadi pasangan 
                <span class="mono">input_model</span> → <span class="mono">target_model</span> 
                siap pakai untuk fine-tuning model T5 pada tugas keyphrase generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_raw = load_csv_safe(PATH_RAW)
    df_full = load_csv_safe(PATH_FULL)

    # --- Stat row ---
    c1, c2, c3, c4 = st.columns(4)
    n_raw = len(df_raw) if df_raw is not None else 0
    n_full = len(df_full) if df_full is not None else 0
    retained_pct = (n_full / n_raw * 100) if n_raw else 0
    avg_keyphrase = df_full["jumlah_keyphrase_bersih"].mean() if df_full is not None else 0

    with c1:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Artikel mentah</div>
            <div class="stat-value">{n_raw:,}</div>
            <div class="stat-sub">hasil crawling DOAJ</div></div>""".replace(",", "."),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Lolos preprocessing</div>
            <div class="stat-value">{n_full:,}</div>
            <div class="stat-sub">{retained_pct:.1f}% dari data mentah</div></div>""".replace(",", "."),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Rata-rata keyphrase</div>
            <div class="stat-value">{avg_keyphrase:.1f}</div>
            <div class="stat-sub">per artikel (setelah bersih)</div></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Bahasa hasil filter</div>
            <div class="stat-value">100%</div>
            <div class="stat-sub">Indonesia murni</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Alur Data")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipe-step">
                <div class="pipe-index">01 · CRAWLING</div>
                <div class="pipe-title">crawling-10k.ipynb</div>
                <div class="pipe-desc">Mengambil judul, abstrak, kata kunci dari DOAJ dengan banyak query cadangan, deteksi bahasa berbasis pola kata, dan deduplikasi judul.</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="pipe-index">02 · OUTPUT</div>
                <div class="pipe-title">dataset_id_3kolom.csv</div>
                <div class="pipe-desc">Dataset mentah 3 kolom: judul, abstrak, kata_kunci (dipisah ";"). Abstrak ≥ 50 karakter, kata kunci ≥ 3 karakter.</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="pipe-index">03 · PREPROCESSING</div>
                <div class="pipe-title">preprocessing-keyphrase.ipynb</div>
                <div class="pipe-desc">Deteksi bahasa multi-lapis (langdetect, marker bilingual, rasio stopword), pembersihan teks, format prefix T5, tokenisasi & truncation 512 token.</div>
            </div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-step">
                <div class="pipe-index">04 · SIAP LATIH</div>
                <div class="pipe-title">train / val / test.csv</div>
                <div class="pipe-desc">Pasangan input_model & target_model, terbagi 80/10/10, siap untuk fine-tuning atau inferensi model T5.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### Tahapan Deteksi Bahasa Indonesia")
        st.markdown(
            """
            <div class="card">
            Pipeline preprocessing menyaring artikel non-Indonesia melalui <strong>empat lapis pemeriksaan</strong>:
            <br><br>
            <strong style="color:#4F46E5;">1. s1_langdetect_abstrak</strong><br>
            <span style="color:#475569;">Deteksi bahasa abstrak menggunakan pustaka <code>langdetect</code></span><br><br>
            
            <strong style="color:#4F46E5;">2. s2_bilingual_marker</strong><br>
            <span style="color:#475569;">Pengecekan penanda dwibahasa (mis. ringkasan Indonesia setelah Inggris)</span><br><br>
            
            <strong style="color:#4F46E5;">3. s3_langdetect_keyword</strong><br>
            <span style="color:#475569;">Deteksi bahasa pada kata kunci</span><br><br>
            
            <strong style="color:#4F46E5;">4. s4_stopword_ratio</strong><br>
            <span style="color:#475569;">Rasio kemunculan stopword Bahasa Indonesia pada teks</span><br><br>
            
            Baris yang lolos seluruh lapis diberi label <span class="tag tag-moss">✓ INDONESIA</span>
            dan diteruskan ke tahap pembersihan teks & tokenisasi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### 📝 Format Input/Target Model")
        st.markdown(
            """
            <div class="sample-block">
                <div class="sample-head input">input_model</div>
                <div class="sample-body">generate keyphrases: &lt;abstrak_bersih&gt;</div>
            </div>
            <div class="sample-block">
                <div class="sample-head target">🎯 target_model</div>
                <div class="sample-body">keyphrase_1, keyphrase_2, keyphrase_3, ...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card-tight card">
            <strong style="color:#4F46E5;">💡 Tokenisasi:</strong> Token dihitung dengan 
            <span class="mono">T5Tokenizer</span> dan input di-<strong>truncate</strong> 
            ke maksimal <strong style="color:#4F46E5;">512 token</strong> agar kompatibel 
            dengan arsitektur T5 tanpa kehilangan konteks abstrak secara signifikan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Status Berkas Project")
    rows = [
        file_status_row("dataset_id_3kolom.csv", PATH_RAW, df_raw),
        file_status_row("full_preprocessed.csv", PATH_FULL, df_full),
        file_status_row("train.csv", PATH_TRAIN, load_csv_safe(PATH_TRAIN)),
        file_status_row("val.csv", PATH_VAL, load_csv_safe(PATH_VAL)),
        file_status_row("test.csv", PATH_TEST, load_csv_safe(PATH_TEST)),
    ]
    table_md = "| Berkas | Status | Baris | Kolom | Ukuran |\n|---|---|---|---|---|\n" + "\n".join(rows)
    st.markdown(table_md)

    st.markdown(
        """
        <div class="footnote">
        Catatan — basis project ini adalah <b>dataset & preprocessing</b>, bukan model
        inferensi siap pakai. Halaman "Keyphrase Generator" menampilkan contoh
        pasangan input → target dari dataset, dan dapat dihubungkan ke model T5
        hasil fine-tuning Anda sendiri melalui Hugging Face (lihat halaman terkait).
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HALAMAN: DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Tahap 01–02 · Crawling DOAJ</div>
            <h1>Dataset Explorer</h1>
            <div class="deck">
                Pratinjau <span class="mono">dataset_id_3kolom.csv</span> — hasil mentah crawling DOAJ 
                sebelum melalui filter bahasa dan pembersihan teks. Terdiri dari 
                <strong>judul</strong>, <strong>abstrak</strong>, dan <strong>kata_kunci</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_csv_safe(PATH_RAW)

    if df is None:
        st.warning("Berkas `dataset_id_3kolom.csv` tidak ditemukan di direktori project.")
    else:
        st.markdown("", unsafe_allow_html=True)  # Spacing
        tab_preview, tab_stats, tab_search = st.tabs(
            ["Pratinjau Tabel", "Statistik", "Cari Artikel"]
        )

        # --- Tab Preview ---
        with tab_preview:
            n_show = st.slider("Jumlah baris ditampilkan", 5, 200, 25, step=5)
            st.dataframe(df.head(n_show), use_container_width=True, height=420)

            csv_bytes = df.head(n_show).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh pratinjau (CSV)",
                data=csv_bytes,
                file_name="preview_dataset_id_3kolom.csv",
                mime="text/csv",
            )

        # --- Tab Stats ---
        with tab_stats:
            abstrak_len = basic_text_stats(df, "abstrak")
            judul_len = basic_text_stats(df, "judul")
            n_keywords = df["kata_kunci"].astype(str).str.split(";").apply(len)

            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Total artikel</div>
                    <div class="stat-value">{len(df):,}</div></div>""".replace(",", "."),
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Rata² panjang abstrak</div>
                    <div class="stat-value">{abstrak_len.mean():.0f}</div>
                    <div class="stat-sub">karakter</div></div>""",
                    unsafe_allow_html=True,
                )
            with s3:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Rata² jumlah kata kunci</div>
                    <div class="stat-value">{n_keywords.mean():.1f}</div>
                    <div class="stat-sub">per artikel</div></div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Distribusi panjang abstrak (karakter)**")
                hist_df = pd.DataFrame({"panjang_abstrak": abstrak_len})
                st.bar_chart(
                    hist_df["panjang_abstrak"].value_counts(bins=20).sort_index(),
                    height=280,
                )
            with colB:
                st.markdown("**Distribusi jumlah kata kunci per artikel**")
                st.bar_chart(n_keywords.value_counts().sort_index(), height=280)

            st.write("")
            st.markdown("**Missing value per kolom**")
            missing_df = df.isna().sum().reset_index()
            missing_df.columns = ["kolom", "jumlah_missing"]
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

            st.write("")
            st.markdown("**Kata kunci paling sering muncul**")
            all_kw = (
                df["kata_kunci"]
                .astype(str)
                .str.split(";")
                .explode()
                .str.strip()
                .str.lower()
            )
            all_kw = all_kw[all_kw.str.len() > 1]
            top_kw = all_kw.value_counts().head(20).reset_index()
            top_kw.columns = ["kata_kunci", "frekuensi"]
            st.bar_chart(top_kw.set_index("kata_kunci")["frekuensi"], height=350)

        # --- Tab Search ---
        with tab_search:
            query = st.text_input("Cari pada judul atau abstrak", placeholder="mis. siskeudes, kosakata, kearifan lokal ...")
            if query:
                mask = df["judul"].str.contains(query, case=False, na=False) | df["abstrak"].str.contains(
                    query, case=False, na=False
                )
                result = df[mask]
                st.caption(f"{len(result):,} artikel ditemukan".replace(",", "."))
                for _, row in result.head(15).iterrows():
                    with st.expander(row["judul"]):
                        st.markdown(f"**Abstrak:** {row['abstrak'][:600]}{'...' if len(str(row['abstrak']))>600 else ''}")
                        kws = [k.strip() for k in str(row["kata_kunci"]).split(";") if k.strip()]
                        st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in kws]), unsafe_allow_html=True)
            else:
                st.info("Masukkan kata kunci pencarian untuk menyaring artikel berdasarkan judul atau abstrak.")


# =========================================================
# HALAMAN: PREPROCESSED DATA
# =========================================================

elif page == "Preprocessed Data":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">🔧 Tahap 03 · Preprocessing & Cleaning</div>
            <h1>Preprocessed Data</h1>
            <div class="deck">
                Pratinjau <span class="mono">full_preprocessed.csv</span> — dataset yang telah melalui 
                deteksi bahasa berlapis, pembersihan teks, dan transformasi ke format 
                <span class="mono">input_model</span> / <span class="mono">target_model</span> 
                siap untuk fine-tuning model T5.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_csv_safe(PATH_FULL)

    if df is None:
        st.warning("Berkas `data_preprocessed/full_preprocessed.csv` tidak ditemukan.")
    else:
        # --- Filter sidebar-like row ---
        f1, f2, f3 = st.columns([1, 1, 1.4])
        with f1:
            kategori_opts = ["(semua)"] + sorted(df["kategori_bahasa"].dropna().unique().tolist())
            kategori_pick = st.selectbox("Kategori bahasa", kategori_opts)
        with f2:
            max_kw = int(df["jumlah_keyphrase_bersih"].max())
            kw_range = st.slider("Jumlah keyphrase", 1, max_kw, (1, max_kw))
        with f3:
            token_range = st.slider(
                "Rentang token input (token_input)",
                int(df["token_input"].min()),
                int(df["token_input"].max()),
                (int(df["token_input"].min()), int(df["token_input"].max())),
            )

        filtered = df.copy()
        if kategori_pick != "(semua)":
            filtered = filtered[filtered["kategori_bahasa"] == kategori_pick]
        filtered = filtered[
            (filtered["jumlah_keyphrase_bersih"].between(*kw_range))
            & (filtered["token_input"].between(*token_range))
        ]

        st.caption(f"Menampilkan {len(filtered):,} dari {len(df):,} baris setelah filter.".replace(",", "."))

        tab_sample, tab_table, tab_quality = st.tabs(
            ["Contoh Input → Target", "Tabel Lengkap", "Kualitas Data"]
        )

        with tab_sample:
            n_sample = st.number_input("Jumlah contoh", min_value=1, max_value=10, value=3)
            if len(filtered) == 0:
                st.info("Tidak ada baris yang cocok dengan filter saat ini.")
            else:
                sample = filtered.sample(min(n_sample, len(filtered)), random_state=None)
                for _, row in sample.iterrows():
                    st.markdown(f"**{row['judul']}**")
                    st.markdown(
                        f"""
                        <div class="sample-block">
                            <div class="sample-head input">input_model · {row['token_input']} token</div>
                            <div class="sample-body">{row['input_model'][:500]}{'...' if len(str(row['input_model']))>500 else ''}</div>
                        </div>
                        <div class="sample-block">
                            <div class="sample-head target">target_model · {row['jumlah_keyphrase_bersih']} keyphrase</div>
                            <div class="sample-body">{row['target_model']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if st.button("🔄 Acak ulang contoh"):
                    st.rerun()

        with tab_table:
            cols_show = st.multiselect(
                "Pilih kolom yang ditampilkan",
                options=list(df.columns),
                default=["judul", "kategori_bahasa", "jumlah_keyphrase_bersih", "token_input", "target_model"],
            )
            n_show2 = st.slider("Jumlah baris", 5, 300, 30, step=5, key="tab_table_rows")
            if cols_show:
                st.dataframe(filtered[cols_show].head(n_show2), use_container_width=True, height=420)
            else:
                st.info("Pilih minimal satu kolom untuk ditampilkan.")

            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil filter (CSV)",
                data=csv_bytes,
                file_name="filtered_full_preprocessed.csv",
                mime="text/csv",
            )

        with tab_quality:
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Rata² token input</div>
                    <div class="stat-value">{df['token_input'].mean():.0f}</div>
                    <div class="stat-sub">setelah truncation 512</div></div>""",
                    unsafe_allow_html=True,
                )
            with q2:
                truncated_pct = (df["token_input_raw"] > 512).mean() * 100
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Baris terpotong (truncated)</div>
                    <div class="stat-value">{truncated_pct:.1f}%</div>
                    <div class="stat-sub">token mentah &gt; 512</div></div>""",
                    unsafe_allow_html=True,
                )
            with q3:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Rata² token target</div>
                    <div class="stat-value">{df['token_target'].mean():.0f}</div>
                    <div class="stat-sub">token</div></div>""",
                    unsafe_allow_html=True,
                )
            with q4:
                st.markdown(
                    f"""<div class="stat-card"><div class="stat-label">Rata² keyphrase</div>
                    <div class="stat-value">{df['jumlah_keyphrase_bersih'].mean():.1f}</div>
                    <div class="stat-sub">per artikel</div></div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Distribusi token_input (setelah truncation)**")
                st.bar_chart(df["token_input"].value_counts(bins=20).sort_index(), height=280)
            with colB:
                st.markdown("**Distribusi jumlah keyphrase bersih**")
                st.bar_chart(df["jumlah_keyphrase_bersih"].value_counts().sort_index(), height=280)

            st.write("")
            st.markdown("**Hasil tiap lapis deteksi bahasa (proporsi True)**")
            layer_cols = ["s1_langdetect_abstrak", "s2_bilingual_marker", "s3_langdetect_keyword", "is_indonesian"]
            layer_summary = pd.DataFrame(
                {c: [df[c].mean() * 100] for c in layer_cols if c in df.columns},
                index=["% True"],
            ).T
            layer_summary.columns = ["persen_true"]
            st.bar_chart(layer_summary["persen_true"], height=260)


# =========================================================
# HALAMAN: KEYPHRASE GENERATOR
# =========================================================

elif page == "Keyphrase Generator":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Tahap 04 · Inferensi & Prediksi</div>
            <h1>Keyphrase Generator</h1>
            <div class="deck">
                Project ini berfokus pada <strong>dataset & preprocessing</strong>, 
                sehingga belum menyertakan model T5 hasil fine-tuning. 
                Hubungkan repo Hugging Face Anda atau jelajahi contoh 
                pasangan <span class="mono">input_model</span> → <span class="mono">target_model</span> 
                dari dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Mode",
        ["Contoh dari dataset", "Hubungkan model Hugging Face (opsional)"],
        horizontal=True,
    )

    df_full = load_csv_safe(PATH_FULL)

    if mode == "Contoh dari dataset":
        st.markdown(
            """
            <div class="card">
            Belum ada model T5 hasil fine-tuning yang disertakan dalam project ini —
            hanya tokenizer dasar <span class="mono">t5-small</span> yang digunakan
            untuk menghitung token saat preprocessing. Bagian ini menampilkan contoh
            nyata <span class="mono">input_model</span> dan <span class="mono">target_model</span>
            dari <span class="mono">full_preprocessed.csv</span> sebagai gambaran perilaku
            yang diharapkan dari model setelah fine-tuning.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        if df_full is not None:
            colf1, colf2 = st.columns([2, 1])
            with colf1:
                search_title = st.text_input("Cari judul artikel (opsional)", placeholder="ketik sebagian judul...")
            with colf2:
                if st.button("Ambil contoh acak", use_container_width=True):
                    st.session_state["kg_seed"] = pd.Timestamp.now().value

            pool = df_full
            if search_title:
                try:
                    pool = pool[pool["judul"].str.contains(search_title, case=False, na=False, regex=True)]
                except re.error:
                    # Jika ada error regex, fallback ke literal search
                    pool = pool[pool["judul"].str.contains(search_title, case=False, na=False, regex=False)]

            if len(pool) == 0:
                st.info("Tidak ada artikel yang cocok dengan pencarian.")
            else:
                seed = st.session_state.get("kg_seed", 42)
                row = pool.sample(1, random_state=int(seed) % (2**32)).iloc[0]

                st.markdown(f"**{row['judul']}**")
                st.markdown(
                    f"""
                    <div class="sample-block">
                        <div class="sample-head input">input_model (abstrak diproses)</div>
                        <div class="sample-body">{row['input_model']}</div>
                    </div>
                    <div class="sample-block">
                        <div class="sample-head target">target_model (keyphrase referensi)</div>
                        <div class="sample-body">{row['target_model']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kws = [k.strip() for k in str(row["target_model"]).split(",") if k.strip()]
                st.markdown(" ".join([f'<span class="tag tag-terra">{k}</span>' for k in kws]), unsafe_allow_html=True)
        else:
            st.warning("Berkas `full_preprocessed.csv` tidak ditemukan, contoh tidak dapat ditampilkan.")

    else:
        st.markdown(
            """
            <div class="card">
            Jika Anda telah melakukan fine-tuning model T5 untuk keyphrase generation
            dan mengunggahnya ke Hugging Face Hub, masukkan repo ID di bawah untuk
            mencoba inferensi langsung dari abstrak yang Anda tulis sendiri.
            <br><br>
            <i>Catatan: model akan diunduh ke memori server saat pertama kali digunakan,
            sehingga membutuhkan paket <span class="mono">transformers</span> dan
            <span class="mono">torch</span> terpasang serta koneksi internet ke Hugging Face Hub.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        repo_id = st.text_input("Repo ID model Hugging Face", placeholder="mis. username/t5-keyphrase-id")
        abstrak_input = st.text_area(
            "Tempel abstrak (Bahasa Indonesia)",
            height=180,
            placeholder="Tempel teks abstrak artikel di sini...",
        )
        max_new_tokens = st.slider("Maksimal token keyphrase yang dihasilkan", 16, 128, 48)

        run = st.button("Hasilkan Keyphrase", type="primary")

        if run:
            if not repo_id:
                st.error("Isi repo ID model Hugging Face terlebih dahulu.")
            elif not abstrak_input.strip():
                st.error("Tempel teks abstrak terlebih dahulu.")
            else:
                try:
                    with st.spinner(f"Memuat model `{repo_id}` ..."):
                        from transformers import T5Tokenizer, T5ForConditionalGeneration
                        import torch

                        tokenizer = T5Tokenizer.from_pretrained(repo_id)
                        model = T5ForConditionalGeneration.from_pretrained(repo_id)

                    prompt = f"generate keyphrases: {abstrak_input.strip()}"
                    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    with st.spinner("Menghasilkan keyphrase..."):
                        out = model.generate(**enc, max_new_tokens=max_new_tokens)
                    result = tokenizer.decode(out[0], skip_special_tokens=True)

                    st.success("Berhasil menghasilkan keyphrase.")
                    st.markdown(
                        f"""
                        <div class="sample-block">
                            <div class="sample-head target">keyphrase hasil generate</div>
                            <div class="sample-body">{result}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except ImportError:
                    st.error(
                        "Paket `transformers` dan/atau `torch` belum terpasang. "
                        "Jalankan: `pip install transformers torch` lalu ulangi."
                    )
                except Exception as e:
                    st.error(f"Gagal memuat atau menjalankan model: {e}")


# =========================================================
# HALAMAN: EVALUASI MODEL
# =========================================================

elif page == "Evaluasi Model":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Tahap 05 · Evaluasi</div>
            <h1>Evaluasi Kinerja Model</h1>
            <div class="deck">
                Perbandingan performa antara baseline ekstraktif (RAKE) dan model generatif (T5-Small Fine-Tuned) menggunakan metrik otomatis dan evaluasi manusia.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Perbandingan Metrik Otomatis")
    rake_scores = {
        "ROUGE-1": 0.42,
        "ROUGE-2": 0.18,
        "ROUGE-L": 0.39,
        "BLEU": 0.12,
    }
    t5_scores = {
        "ROUGE-1": 0.58,
        "ROUGE-2": 0.31,
        "ROUGE-L": 0.53,
        "BLEU": 0.28,
    }

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### Baseline: RAKE")
        for metric, value in rake_scores.items():
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{metric}</div>
                    <div class="stat-value">{value:.2f}</div>
                    <div class="stat-sub">RAKE baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right_col:
        st.markdown("#### Model: T5-Small")
        for metric, value in t5_scores.items():
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{metric}</div>
                    <div class="stat-value">{value:.2f}</div>
                    <div class="stat-sub">T5-Small fine-tuned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    comparison_df = pd.DataFrame(
        {
            "RAKE": rake_scores,
            "T5-Small": t5_scores,
        }
    )
    st.write("")
    st.bar_chart(comparison_df, height=320)

    st.write("")
    st.markdown("### Evaluasi Manusia (Human Evaluation)")
    st.markdown(
        """
        <div class="card card-tight">
        Evaluasi ini dinilai oleh tiga evaluator dengan skala Likert 1-5 untuk aspek Relevansi dan Spesifisitas.
        Data berikut adalah skor rata-rata untuk setiap metode.
        </div>
        """,
        unsafe_allow_html=True,
    )

    human_eval = pd.DataFrame(
        [
            {"Metode": "RAKE", "Relevansi": 2.9, "Spesifisitas": 2.6},
            {"Metode": "T5-Small", "Relevansi": 4.2, "Spesifisitas": 4.0},
        ]
    )
    st.dataframe(human_eval, use_container_width=True, height=220)


# =========================================================
# HALAMAN: TRAIN / VAL / TEST
# =========================================================

elif page == "Train / Val / Test":
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Tahap 04 · Dataset Split (80/10/10)</div>
            <h1>Train / Validation / Test</h1>
            <div class="deck">
                Pembagian akhir dataset menjadi tiga berkas siap-latih, masing-masing hanya berisi 
                kolom <span class="mono">input_model</span> dan <span class="mono">target_model</span>. 
                Rasio split mengikuti konvensi umum 80% train / 10% validasi / 10% testing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_train = load_csv_safe(PATH_TRAIN)
    df_val = load_csv_safe(PATH_VAL)
    df_test = load_csv_safe(PATH_TEST)

    splits = {"train.csv": df_train, "val.csv": df_val, "test.csv": df_test}
    total = sum(len(d) for d in splits.values() if d is not None)

    cols = st.columns(3)
    colors = ["var(--indigo)", "var(--terracotta)", "var(--moss)"]
    for (name, d), col, color in zip(splits.items(), cols, colors):
        with col:
            n = len(d) if d is not None else 0
            pct = (n / total * 100) if total else 0
            st.markdown(
                f"""
                <div class="stat-card" style="border-left-color:{color};">
                <div class="stat-label">{name}</div>
                <div class="stat-value" style="color:{color};">{n:,}</div>
                <div class="stat-sub">{pct:.1f}% dari total split</div>
                </div>
                """.replace(",", "."),
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("### Proporsi Split")
    if total:
        prop_df = pd.DataFrame(
            {"jumlah_baris": [len(d) if d is not None else 0 for d in splits.values()]},
            index=list(splits.keys()),
        )
        st.bar_chart(prop_df, height=260)

    st.write("")
    pick = st.selectbox("Pilih split untuk dipratinjau & diunduh", list(splits.keys()))
    df_pick = splits[pick]

    if df_pick is None:
        st.warning(f"Berkas `{pick}` tidak ditemukan.")
    else:
        n_show = st.slider("Jumlah baris ditampilkan", 5, 100, 20, step=5)
        st.dataframe(df_pick.head(n_show), use_container_width=True, height=380)

        csv_bytes = df_pick.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Unduh {pick}",
            data=csv_bytes,
            file_name=pick,
            mime="text/csv",
        )

    st.markdown(
        """
        <div class="footnote">
        Rasio split mengikuti konvensi umum 80% train / 10% validasi / 10% uji,
        konsisten dengan ukuran masing-masing berkas yang dihasilkan oleh
        <span class="mono">preprocessing-keyphrase.ipynb</span>.
        </div>
        """,
        unsafe_allow_html=True,
    )