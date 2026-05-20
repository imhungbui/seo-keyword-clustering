import os
import re
import io
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEO Keyword Clustering Tool",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0d0f;
    color: #e8e6e0;
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    color: #f0ede6;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}
.hero p {
    font-size: 1.05rem;
    color: #7a7870;
    font-weight: 300;
}
.accent { color: #c8f564; }

/* Cards */
.card {
    background: #16161a;
    border: 1px solid #2a2a2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #c8f564;
    margin-bottom: 0.8rem;
}

/* Textarea */
textarea {
    background: #0d0d0f !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 8px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
textarea:focus {
    border-color: #c8f564 !important;
    box-shadow: 0 0 0 2px rgba(200, 245, 100, 0.1) !important;
}

/* Slider */
.stSlider > div { padding-top: 0.3rem; }

/* Button */
.stButton > button {
    background: #c8f564 !important;
    color: #0d0d0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #d8ff74 !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(200, 245, 100, 0.25) !important;
}

/* Download button */
.stDownloadButton > button {
    background: #1e1e24 !important;
    color: #c8f564 !important;
    border: 1px solid #c8f564 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 0.75rem 2rem !important;
}
.stDownloadButton > button:hover {
    background: rgba(200, 245, 100, 0.08) !important;
}

/* Stats */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
}
.stat-box {
    flex: 1;
    background: #16161a;
    border: 1px solid #2a2a2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #c8f564;
}
.stat-label {
    font-size: 0.75rem;
    color: #7a7870;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Table */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* Divider */
hr { border-color: #2a2a2e; }

/* Info box */
.info-box {
    background: rgba(200, 245, 100, 0.06);
    border: 1px solid rgba(200, 245, 100, 0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #b8d458;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helper functions (giữ nguyên logic gốc) ──────────────────────────────────

def parse_volume(vol_str):
    if not vol_str:
        return 0
    vol_clean = re.sub(r'[^\d]', '', str(vol_str).strip())
    return int(vol_clean) if vol_clean else 0

def split_line_smart(line):
    line = line.strip()
    if not line:
        return None, None
    if '\t' in line:
        parts = line.split('\t')
        keyword = parts[0].strip()
        volume = parse_volume(parts[-1]) if len(parts) > 1 else 0
        return keyword, volume
    if ';' in line:
        parts = line.rsplit(';', 1)
        return parts[0].strip(), parse_volume(parts[1]) if len(parts) > 1 else 0
    match = re.search(r'^(.*?)[\s,]+([\d.,]+)\s*$', line)
    if match:
        keyword = match.group(1).strip().rstrip(',').strip()
        volume = parse_volume(match.group(2))
        return keyword, volume
    return line, 0

def process_input(raw_text):
    parsed_data = []
    for line in raw_text.strip().split('\n'):
        keyword, volume = split_line_smart(line)
        if keyword:
            parsed_data.append({'keyword': keyword, 'volume': volume})
    return pd.DataFrame(parsed_data)

@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('intfloat/multilingual-e5-base')

def run_clustering(df, similarity_threshold=0.88):
    from sklearn.cluster import AgglomerativeClustering
    model = load_model()
    queries = ["query: " + kw for kw in df['keyword'].tolist()]
    embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    distance_threshold = 1 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='complete',
        distance_threshold=distance_threshold
    )
    df['cluster_id'] = clustering.fit_predict(embeddings)
    return df

def build_excel(df_clustered):
    final_data = []
    for cluster_id, group in df_clustered.groupby('cluster_id'):
        group = group.sort_values(by='volume', ascending=False)
        keywords_list = group.to_dict('records')
        primary = keywords_list[0]
        for kw in keywords_list:
            final_data.append({
                'Main Topic': primary['keyword'],
                'Cluster Volume': int(group['volume'].sum()),
                'Keywords in Cluster': len(keywords_list),
                'Keyword': kw['keyword'],
                'Volume': kw['volume'],
            })
    df_export = pd.DataFrame(final_data)
    df_export = df_export.sort_values(
        by=['Cluster Volume', 'Volume'], ascending=[False, False]
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Clusters')
        ws = writer.sheets['Clusters']
        # Auto-width columns
        for col in ws.columns:
            max_len = max((len(str(cell.value or '')) for cell in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    buf.seek(0)
    return buf, df_export

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <h1>🔍 Keyword <span class="accent">Clustering</span></h1>
  <p>Group keywords by semantic meaning — powered by multilingual AI (EN + VI)</p>
</div>
""", unsafe_allow_html=True)

# Input card
st.markdown('<div class="card"><div class="card-title">📋 Keyword Input</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">✦ Paste keywords + volume from Excel, Ahrefs, SEMrush, Google Sheets…<br>Supports: tab, comma, semicolon, or space separated.</div>', unsafe_allow_html=True)

raw_text = st.text_area(
    label="keywords",
    label_visibility="collapsed",
    placeholder="Example:\nbuy running shoes\t4400\nbest running shoes for men\t2900\nrunning shoes sale\t1600",
    height=220,
)
st.markdown('</div>', unsafe_allow_html=True)

# Settings card
st.markdown('<div class="card"><div class="card-title">⚙️ Settings</div>', unsafe_allow_html=True)
similarity = st.slider(
    "Similarity threshold",
    min_value=0.70,
    max_value=0.98,
    value=0.88,
    step=0.01,
    help="Higher = smaller, tighter clusters. Lower = broader groups."
)
st.caption(f"Current: **{similarity}** — {'Tight clusters' if similarity >= 0.90 else 'Balanced' if similarity >= 0.82 else 'Broad clusters'}")
st.markdown('</div>', unsafe_allow_html=True)

# Run button
run = st.button("⚡ Run Clustering")

if run:
    if not raw_text.strip():
        st.error("Please paste some keywords first!")
    else:
        df = process_input(raw_text)
        if df.empty:
            st.error("No valid keywords found. Check your input format.")
        else:
            with st.spinner(f"Loading AI model & clustering {len(df)} keywords…"):
                df_clustered = run_clustering(df.copy(), similarity_threshold=similarity)
                excel_buf, df_export = build_excel(df_clustered)

            num_clusters = df_export['Main Topic'].nunique()

            # Stats
            st.markdown(f"""
            <div class="stats-row">
              <div class="stat-box">
                <div class="stat-num">{len(df)}</div>
                <div class="stat-label">Keywords</div>
              </div>
              <div class="stat-box">
                <div class="stat-num">{num_clusters}</div>
                <div class="stat-label">Clusters</div>
              </div>
              <div class="stat-box">
                <div class="stat-num">{round(len(df)/num_clusters,1)}</div>
                <div class="stat-label">Avg per cluster</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Preview table
            st.markdown('<div class="card-title" style="margin-top:1rem">📊 Preview</div>', unsafe_allow_html=True)
            preview = df_export.head(30)
            st.dataframe(preview, use_container_width=True, hide_index=True)

            # Download
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download Full Excel File",
                data=excel_buf,
                file_name=f"Keyword_Clusters_{now}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
