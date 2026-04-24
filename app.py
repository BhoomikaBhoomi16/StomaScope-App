import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import sqlite3
import datetime
import pandas as pd

st.set_page_config(page_title="StomaScope", page_icon="🌿", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark Botanical Luxury Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important;
    color: #e8f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1f0d 0%, #050a05 60%, #0a0f0a 100%) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070d07, #0a0f0a) !important;
    border-right: 1px solid rgba(74,222,128,0.1) !important;
}
[data-testid="stSidebar"] * { color: #c4dcc4 !important; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem; font-weight: 900;
    background: linear-gradient(135deg, #4ade80 0%, #86efac 50%, #bbf7d0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center; letter-spacing: -2px; margin-bottom: 0.3rem;
}
.hero-sub {
    text-align: center; font-size: 0.85rem; letter-spacing: 4px;
    text-transform: uppercase; color: #6b9e6b; margin-bottom: 0.5rem;
}
.hero-line {
    width: 80px; height: 2px; margin: 1rem auto 2rem;
    background: linear-gradient(90deg, transparent, #4ade80, transparent);
}
.sec-head {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; color: #86efac; margin: 1.8rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(74,222,128,0.25), transparent);
}
.card {
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(74,222,128,0.15); border-radius: 18px;
    padding: 1.6rem; position: relative; overflow: hidden; height: 100%;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 18px 18px 0 0;
}
.card-g::before { background: linear-gradient(90deg,#4ade80,#22c55e); }
.card-o::before { background: linear-gradient(90deg,#fb923c,#f97316); }
.card-b::before { background: linear-gradient(90deg,#38bdf8,#0ea5e9); }
.c-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.c-lbl { font-size: 0.68rem; letter-spacing: 3px; text-transform: uppercase; color: #6b9e6b; margin-bottom: 0.3rem; }
.c-val { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #e8f0e8; margin-bottom: 0.6rem; }
.c-txt { font-size: 0.85rem; color: #9cb89c; line-height: 1.6; }
.conf-wrap { margin-top: 0.8rem; }
.conf-row { display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b9e6b; margin-bottom: 0.3rem; }
.conf-bg { background: rgba(255,255,255,0.06); border-radius: 999px; height: 7px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#4ade80,#86efac); }
.badge {
    display: inline-block; padding: 0.25rem 0.9rem; border-radius: 999px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 0.6rem;
}
.sev-l { background:rgba(74,222,128,0.15); color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.sev-m { background:rgba(251,146,60,0.15); color:#fb923c; border:1px solid rgba(251,146,60,0.3); }
.sev-h { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.t3row { display:flex; align-items:center; gap:10px; margin-bottom:0.8rem; }
.t3nm  { font-size:0.8rem; color:#c4dcc4; width:210px; flex-shrink:0; }
.t3bg  { flex:1; background:rgba(255,255,255,0.06); border-radius:999px; height:6px; overflow:hidden; }
.t3bar { height:100%; border-radius:999px; }
.t3pc  { font-size:0.75rem; color:#6b9e6b; width:42px; text-align:right; flex-shrink:0; }
.stat-card {
    background: linear-gradient(145deg,rgba(74,222,128,0.06),rgba(74,222,128,0.01));
    border:1px solid rgba(74,222,128,0.15); border-radius:14px;
    padding:0.9rem 1rem; text-align:center; margin-bottom:0.5rem;
}
.stat-n { font-family:'Playfair Display',serif; font-size:2rem; color:#4ade80; line-height:1; }
.stat-l { font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; color:#6b9e6b; margin-top:0.2rem; }
.info-box {
    background:rgba(74,222,128,0.04); border:1px solid rgba(74,222,128,0.1);
    border-radius:11px; padding:0.9rem 1.1rem; font-size:0.8rem;
    color:#6b9e6b; line-height:1.7; margin-top:1rem;
}
[data-testid="stTabs"] button { color:#6b9e6b !important; font-size:0.82rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#4ade80 !important; border-bottom-color:#4ade80 !important; }
[data-testid="stButton"] button {
    background:rgba(74,222,128,0.08) !important; border:1px solid rgba(74,222,128,0.25) !important;
    color:#4ade80 !important; border-radius:9px !important;
}
[data-testid="stDownloadButton"] button {
    background:linear-gradient(135deg,#4ade80,#22c55e) !important;
    color:#0a0f0a !important; border:none !important; font-weight:600 !important; border-radius:9px !important;
}
[data-testid="stFileUploader"] {
    background:rgba(74,222,128,0.03) !important;
    border:1.5px dashed rgba(74,222,128,0.25) !important; border-radius:16px !important;
}
.footer { text-align:center; padding:2rem 0 1rem; color:#3a5a3a; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:rgba(74,222,128,0.2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "stomascope_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL)""")
    conn.commit(); conn.close()

def log_prediction(image_name, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?)",
        (image_name, prediction, round(float(confidence), 2),
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

def fetch_logs(limit=100):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id,image_name,prediction,confidence,timestamp FROM predictions ORDER BY id DESC LIMIT ?",
        conn, params=(limit,))
    conn.close(); return df

def fetch_stats():
    conn = sqlite3.connect(DB_PATH)
    total    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    top      = conn.execute("SELECT prediction FROM predictions GROUP BY prediction ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
    conn.close()
    return total, round(avg_conf, 1), (top[0] if top else "—")

def clear_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions"); conn.commit(); conn.close()

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Stats + Log
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Prediction Log")
    st.markdown(f"<small style='color:#3a5a3a'>SQLite · {DB_PATH}</small>", unsafe_allow_html=True)
    st.divider()

    total, avg_conf, top_disease = fetch_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-n">{total}</div><div class="stat-l">Scans</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-n">{avg_conf}%</div><div class="stat-l">Avg Conf</div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="stat-card" style="text-align:left;">
        <div class="stat-l">Most Detected</div>
        <div style="font-size:0.88rem;color:#86efac;margin-top:0.2rem;font-weight:500;">
            {top_disease.replace('_',' ') if top_disease != '—' else '—'}
        </div></div>""", unsafe_allow_html=True)

    st.divider()
    logs_df = fetch_logs()
    if logs_df.empty:
        st.caption("No predictions yet.")
    else:
        st.dataframe(
            logs_df.rename(columns={"id":"ID","image_name":"Image","prediction":"Disease","confidence":"Conf%","timestamp":"Time"}),
            use_container_width=True, hide_index=True, height=200)
        st.download_button("⬇️ Export CSV", data=logs_df.to_csv(index=False).encode(),
            file_name="stomascope_predictions.csv", mime="text/csv", use_container_width=True)
        if st.button("🗑️ Clear Logs", use_container_width=True):
            clear_logs(); st.success("Cleared!"); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 StomaScope</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Crop Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-line"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  ← ORIGINAL WORKING CODE, COMPLETELY UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam(img_array, model):
    """Fixed Grad-CAM for MobileNetV2 transfer learning"""
    # Get the base MobileNetV2 model
    base_model = model.layers[1]

    with tf.GradientTape() as tape:
        # Forward pass
        conv_output = base_model(img_array, training=False)
        tape.watch(conv_output)

        # Pass through remaining layers (GlobalAveragePooling + Dense)
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        preds = model.layers[-1](x)   # final Dense layer

        # Get the predicted class score
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_score, conv_output)

    # ── Debug info shown in sidebar ──
    with st.sidebar:
        st.markdown("**Grad-CAM Debug**")
        st.write("grads is None:", grads is None)
        if grads is not None:
            g_max = float(tf.reduce_max(tf.abs(grads)).numpy())
            st.write(f"grad max abs: {g_max:.6f}")
            st.write(f"conv_output shape: {conv_output.shape}")

    if grads is None:
        st.warning("⚠️ Could not generate Grad-CAM heatmap")
        return np.zeros((7, 7)), int(pred_index)

    # Generate heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), int(pred_index)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_severity(conf, pred_class):
    if "healthy" in pred_class.lower(): return "LOW", "sev-l"
    if conf >= 80: return "HIGH", "sev-h"
    elif conf >= 55: return "MEDIUM", "sev-m"
    else: return "LOW", "sev-l"

def get_cause(pred_class):
    if "Late_blight"  in pred_class: return "Fungal infection by <em>Phytophthora infestans</em>. Spreads in cool, wet conditions."
    if "Early_blight" in pred_class: return "Caused by <em>Alternaria solani</em>. Spreads in warm, humid weather."
    if "healthy"      in pred_class.lower(): return "No disease detected. Leaf appears healthy and vigorous."
    return "Fungal or bacterial pathogen common in this crop. Spreads via spores or water splash."

def get_treatment(pred_class):
    if "Late_blight"  in pred_class: return "Apply Mancozeb or Chlorothalonil. Remove infected leaves immediately. Avoid overhead irrigation."
    if "Early_blight" in pred_class: return "Use copper-based fungicide or Azoxystrobin. Improve canopy airflow. Remove lower leaves."
    if "healthy"      in pred_class.lower(): return "Continue good agricultural practices. Maintain soil health and routine inspection."
    return "Use appropriate fungicide. Improve air circulation and remove affected plant matter."


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:1.4rem;color:#86efac;margin-bottom:0.3rem;">Upload Leaf Image</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:0.8rem;color:#6b9e6b;margin-bottom:0.6rem;letter-spacing:1px;">JPG · JPEG · PNG — clear, well-lit photos give best results</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:

    # ── Pre-process (same as original) ────────────────────────────────────────
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── Predict (same as original) ────────────────────────────────────────────
    preds      = model(img_array, training=False)
    pred_idx   = np.argmax(preds[0])
    confidence = float(preds[0][pred_idx]) * 100
    pred_class = class_names[pred_idx]

    # top-3
    top3_idx = np.argsort(preds[0].numpy())[::-1][:3]
    top3     = [(class_names[i], float(preds[0][i]) * 100) for i in top3_idx]

    sev_label, sev_cls = get_severity(confidence, pred_class)

    # ── Log to DB ─────────────────────────────────────────────────────────────
    log_prediction(uploaded_file.name, pred_class, confidence)
    st.success(f"✅ Prediction logged to `{DB_PATH}`")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🧬  Diagnosis", "📊  Top Predictions", "🔍  Grad-CAM"])

    # ── TAB 1 : Diagnosis ─────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="sec-head">Diagnosis Report</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="card card-g">
                <div class="c-icon">🦠</div>
                <div class="c-lbl">Predicted Disease</div>
                <div class="c-val">{pred_class.replace('_',' ')}</div>
                <div class="conf-wrap">
                    <div class="conf-row"><span>Confidence</span><span>{confidence:.1f}%</span></div>
                    <div class="conf-bg"><div class="conf-fill" style="width:{confidence:.1f}%"></div></div>
                </div>
                <span class="badge {sev_cls}">⚠ {sev_label} Severity</span>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card card-o">
                <div class="c-icon">🔬</div>
                <div class="c-lbl">Root Cause</div>
                <div class="c-val">Pathogen Analysis</div>
                <div class="c-txt">{get_cause(pred_class)}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card card-b">
                <div class="c-icon">💊</div>
                <div class="c-lbl">Recommended Treatment</div>
                <div class="c-val">Action Plan</div>
                <div class="c-txt">{get_treatment(pred_class)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:1.8rem;">Uploaded Image</div>', unsafe_allow_html=True)
        ci, _ = st.columns([1, 2])
        with ci:
            st.image((img_array[0] * 255).astype(np.uint8),
                     caption=f"📁 {uploaded_file.name}", width=260)

    # ── TAB 2 : Top-3 ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-head">Top 3 Predictions</div>', unsafe_allow_html=True)
        colors  = ["#4ade80", "#86efac", "#bbf7d0"]
        medals  = ["🥇", "🥈", "🥉"]
        for i, (name, prob) in enumerate(top3):
            st.markdown(f"""
            <div class="t3row">
                <div class="t3nm">{medals[i]} {name.replace('_',' ')}</div>
                <div class="t3bg"><div class="t3bar" style="width:{prob:.1f}%;background:{colors[i]};"></div></div>
                <div class="t3pc">{prob:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            💡 A large gap between 1st and 2nd place means a confident, reliable prediction.
            If values are close, try uploading a clearer image.
        </div>""", unsafe_allow_html=True)

    # ── TAB 3 : Grad-CAM ──────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-head">Model Explanation — Grad-CAM</div>', unsafe_allow_html=True)

        with st.spinner("Generating heatmap..."):
            # ── GRAD-CAM: identical rendering to original working code ──
            heatmap, _ = get_gradcam(img_array, model)

            original  = (img_array[0] * 255).astype(np.uint8)
            h         = cv2.resize(heatmap, (224, 224))
            h         = np.uint8(255 * h)
            h_color   = cv2.applyColorMap(h, cv2.COLORMAP_JET)
            overlay   = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
            overlay   = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        ca, cb = st.columns(2, gap="large")
        with ca:
            st.image(original, caption="📷 Original Image", use_container_width=True)
        with cb:
            st.image(overlay,  caption="🌡️ Grad-CAM — Red = High Model Attention", use_container_width=True)

        st.markdown("""<div class="info-box">
            🔴 <strong style="color:#c4dcc4">Red / Yellow</strong> — Regions the model focused on most.<br>
            🔵 <strong style="color:#c4dcc4">Blue / Green</strong> — Low-attention background areas.<br>
            Gradient-weighted Class Activation Mapping (Grad-CAM) makes the AI's decision transparent and verifiable.
        </div>""", unsafe_allow_html=True)

        st.caption("Red and yellow regions show where the model paid most attention while making its prediction.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="footer">
    Made with ❤️ for Farmers &nbsp;·&nbsp; AI-Powered Crop Disease Detection
    &nbsp;·&nbsp; Bhoomika D, MCA — Soundarya Institute of Management & Science
</div>""", unsafe_allow_html=True)
