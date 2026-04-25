import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import sqlite3
import datetime
import pandas as pd
import io
import base64
import tempfile
import os

st.set_page_config(page_title="StomaScope", page_icon="🌿", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important; color: #e8f0e8 !important;
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
    font-family: 'Playfair Display', serif; font-size: 4rem; font-weight: 900;
    background: linear-gradient(135deg, #4ade80 0%, #86efac 50%, #bbf7d0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    text-align: center; letter-spacing: -2px; margin-bottom: 0.3rem;
}
.hero-sub { text-align:center; font-size:0.85rem; letter-spacing:4px; text-transform:uppercase; color:#6b9e6b; }
.hero-line { width:80px; height:2px; margin:1rem auto 2rem; background:linear-gradient(90deg,transparent,#4ade80,transparent); }
.sec-head {
    font-family:'Playfair Display',serif; font-size:1.4rem; color:#86efac;
    margin:1.8rem 0 1rem; display:flex; align-items:center; gap:10px;
}
.sec-head::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(74,222,128,0.25),transparent); }
.card {
    background:linear-gradient(145deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
    border:1px solid rgba(74,222,128,0.15); border-radius:18px;
    padding:1.6rem; position:relative; overflow:hidden; height:100%;
    transition:all 0.25s ease;
}
.card:hover { transform:translateY(-4px); box-shadow:0 8px 30px rgba(74,222,128,0.1); border-color:rgba(74,222,128,0.35) !important; }
.card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:18px 18px 0 0; }
.card-g::before { background:linear-gradient(90deg,#4ade80,#22c55e); }
.card-o::before { background:linear-gradient(90deg,#fb923c,#f97316); }
.card-b::before { background:linear-gradient(90deg,#38bdf8,#0ea5e9); }
.card-p::before { background:linear-gradient(90deg,#a78bfa,#8b5cf6); }
.c-icon { font-size:1.8rem; margin-bottom:0.6rem; }
.c-lbl { font-size:0.68rem; letter-spacing:3px; text-transform:uppercase; color:#6b9e6b; margin-bottom:0.3rem; }
.c-val { font-family:'Playfair Display',serif; font-size:1.3rem; color:#e8f0e8; margin-bottom:0.6rem; }
.c-txt { font-size:0.85rem; color:#9cb89c; line-height:1.6; }
.conf-wrap { margin-top:0.8rem; }
.conf-row { display:flex; justify-content:space-between; font-size:0.75rem; color:#6b9e6b; margin-bottom:0.3rem; }
.conf-bg { background:rgba(255,255,255,0.06); border-radius:999px; height:7px; overflow:hidden; }
.conf-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#4ade80,#86efac); }
.badge { display:inline-block; padding:0.25rem 0.9rem; border-radius:999px; font-size:0.68rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin-top:0.6rem; }
.sev-l { background:rgba(74,222,128,0.15); color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.sev-m { background:rgba(251,146,60,0.15); color:#fb923c; border:1px solid rgba(251,146,60,0.3); }
.sev-h { background:rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.t3row { display:flex; align-items:center; gap:10px; margin-bottom:0.8rem; }
.t3nm  { font-size:0.8rem; color:#c4dcc4; width:210px; flex-shrink:0; }
.t3bg  { flex:1; background:rgba(255,255,255,0.06); border-radius:999px; height:6px; overflow:hidden; }
.t3bar { height:100%; border-radius:999px; }
.t3pc  { font-size:0.75rem; color:#6b9e6b; width:42px; text-align:right; flex-shrink:0; }
.stat-card { background:linear-gradient(145deg,rgba(74,222,128,0.06),rgba(74,222,128,0.01)); border:1px solid rgba(74,222,128,0.15); border-radius:14px; padding:0.9rem 1rem; text-align:center; margin-bottom:0.5rem; }
.stat-n { font-family:'Playfair Display',serif; font-size:2rem; color:#4ade80; line-height:1; text-shadow:0 0 20px rgba(74,222,128,0.4); }
.stat-l { font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; color:#6b9e6b; margin-top:0.2rem; }
.info-box { background:rgba(74,222,128,0.04); border:1px solid rgba(74,222,128,0.1); border-radius:11px; padding:0.9rem 1.1rem; font-size:0.8rem; color:#6b9e6b; line-height:1.7; margin-top:1rem; }
.warn-box { background:rgba(251,146,60,0.07); border:1px solid rgba(251,146,60,0.25); border-radius:11px; padding:0.9rem 1.1rem; font-size:0.85rem; color:#fb923c; line-height:1.6; margin:1rem 0; }
.disease-info { background:rgba(74,222,128,0.03); border:1px solid rgba(74,222,128,0.1); border-radius:14px; padding:1.2rem 1.4rem; margin-top:1rem; }
.di-title { font-family:'Playfair Display',serif; font-size:1.1rem; color:#86efac; margin-bottom:0.6rem; }
.di-row { display:flex; gap:8px; margin-bottom:0.4rem; font-size:0.82rem; color:#9cb89c; }
.di-tag { background:rgba(74,222,128,0.1); border-radius:6px; padding:0.15rem 0.6rem; font-size:0.72rem; color:#4ade80; margin-right:4px; display:inline-block; margin-bottom:4px; }
.batch-row { background:rgba(255,255,255,0.02); border:1px solid rgba(74,222,128,0.1); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:12px; }
[data-testid="stTabs"] button { color:#6b9e6b !important; font-size:0.82rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#4ade80 !important; border-bottom-color:#4ade80 !important; }
[data-testid="stButton"] button { background:rgba(74,222,128,0.08) !important; border:1px solid rgba(74,222,128,0.25) !important; color:#4ade80 !important; border-radius:9px !important; }
[data-testid="stDownloadButton"] button { background:linear-gradient(135deg,#4ade80,#22c55e) !important; color:#0a0f0a !important; border:none !important; font-weight:600 !important; border-radius:9px !important; }
[data-testid="stFileUploader"] { background:rgba(74,222,128,0.03) !important; border-radius:16px !important; }
[data-testid="stAlert"] { background:rgba(74,222,128,0.07) !important; border:1px solid rgba(74,222,128,0.2) !important; border-radius:12px !important; }
@keyframes borderPulse {
    0%,100% { border-color:rgba(74,222,128,0.25); box-shadow:none; }
    50%      { border-color:rgba(74,222,128,0.6); box-shadow:0 0 20px rgba(74,222,128,0.12); }
}
[data-testid="stFileUploader"] { animation:borderPulse 3s ease-in-out infinite !important; }
.footer { text-align:center; padding:2rem 0 1rem; color:#3a5a3a; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:rgba(74,222,128,0.2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "stomascope_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL, prediction TEXT NOT NULL,
        confidence REAL NOT NULL, timestamp TEXT NOT NULL)""")
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
# SIDEBAR
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
# HERO
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
# GRAD-CAM  ← ORIGINAL WORKING CODE, UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam(img_array, model):
    base_model = model.layers[1]
    with tf.GradientTape() as tape:
        conv_output = base_model(img_array, training=False)
        tape.watch(conv_output)
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        preds = model.layers[-1](x)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]
    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        return np.zeros((7, 7)), int(pred_index)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    hmap = heatmap.numpy()
    low  = np.percentile(hmap, 10)
    high = np.percentile(hmap, 100)
    if high - low < 1e-8: high = hmap.max() + 1e-8
    hmap = np.clip((hmap - low) / (high - low), 0, 1)
    return hmap, int(pred_index)


def render_gradcam(img_array):
    heatmap, _ = get_gradcam(img_array, model)
    original   = (img_array[0] * 255).astype(np.uint8)
    h          = cv2.resize(heatmap.astype(np.float32), (224, 224))
    h          = np.uint8(255 * h)
    h_color    = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    orig_bgr   = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay    = cv2.addWeighted(orig_bgr, 0.6, h_color, 0.4, 0)
    overlay    = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return original, overlay



# ─────────────────────────────────────────────────────────────────────────────
# LEAF ISOLATION — Remove background from camera images
# ─────────────────────────────────────────────────────────────────────────────
def isolate_leaf(pil_img):
    """
    Isolates the leaf from background using HSV green/yellow masking + GrabCut.
    Returns a PIL Image with background replaced by black (same as training data).
    """
    img_rgb  = np.array(pil_img.convert("RGB"))
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w     = img_bgr.shape[:2]

    # ── Step 1: HSV mask — keep green/yellow-green leaf tones ──
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Green range
    mask_g = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
    # Yellow-brown range (diseased areas)
    mask_y = cv2.inRange(hsv, np.array([10, 20, 40]), np.array([30, 255, 255]))
    mask   = cv2.bitwise_or(mask_g, mask_y)

    # ── Step 2: Morphology — fill holes, remove noise ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    # ── Step 3: Find largest contour (the leaf) ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)

        # ── Step 4: GrabCut refinement using the contour bounding box ──
        x, y, bw, bh = cv2.boundingRect(largest)
        # Add padding
        pad = 10
        x, y = max(0, x-pad), max(0, y-pad)
        bw, bh = min(w-x, bw+2*pad), min(h-y, bh+2*pad)

        gc_mask  = np.where(clean_mask > 0,
                            cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, (x, y, bw, bh),
                        bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            final_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except Exception:
            final_mask = clean_mask
    else:
        # Fallback: use centre crop if no green contour found
        final_mask = np.zeros((h, w), np.uint8)
        cx, cy = w//2, h//2
        r = min(cx, cy) - 20
        cv2.circle(final_mask, (cx, cy), r, 255, -1)

    # ── Step 5: Apply mask — black background ──
    result = img_bgr.copy()
    result[final_mask == 0] = [0, 0, 0]
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), final_mask

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_severity(conf, pred_class):
    if "healthy" in pred_class.lower(): return "LOW", "sev-l"
    if conf >= 80: return "HIGH", "sev-h"
    elif conf >= 55: return "MEDIUM", "sev-m"
    return "LOW", "sev-l"

def get_cause(pred_class):
    if "Late_blight"  in pred_class: return "Fungal infection by Phytophthora infestans. Spreads in cool, wet conditions."
    if "Early_blight" in pred_class: return "Caused by Alternaria solani fungus. Spreads in warm, humid weather."
    if "healthy"      in pred_class.lower(): return "No disease detected. Leaf appears healthy."
    return "Fungal or bacterial pathogen common in this crop. Spreads via spores or water splash."

def get_treatment(pred_class):
    if "Late_blight"  in pred_class: return "Apply Mancozeb or Chlorothalonil. Remove infected leaves immediately. Avoid overhead irrigation."
    if "Early_blight" in pred_class: return "Use copper-based fungicide or Azoxystrobin. Improve canopy airflow. Remove lower leaves."
    if "healthy"      in pred_class.lower(): return "Continue good agricultural practices. Maintain soil health."
    return "Use appropriate fungicide. Improve air circulation and remove affected plant matter."

# ── Disease info database ──────────────────────────────────────────────────
DISEASE_INFO = {
    "Late_blight": {
        "full_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "crops": ["Tomato", "Potato"],
        "symptoms": "Dark brown water-soaked lesions on leaves. White fuzzy growth on underside. Rapid wilting and plant death.",
        "conditions": "Cool (10–20°C), wet, humid weather. High rainfall or heavy dew.",
        "spread": "Wind-borne spores, infected seed tubers, rain splash.",
        "prevention": "Avoid overhead irrigation. Use certified disease-free seeds. Crop rotation.",
        "severity": "🔴 Very High — Can destroy entire crop within days if untreated."
    },
    "Early_blight": {
        "full_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "crops": ["Tomato", "Potato"],
        "symptoms": "Dark brown circular spots with concentric rings (target-board pattern). Yellow halo around lesions.",
        "conditions": "Warm (24–29°C), humid conditions. Wet foliage for extended periods.",
        "spread": "Wind, rain, infected plant debris in soil.",
        "prevention": "Remove infected leaves. Mulching. Avoid wetting foliage.",
        "severity": "🟠 Moderate — Significant yield loss if not managed early."
    },
}

def get_disease_info(pred_class):
    for key, info in DISEASE_INFO.items():
        if key in pred_class:
            return info
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR (using only stdlib — no external PDF library needed)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(image_pil, pred_class, confidence, cause, treatment,
                         sev_label, top3, timestamp, image_name):
    """Generate a styled HTML report and convert it to PDF bytes via weasyprint,
    falling back to a plain-text PDF if weasyprint is unavailable."""
    # Build HTML content
    top3_rows = "".join([
        f"<tr><td>{i+1}</td><td>{n.replace('_',' ')}</td><td>{p:.1f}%</td></tr>"
        for i, (n, p) in enumerate(top3)
    ])

    # Convert PIL image to base64 for embedding
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    sev_color = {"LOW": "#4ade80", "MEDIUM": "#fb923c", "HIGH": "#ef4444"}.get(sev_label, "#4ade80")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; background:#fff; color:#1a1a1a; margin:0; padding:0; }}
  .header {{ background:linear-gradient(135deg,#1b5e20,#2e7d32); color:white; padding:30px 40px; }}
  .header h1 {{ margin:0; font-size:2rem; }}
  .header p  {{ margin:4px 0 0; opacity:0.8; font-size:0.9rem; letter-spacing:2px; }}
  .body {{ padding:30px 40px; }}
  .section {{ margin-bottom:24px; }}
  .section h2 {{ font-size:1rem; color:#2e7d32; border-bottom:2px solid #e8f5e9; padding-bottom:6px; text-transform:uppercase; letter-spacing:1px; }}
  .row {{ display:flex; gap:20px; margin-top:10px; }}
  .chip {{ background:#e8f5e9; border-radius:6px; padding:6px 14px; font-size:0.85rem; color:#1b5e20; font-weight:bold; }}
  .sev  {{ background:{sev_color}22; border:1px solid {sev_color}; border-radius:20px; padding:4px 14px; font-size:0.8rem; color:{sev_color}; font-weight:bold; display:inline-block; }}
  .conf-bar-bg {{ background:#e0e0e0; border-radius:999px; height:10px; width:100%; margin-top:6px; }}
  .conf-bar-fill {{ background:#4caf50; border-radius:999px; height:10px; width:{confidence:.1f}%; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  th {{ background:#e8f5e9; color:#1b5e20; padding:8px 12px; text-align:left; }}
  td {{ padding:7px 12px; border-bottom:1px solid #f0f0f0; }}
  .info-label {{ font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }}
  .info-val   {{ font-size:0.95rem; color:#1a1a1a; margin-bottom:12px; }}
  .footer {{ background:#f5f5f5; padding:14px 40px; font-size:0.75rem; color:#888; text-align:center; }}
  img {{ border-radius:10px; border:1px solid #e0e0e0; }}
</style>
</head>
<body>
<div class="header">
  <h1>🌿 StomaScope — Disease Report</h1>
  <p>AI-POWERED CROP DISEASE DETECTION SYSTEM</p>
</div>
<div class="body">

  <div class="section">
    <h2>Scan Information</h2>
    <div class="row">
      <div><div class="info-label">Image File</div><div class="info-val">{image_name}</div></div>
      <div><div class="info-label">Timestamp</div><div class="info-val">{timestamp}</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Diagnosis Result</h2>
    <div class="row" style="align-items:flex-start;">
      <img src="data:image/png;base64,{img_b64}" width="160" height="160" style="object-fit:cover;"/>
      <div style="flex:1;">
        <div class="info-label">Predicted Disease</div>
        <div style="font-size:1.4rem;font-weight:bold;color:#1b5e20;margin-bottom:8px;">{pred_class.replace('_',' ')}</div>
        <div class="info-label">Confidence</div>
        <div style="font-size:1rem;font-weight:bold;">{confidence:.1f}%</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill"></div></div>
        <div style="margin-top:10px;"><span class="sev">⚠ {sev_label} SEVERITY</span></div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Root Cause</h2>
    <p style="color:#555;line-height:1.6;">{cause}</p>
  </div>

  <div class="section">
    <h2>Recommended Treatment</h2>
    <p style="color:#555;line-height:1.6;">{treatment}</p>
  </div>

  <div class="section">
    <h2>Top 3 Model Predictions</h2>
    <table>
      <tr><th>Rank</th><th>Disease</th><th>Confidence</th></tr>
      {top3_rows}
    </table>
  </div>

  <div class="section">
    <h2>About This Report</h2>
    <p style="font-size:0.82rem;color:#888;line-height:1.7;">
      This report was generated by StomaScope, an AI-powered plant disease detection system
      using a MobileNetV2-based deep learning model trained on the PlantVillage dataset.
      Grad-CAM (Gradient-weighted Class Activation Mapping) is used for model explainability.
      This report is for informational purposes. Consult an agricultural expert for confirmed diagnosis.
    </p>
  </div>

</div>
<div class="footer">
  StomaScope &nbsp;·&nbsp; Bhoomika D, MCA — Soundarya Institute of Management & Science &nbsp;·&nbsp; {timestamp}
</div>
</body></html>"""

    # Try weasyprint first (available on Streamlit Cloud)
    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except Exception:
        pass

    # Fallback: return HTML as-is (user can print-to-PDF from browser)
    return html.encode("utf-8"), "html"


def predict_single(img_pil):
    """Run prediction on a PIL image, return (pred_class, confidence, top3, img_array)."""
    img     = img_pil.resize((224, 224))
    arr     = np.array(img) / 255.0
    arr     = np.expand_dims(arr, axis=0)
    preds   = model(arr, training=False)
    idx     = np.argmax(preds[0])
    conf    = float(preds[0][idx]) * 100
    cls     = class_names[idx]
    top3_idx = np.argsort(preds[0].numpy())[::-1][:3]
    top3     = [(class_names[i], float(preds[0][i]) * 100) for i in top3_idx]
    return cls, conf, top3, arr


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
mode = st.radio("", ["📁 Single Image", "📷 Camera", "🗂️ Batch Prediction"],
                horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE IMAGE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📁 Single Image":

    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <p style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;
           color:#4ade80;margin-bottom:0.3rem;letter-spacing:-0.5px;">📤 Upload Leaf Image</p>
        <p style="font-size:0.82rem;color:#6b9e6b;letter-spacing:2px;text-transform:uppercase;margin:0;">
           JPG &nbsp;·&nbsp; JPEG &nbsp;·&nbsp; PNG &nbsp;—&nbsp; clear, well-lit photos give best results
        </p>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        pred_class, confidence, top3, img_array = predict_single(img_pil)
        sev_label, sev_cls = get_severity(confidence, pred_class)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_prediction(uploaded_file.name, pred_class, confidence)

        # ── Low confidence warning ─────────────────────────────────────────
        if confidence < 50:
            st.markdown("""<div class="warn-box">
                ⚠️ <strong>Low Confidence Detection</strong> — The model is less than 50% confident.
                Please try uploading a <strong>clearer, well-lit image</strong> of the leaf for a more reliable result.
            </div>""", unsafe_allow_html=True)
        else:
            st.success(f"✅ Prediction logged to `{DB_PATH}`")

        # ── TABS ──────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧬 Diagnosis", "📊 Top Predictions", "🔍 Grad-CAM",
            "📖 Disease Info", "📄 Download Report"
        ])

        # ── TAB 1 : Diagnosis ─────────────────────────────────────────────
        with tab1:
            st.markdown('<div class="sec-head">Diagnosis Report</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.markdown(f"""<div class="card card-g">
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
                st.markdown(f"""<div class="card card-o">
                    <div class="c-icon">🔬</div>
                    <div class="c-lbl">Root Cause</div>
                    <div class="c-val">Pathogen Analysis</div>
                    <div class="c-txt">{get_cause(pred_class)}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="card card-b">
                    <div class="c-icon">💊</div>
                    <div class="c-lbl">Recommended Treatment</div>
                    <div class="c-val">Action Plan</div>
                    <div class="c-txt">{get_treatment(pred_class)}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-head" style="margin-top:1.8rem;">Uploaded Image</div>', unsafe_allow_html=True)
            ci, _ = st.columns([1, 2])
            with ci:
                st.image((img_array[0]*255).astype(np.uint8), caption=f"📁 {uploaded_file.name}", width=260)

        # ── TAB 2 : Top-3 ─────────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="sec-head">Top 3 Predictions</div>', unsafe_allow_html=True)
            colors = ["#4ade80","#86efac","#bbf7d0"]
            medals = ["🥇","🥈","🥉"]
            for i,(name,prob) in enumerate(top3):
                st.markdown(f"""<div class="t3row">
                    <div class="t3nm">{medals[i]} {name.replace('_',' ')}</div>
                    <div class="t3bg"><div class="t3bar" style="width:{prob:.1f}%;background:{colors[i]};"></div></div>
                    <div class="t3pc">{prob:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="info-box">
                💡 A large gap between 1st and 2nd place means a confident, reliable prediction.
                If values are close, try uploading a clearer image.
            </div>""", unsafe_allow_html=True)

        # ── TAB 3 : Grad-CAM ──────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="sec-head">Model Explanation — Grad-CAM</div>', unsafe_allow_html=True)
            with st.spinner("Generating heatmap..."):
                original, overlay = render_gradcam(img_array)
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.image(original, caption="📷 Original Image", use_container_width=True)
            with cb:
                st.image(overlay,  caption="🌡️ Grad-CAM — Red = High Model Attention", use_container_width=True)
            st.markdown("""<div class="info-box">
                🔴 <strong style="color:#c4dcc4">Red / Yellow</strong> — Regions the model focused on most.<br>
                🔵 <strong style="color:#c4dcc4">Blue / Green</strong> — Low-attention background areas.<br>
                Grad-CAM makes the AI's decision transparent and verifiable — a key feature for real-world trust.
            </div>""", unsafe_allow_html=True)

        # ── TAB 4 : Disease Info ──────────────────────────────────────────
        with tab4:
            st.markdown('<div class="sec-head">Disease Information</div>', unsafe_allow_html=True)
            info = get_disease_info(pred_class)
            if info:
                st.markdown(f"""<div class="disease-info">
                    <div class="di-title">🔬 {info['full_name']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Pathogen</b>{info['pathogen']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Affected Crops</b>
                        {''.join([f'<span class="di-tag">{c}</span>' for c in info['crops']])}
                    </div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Symptoms</b>{info['symptoms']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Conditions</b>{info['conditions']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Spread</b>{info['spread']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Prevention</b>{info['prevention']}</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Severity</b>{info['severity']}</div>
                </div>""", unsafe_allow_html=True)
            elif "healthy" in pred_class.lower():
                st.markdown("""<div class="disease-info">
                    <div class="di-title">✅ Healthy Leaf</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Status</b>No disease detected.</div>
                    <div class="di-row"><b style="color:#86efac;width:110px;flex-shrink:0;">Recommendation</b>
                        Continue good agricultural practices. Regular monitoring, adequate irrigation, and balanced fertilisation.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Detailed disease info not available for this class yet.")

        # ── TAB 5 : Download Report ───────────────────────────────────────
        with tab5:
            st.markdown('<div class="sec-head">Download Disease Report</div>', unsafe_allow_html=True)
            st.markdown("""<div class="info-box" style="margin-bottom:1.5rem;">
                📄 Generate a complete PDF report containing the diagnosis, confidence score,
                cause, treatment plan, top-3 predictions, and scan metadata.
                Useful for record-keeping and sharing with agricultural experts.
            </div>""", unsafe_allow_html=True)

            if st.button("🖨️ Generate Report", use_container_width=False):
                with st.spinner("Building report..."):
                    result = generate_pdf_report(
                        image_pil  = img_pil,
                        pred_class = pred_class,
                        confidence = confidence,
                        cause      = get_cause(pred_class),
                        treatment  = get_treatment(pred_class),
                        sev_label  = sev_label,
                        top3       = top3,
                        timestamp  = timestamp,
                        image_name = uploaded_file.name
                    )

                if isinstance(result, tuple):
                    # Fallback: HTML file
                    html_bytes, _ = result
                    st.download_button(
                        "⬇️ Download Report (HTML — open in browser → Print → Save as PDF)",
                        data=html_bytes,
                        file_name=f"stomascope_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html", use_container_width=True
                    )
                    st.info("💡 Tip: Open the HTML file in Chrome → Press Ctrl+P → Save as PDF")
                else:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=result,
                        file_name=f"stomascope_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True
                    )


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 — CAMERA INPUT
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "📷 Camera":
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <p style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;
           color:#4ade80;margin-bottom:0.3rem;">📷 Camera Capture</p>
        <p style="font-size:0.82rem;color:#6b9e6b;letter-spacing:2px;text-transform:uppercase;margin:0;">
           Point camera at a leaf — background is automatically removed before prediction
        </p>
    </div>""", unsafe_allow_html=True)

    # Tips box
    st.markdown("""
    <div class="info-box" style="margin-bottom:1.2rem;">
        📌 <strong style="color:#c4dcc4">Tips for best results:</strong><br>
        &nbsp;&nbsp;• Hold the leaf against a <strong style="color:#4ade80">plain background</strong> (white paper / dark cloth)<br>
        &nbsp;&nbsp;• Ensure <strong style="color:#4ade80">good lighting</strong> — natural daylight is best<br>
        &nbsp;&nbsp;• Fill the frame with the leaf as much as possible<br>
        &nbsp;&nbsp;• The system will <strong style="color:#4ade80">auto-remove the background</strong> before prediction
    </div>""", unsafe_allow_html=True)

    camera_img = st.camera_input("")

    if camera_img is not None:
        raw_pil = Image.open(camera_img).convert("RGB")

        with st.spinner("🍃 Isolating leaf from background..."):
            isolated_pil, leaf_mask = isolate_leaf(raw_pil)

        # Show before / after isolation
        st.markdown('<div class="sec-head">Background Removal Preview</div>', unsafe_allow_html=True)
        pr1, pr2 = st.columns(2, gap="large")
        with pr1:
            st.image(np.array(raw_pil),      caption="📷 Original Camera Capture", use_container_width=True)
        with pr2:
            st.image(np.array(isolated_pil), caption="🍃 Leaf Isolated (used for prediction)", use_container_width=True)

        # Check if leaf was actually detected
        leaf_area = float(np.sum(leaf_mask > 0)) / leaf_mask.size
        if leaf_area < 0.05:
            st.markdown("""<div class="warn-box">
                ⚠️ <strong>No leaf detected</strong> — The leaf could not be isolated from the background.
                Try placing the leaf on a contrasting background (white paper works best) and retake the photo.
            </div>""", unsafe_allow_html=True)
        else:
            # Run prediction on isolated leaf
            pred_class, confidence, top3, img_array = predict_single(isolated_pil)
            sev_label, sev_cls = get_severity(confidence, pred_class)
            log_prediction("camera_capture", pred_class, confidence)

            if confidence < 50:
                st.markdown("""<div class="warn-box">
                    ⚠️ <strong>Low Confidence ({:.1f}%)</strong> — Prediction may not be reliable.
                    Try better lighting or a plainer background.
                </div>""".format(confidence), unsafe_allow_html=True)
            else:
                st.success(f"✅ Leaf detected ({leaf_area*100:.0f}% of frame) — Prediction complete!")

            st.markdown('<div class="sec-head">Diagnosis</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.markdown(f"""<div class="card card-g">
                    <div class="c-icon">🦠</div>
                    <div class="c-lbl">Predicted Disease</div>
                    <div class="c-val">{pred_class.replace("_"," ")}</div>
                    <div class="conf-wrap">
                        <div class="conf-row"><span>Confidence</span><span>{confidence:.1f}%</span></div>
                        <div class="conf-bg"><div class="conf-fill" style="width:{confidence:.1f}%"></div></div>
                    </div>
                    <span class="badge {sev_cls}">⚠ {sev_label} Severity</span>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="card card-o">
                    <div class="c-icon">🔬</div><div class="c-lbl">Root Cause</div>
                    <div class="c-val">Pathogen Analysis</div>
                    <div class="c-txt">{get_cause(pred_class)}</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="card card-b">
                    <div class="c-icon">💊</div><div class="c-lbl">Recommended Treatment</div>
                    <div class="c-val">Action Plan</div>
                    <div class="c-txt">{get_treatment(pred_class)}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-head" style="margin-top:1.5rem;">Grad-CAM</div>', unsafe_allow_html=True)
            with st.spinner("Generating heatmap..."):
                original, overlay = render_gradcam(img_array)
            ca, cb = st.columns(2, gap="large")
            with ca:
                st.image(original, caption="🍃 Isolated Leaf",      use_container_width=True)
            with cb:
                st.image(overlay,  caption="🌡️ Grad-CAM Heatmap", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODE 3 — BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "🗂️ Batch Prediction":
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <p style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;
           color:#4ade80;margin-bottom:0.3rem;">🗂️ Batch Prediction</p>
        <p style="font-size:0.82rem;color:#6b9e6b;letter-spacing:2px;text-transform:uppercase;margin:0;">
           Upload multiple leaf images — get predictions for all at once
        </p>
    </div>""", unsafe_allow_html=True)

    batch_files = st.file_uploader("", type=["jpg","jpeg","png"],
                                   accept_multiple_files=True,
                                   label_visibility="collapsed")

    if batch_files:
        st.markdown(f'<div class="sec-head">Results — {len(batch_files)} image(s)</div>', unsafe_allow_html=True)

        results = []
        prog = st.progress(0, text="Analysing images...")

        for i, f in enumerate(batch_files):
            img_pil = Image.open(f).convert("RGB")
            cls, conf, _, _ = predict_single(img_pil)
            sev, _ = get_severity(conf, cls)
            log_prediction(f.name, cls, conf)
            results.append({
                "Image": f.name,
                "Disease": cls.replace("_"," "),
                "Confidence (%)": round(conf, 1),
                "Severity": sev,
                "Treatment": get_treatment(cls)
            })
            prog.progress((i+1)/len(batch_files), text=f"Analysed {i+1}/{len(batch_files)}")

        prog.empty()

        df_results = pd.DataFrame(results)

        # Summary cards
        healthy_count  = sum(1 for r in results if "healthy" in r["Disease"].lower())
        diseased_count = len(results) - healthy_count
        avg_conf_batch = round(sum(r["Confidence (%)"] for r in results) / len(results), 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-n">{len(results)}</div><div class="stat-l">Total Scanned</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-n" style="color:#ef4444">{diseased_count}</div><div class="stat-l">Diseased</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-n">{avg_conf_batch}%</div><div class="stat-l">Avg Confidence</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Batch Results as CSV",
            data=df_results.to_csv(index=False).encode(),
            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True
        )

        # Show individual thumbnails
        st.markdown('<div class="sec-head">Image Thumbnails</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(batch_files), 5))
        for i, f in enumerate(batch_files[:10]):
            with cols[i % 5]:
                img_pil = Image.open(f).convert("RGB")
                sev_badge = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"🟢"}.get(results[i]["Severity"],"⚪")
                st.image(img_pil, caption=f"{sev_badge} {results[i]['Disease']}", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="footer">
    Made with ❤️ for Farmers &nbsp;·&nbsp; AI-Powered Crop Disease Detection
    &nbsp;·&nbsp; Bhoomika D, MCA — Soundarya Institute of Management & Science
</div>""", unsafe_allow_html=True)
