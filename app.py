import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import sqlite3
import datetime
import pandas as pd
import os

st.set_page_config(page_title="StomaScope", page_icon="🌿", layout="wide")

# Modern CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #f0f7f0, #e8f5e9);}
    h1 {color: #1b5e20; text-align: center; font-size: 3rem;}
    .section-title {color: #2e7d32; font-size: 1.8rem; margin: 30px 0 15px;}
    .result-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-box {border-left: 6px solid #4caf50;}
    .cause-box {border-left: 6px solid #ff9800;}
    .treatment-box {border-left: 6px solid #2196f3;}
    .log-box {
        background: #f9fbe7;
        border-left: 6px solid #8bc34a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE SETUP (SQLite)
# ─────────────────────────────────────────────
DB_PATH = "stomascope_logs.db"

def init_db():
    """Create predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name  TEXT    NOT NULL,
            prediction  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(image_name: str, prediction: str, confidence: float):
    """Insert one prediction record into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (image_name, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (image_name, prediction, round(confidence, 2),
         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def fetch_logs(limit: int = 50) -> pd.DataFrame:
    """Return the most recent prediction records as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, image_name, prediction, confidence, timestamp "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df

def clear_logs():
    """Delete all records from the predictions table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

# Initialise DB on every run (safe – uses CREATE IF NOT EXISTS)
init_db()

# ─────────────────────────────────────────────
# SIDEBAR – Prediction Log Viewer
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Prediction Log")
    st.markdown("*All predictions are stored in a local SQLite database.*")
    st.markdown(f"**DB file:** `{DB_PATH}`")

    logs_df = fetch_logs()
    if logs_df.empty:
        st.info("No predictions logged yet. Upload a leaf image to get started.")
    else:
        st.markdown(f"**Total records:** {len(fetch_logs(limit=100000))}")
        st.dataframe(
            logs_df.rename(columns={
                "id": "ID", "image_name": "Image",
                "prediction": "Prediction", "confidence": "Confidence (%)",
                "timestamp": "Timestamp"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Download as CSV
        csv_data = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Log as CSV",
            data=csv_data,
            file_name="stomascope_predictions.csv",
            mime="text/csv"
        )

        # Clear log
        if st.button("🗑️ Clear All Logs", type="secondary"):
            clear_logs()
            st.success("Logs cleared!")
            st.rerun()

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
st.title("🌿 StomaScope")
st.markdown("**AI-Powered Crop Disease Detection System**")

# Load model
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# Grad-CAM
def get_gradcam(img_array, model):
    """Fixed Grad-CAM for MobileNetV2 transfer learning."""
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
        st.warning("⚠️ Could not generate Grad-CAM heatmap")
        return np.zeros((7, 7)), int(pred_index)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), int(pred_index)

# ─────────────────────────────────────────────
# FILE UPLOAD & PREDICTION
# ─────────────────────────────────────────────
st.markdown("### Upload Leaf Image")
uploaded_file = st.file_uploader("Choose a clear photo of the affected leaf",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = float(preds[0][pred_idx]) * 100
    pred_class = class_names[pred_idx]

    # ── Log to SQLite ──────────────────────────
    log_prediction(
        image_name=uploaded_file.name,
        prediction=pred_class,
        confidence=confidence
    )
    st.success(f"✅ Prediction logged to database (`{DB_PATH}`)")

    # ── Result Cards ───────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="result-box prediction-box">', unsafe_allow_html=True)
        st.markdown("**Predicted Disease**")
        st.markdown(f"<h2 style='color:#1b5e20;'>{pred_class.replace('_', ' ')}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-box cause-box">', unsafe_allow_html=True)
        st.markdown("**Cause**")
        if "Late_blight" in pred_class:
            st.write("Fungal infection by *Phytophthora infestans*. Spreads in cool, wet weather.")
        elif "healthy" in pred_class.lower():
            st.write("No disease detected. Leaf appears healthy.")
        else:
            st.write("Fungal / bacterial infection common in this crop.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="result-box treatment-box">', unsafe_allow_html=True)
        st.markdown("**Recommended Treatment**")
        if "Late_blight" in pred_class:
            st.write("Apply Mancozeb or Chlorothalonil. Remove infected leaves immediately.")
        elif "healthy" in pred_class.lower():
            st.write("Continue good agricultural practices.")
        else:
            st.write("Use appropriate fungicide and improve air circulation.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Grad-CAM Visualization ─────────────────
    st.markdown("### 🔍 Model Explanation (Grad-CAM)")
    with st.spinner("Generating visual explanation..."):
        heatmap, _ = get_gradcam(img_array, model)
        original = (img_array[0] * 255).astype(np.uint8)
        h = cv2.resize(heatmap, (224, 224))
        h = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", width=450)
    with col2:
        st.image(overlay,
                 caption="Grad-CAM: Areas the AI focused on (Red = High Attention)",
                 width=450)

    st.caption("Red and yellow regions show where the model paid most attention while making its prediction.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>Made with ❤️ for Farmers | "
    "AI-Powered Crop Disease Detection</p>",
    unsafe_allow_html=True
)
