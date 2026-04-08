import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import sqlite3
from datetime import datetime
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="StomaScope", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #f0f7f0, #e8f5e9);}
    h1 {color: #1b5e20; text-align: center;}
    .card {background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ==================== DATABASE SETUP ====================
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY, username TEXT, image_name TEXT, disease TEXT, 
                  confidence REAL, date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==================== LOGIN SYSTEM ====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def login():
    st.title("🌿 StomaScope - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Simple demo login (you can make it more secure later)
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Please enter username and password")

if not st.session_state.logged_in:
    login()
    st.stop()

# ==================== MAIN APP ====================
st.title(f"🌿 Welcome back, {st.session_state.username}!")

# Load model
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# Grad-CAM function (fixed version)
def get_gradcam(img_array, model):
    base = model.layers[1]
    with tf.GradientTape() as tape:
        conv_output = base(img_array)
        tape.watch(conv_output)
        preds = model(img_array, training=False)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]
    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        return np.zeros((7, 7)), int(pred_index)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy(), int(pred_index)

# Upload
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    # Save to history
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections (username, image_name, disease, confidence, date) VALUES (?, ?, ?, ?, ?)",
              (st.session_state.username, uploaded_file.name, pred_class, confidence, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

    st.success(f"**Predicted Disease:** {pred_class.replace('_', ' ')}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Cause & Treatment
    disease_info = {
        "Tomato___Late_blight": {"cause": "Fungal infection by Phytophthora infestans.", "treatment": "Mancozeb or Chlorothalonil fungicide."},
        "Potato___Late_blight": {"cause": "Phytophthora infestans fungus.", "treatment": "Metalaxyl or Mancozeb fungicides."},
        "Potato___healthy": {"cause": "No disease detected.", "treatment": "Maintain good practices."},
        "Tomato___healthy": {"cause": "No disease detected.", "treatment": "Continue good practices."}
    }

    if pred_class in disease_info:
        info = disease_info[pred_class]
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Cause:** {info['cause']}")
        with col2:
            st.success(f"**Recommended Treatment:** {info['treatment']}")

    # Grad-CAM
    with st.spinner("Generating Grad-CAM..."):
        heatmap, _ = get_gradcam(img_array, model)
        original = (img_array[0] * 255).astype(np.uint8)
        h = cv2.resize(heatmap, (224, 224))
        h = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", width=400)
    with col2:
        st.image(overlay, caption="Grad-CAM (Red = High Attention)", width=400)

# ==================== HISTORY SECTION ====================
st.markdown("### 📜 Previous Detections")

conn = sqlite3.connect('history.db')
c = conn.cursor()
c.execute("SELECT date, disease, confidence FROM detections WHERE username = ? ORDER BY date DESC", 
          (st.session_state.username,))
history = c.fetchall()
conn.close()

if history:
    for entry in history:
        st.write(f"**{entry[0]}** - {entry[1]} ({entry[2]:.2f}%)")
else:
    st.info("No previous detections yet.")

# Logout button
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
