import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="StomaScope - Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('https://source.unsplash.com/1600x900/?nature,plants');
        background-size: cover;
        background-position: center;
        color: white;
    }
    .header {
        text-align: center;
        padding: 80px 20px 40px;
        background: rgba(0,0,0,0.5);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    h1 {
        font-size: 3.5rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.6);
    }
    .subtitle {
        font-size: 1.4rem;
        color: #a5d6a7;
    }
    .card {
        background: white;
        color: #1b5e20;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 15px 0;
    }
    .result-card {
        background: linear-gradient(90deg, #4caf50, #81c784);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header (similar to your image)
st.markdown("""
<div class="header">
    <h1>🌿 StomaScope</h1>
    <p class="subtitle">AI-Powered Plant Disease Detection System</p>
    <p>Early detection • Visual explanation • Smart recommendations for farmers</p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# Grad-CAM function
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

# Main Upload Section
st.markdown("### 📸 Upload a Leaf Image for Analysis")

uploaded_file = st.file_uploader(
    "Choose a clear photo of the affected leaf",
    type=["jpg", "jpeg", "png"],
    help="For best results, use well-lit, close-up images"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    # Beautiful Result Card
    st.markdown(f"""
    <div class="result-card">
        <h2>Predicted Disease</h2>
        <h1 style="margin:10px 0;">{pred_class.replace('_', ' ')}</h1>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Disease Info
    disease_info = {
        "Tomato___Late_blight": {
            "cause": "Fungal infection by Phytophthora infestans. Spreads in cool, wet weather.",
            "treatment": "Apply Mancozeb or Chlorothalonil. Remove infected leaves immediately."
        },
        "Potato___Late_blight": {
            "cause": "Caused by Phytophthora infestans. Common in moist conditions.",
            "treatment": "Use Metalaxyl or Mancozeb fungicides. Use certified seeds."
        },
        "Potato___healthy": {
            "cause": "No disease detected.",
            "treatment": "Maintain proper irrigation and fertilization."
        },
        "Tomato___healthy": {
            "cause": "No disease detected.",
            "treatment": "Continue good agricultural practices."
        }
    }

    if pred_class in disease_info:
        info = disease_info[pred_class]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔬 Cause**")
            st.info(info["cause"])
        with col2:
            st.markdown("**🧪 Recommended Treatment**")
            st.success(info["treatment"])

    # Grad-CAM
    st.markdown("### 🔍 AI Model Explanation (Grad-CAM)")
    with st.spinner("Generating explanation..."):
        heatmap, _ = get_gradcam(img_array, model)
        original = (img_array[0] * 255).astype(np.uint8)
        h = cv2.resize(heatmap, (224, 224))
        h = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(original, caption="Original Image", width=500)
    with col_right:
        st.image(overlay, caption="Grad-CAM: Areas the AI focused on (Red = High Attention)", width=500)

else:
    st.markdown("""
    <div style="text-align:center; padding:100px 20px; color:#ddd;">
        <h2>Upload a clear leaf image to start diagnosis</h2>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#a5d6a7;'>Made with ❤️ for Farmers | AI-Powered Crop Health Monitoring</p>",
    unsafe_allow_html=True
)
