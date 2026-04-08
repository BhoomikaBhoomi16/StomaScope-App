import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime

# ==================== MODERN PAGE CONFIG ====================
st.set_page_config(
    page_title="StomaScope - AI Crop Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Modern CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f5132 0%, #1e7d44 100%);
        color: white;
    }
    .stApp h1 {
        color: #ffffff;
        font-size: 3.5rem;
        text-align: center;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .header-subtitle {
        text-align: center;
        color: #a5d6a7;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255,255,255,0.95);
        color: #1b5e20;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 15px 0;
    }
    .result-box {
        background: linear-gradient(90deg, #4caf50, #81c784);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    }
    .stButton>button {
        background: #ffffff;
        color: #1b5e20;
        border: none;
        border-radius: 12px;
        height: 52px;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("<h1>🌿 StomaScope</h1>", unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">AI-Powered Crop Disease Detection System</p>', unsafe_allow_html=True)

# Sidebar with nice design
with st.sidebar:
    st.markdown("### 🌱 About StomaScope")
    st.write("Advanced AI system that detects crop diseases and provides actionable treatment recommendations.")
    st.markdown("---")
    st.write("**Key Features**")
    st.write("• High Accuracy Prediction")
    st.write("• Visual AI Explanation (Grad-CAM)")
    st.write("• Cause Analysis")
    st.write("• Treatment Recommendations")
    st.markdown("---")
    st.write(f"**Student:** Bhoomika")
    st.write(f"**Date:** {datetime.now().strftime('%d %B %Y')}")

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
st.markdown("### 📸 Upload Leaf Image")

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
    <div class="result-box">
        <h2>Predicted Disease</h2>
        <h1 style="margin:0; color:white;">{pred_class.replace('_', ' ')}</h1>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Disease Information (Clean Cards)
    disease_info = {
        "Tomato___Late_blight": {
            "cause": "Fungal disease caused by Phytophthora infestans. Spreads rapidly in cool, wet weather.",
            "treatment": "Apply Mancozeb, Chlorothalonil or Ridomil Gold. Remove infected leaves immediately."
        },
        "Potato___Late_blight": {
            "cause": "Caused by Phytophthora infestans. Common in moist and cool conditions.",
            "treatment": "Use Metalaxyl or Mancozeb fungicides. Use certified seed potatoes."
        },
        "Potato___healthy": {
            "cause": "No disease detected.",
            "treatment": "Continue proper irrigation and fertilization practices."
        },
        "Tomato___healthy": {
            "cause": "No disease detected.",
            "treatment": "Maintain good agricultural practices."
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

    # Grad-CAM Section
    st.markdown("### 🔍 AI Model Explanation (Grad-CAM)")
    with st.spinner("Analyzing where the AI focused..."):
        heatmap, _ = get_gradcam(img_array, model)
        original = (img_array[0] * 255).astype(np.uint8)
        h = cv2.resize(heatmap, (224, 224))
        h = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(original, caption="Original Leaf", use_column_width=True)
    with col_right:
        st.image(overlay, caption="Grad-CAM Heatmap (Red = High Attention)", use_column_width=True)

    st.caption("The red/yellow areas show the parts of the leaf the AI used to make its decision.")

else:
    st.markdown("""
    <div style="text-align:center; padding:100px 20px; color:#a5d6a7;">
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
