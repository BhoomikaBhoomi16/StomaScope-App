import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="StomaScope - AI Crop Doctor",
    page_icon="🌿",
    layout="wide"
)

# Modern Green Theme CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f5132 0%, #1e7d44 100%);
    }
    .stApp h1 {
        color: #ffffff;
        font-size: 3.8rem;
        text-align: center;
        text-shadow: 0 4px 15px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #a5d6a7;
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2.5rem;
    }
    .card {
        background: rgba(255,255,255,0.95);
        color: #1b5e20;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin: 15px 0;
    }
    .prediction-card {
        background: linear-gradient(90deg, #4caf50, #66bb6a);
        color: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }
    .stButton>button {
        background: #ffffff;
        color: #1b5e20;
        border-radius: 12px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🌿 StomaScope</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Crop Disease Detection with Explainable Insights</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant.png", width=80)
    st.title("StomaScope")
    st.write("Smart AI Assistant for Farmers")
    st.markdown("---")
    st.write("**Features**")
    st.write("• Accurate Disease Prediction")
    st.write("• Visual AI Explanation")
    st.write("• Cause Analysis")
    st.write("• Treatment Recommendations")
    st.markdown("---")
    st.write("Made for Farmers")
    st.write(f"Date: {datetime.now().strftime('%d %B %Y')}")

# Load Model
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# Grad-CAM Function
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

# Upload Section
st.markdown("### 📸 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Drag & drop or browse a clear photo of the affected leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    # Prediction Card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>Predicted Disease</h2>
        <h1 style="margin:10px 0; color:white;">{pred_class.replace('_', ' ')}</h1>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Cause & Treatment in Cards
    col_cause, col_treat = st.columns(2)

    with col_cause:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🔬 Cause of Disease**")
        if "Late_blight" in pred_class:
            st.write("Fungal infection by *Phytophthora infestans*. Spreads rapidly in cool, wet weather.")
        elif "healthy" in pred_class.lower():
            st.write("No disease detected. The leaf appears healthy.")
        else:
            st.write("Common fungal or bacterial infection in this crop.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_treat:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🧪 Recommended Treatment**")
        if "Late_blight" in pred_class:
            st.write("Apply Mancozeb, Chlorothalonil or Ridomil Gold. Remove infected leaves immediately.")
        elif "healthy" in pred_class.lower():
            st.write("Maintain good irrigation, fertilization and crop rotation.")
        else:
            st.write("Use appropriate fungicide and improve air circulation.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Grad-CAM Section
    st.markdown("### 🔍 AI Model Explanation (Grad-CAM)")
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
        st.image(original, caption="Original Image", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM: Areas the AI focused on (Red = High Attention)", use_column_width=True)

    st.caption("Red and yellow regions show where the model paid most attention while making its prediction.")

else:
    st.markdown("""
    <div style="text-align:center; padding:120px 20px; color:#a5d6a7;">
        <h2>Upload a clear leaf image to start diagnosis</h2>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#a5d6a7; font-size:1.1rem;'>"
    "Made with ❤️ for Farmers | AI-Powered Crop Health Monitoring"
    "</p>",
    unsafe_allow_html=True
)
