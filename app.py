import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="StomaScope - Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern & Professional Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%);
    }
    .stApp h1 {
        color: #1b5e20;
        font-size: 3.2rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .header-subtitle {
        text-align: center;
        color: #2e7d32;
        font-size: 1.4rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 20px 0;
    }
    .stButton>button {
        background: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: 600;
    }
    .expander {
        border-radius: 12px;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("<h1>🌿 StomaScope</h1>", unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">AI-Powered Crop Disease Detection with Explainable AI</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant.png", width=80)
    st.title("StomaScope")
    st.markdown("### Smart Farming Assistant")
    st.write("Early detection of crop diseases using Deep Learning and Grad-CAM visualization.")
    st.markdown("---")
    st.write("**Key Features:**")
    st.write("• Instant Disease Prediction")
    st.write("• Visual Explanation (Grad-CAM)")
    st.write("• Cause & Treatment Recommendations")
    st.markdown("---")
    st.write(f"**Developed by:** Bhoomika")
    st.write(f"**Date:** {datetime.now().strftime('%d %B %Y')}")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# ==================== GRAD-CAM FUNCTION ====================
def get_gradcam(img_array, model):
    """Robust Grad-CAM that works with MobileNetV2 transfer learning"""
    # Get the base model (MobileNetV2)
    base_model = model.layers[1]
    
    with tf.GradientTape() as tape:
        # Forward pass through base model
        conv_output = base_model(img_array, training=False)
        tape.watch(conv_output)
        
        # Continue through the rest of the model (GlobalAveragePooling + Dense)
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        preds = model.layers[-1](x)   # final Dense layer
        
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    # Get gradients
    grads = tape.gradient(class_score, conv_output)
    
    if grads is None:
        st.warning("⚠️ Gradients were None - using empty heatmap")
        return np.zeros((7, 7)), int(pred_index)
    
    # Create heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    
    return heatmap.numpy(), int(pred_index)
# ==================== MAIN INTERFACE ====================
st.markdown("### 📸 Upload a Leaf Image for Analysis")

uploaded_file = st.file_uploader(
    "Drag and drop or browse a clear photo of the affected leaf",
    type=["jpg", "jpeg", "png"],
    help="For best results, use well-lit, close-up images of the leaf"
)

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    # Result Card
    st.markdown(f"""
    <div class="result-card">
        <h2>✅ Predicted Disease: <b>{pred_class.replace('_', ' ')}</b></h2>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Disease Information
    disease_info = {
        "Tomato___Late_blight": {
            "cause": "Caused by the fungus Phytophthora infestans. Spreads quickly in cool, wet weather and high humidity.",
            "treatment": "Apply Mancozeb, Chlorothalonil, or Ridomil Gold fungicide. Remove infected leaves and improve air circulation."
        },
        "Potato___Late_blight": {
            "cause": "Caused by Phytophthora infestans. Favoured by cool, moist conditions and infected seed tubers.",
            "treatment": "Use Mancozeb or Metalaxyl-based fungicides early. Use certified disease-free seed potatoes."
        },
        "Potato___healthy": {
            "cause": "No disease detected. The leaf appears healthy.",
            "treatment": "Maintain proper irrigation, balanced fertilization, and crop rotation."
        },
        "Tomato___healthy": {
            "cause": "No disease detected. The leaf appears healthy.",
            "treatment": "Continue good agricultural practices."
        },
        "Apple___Apple_scab": {
            "cause": "Fungal disease caused by Venturia inaequalis. Spreads during wet spring weather.",
            "treatment": "Apply Captan or Mancozeb fungicide. Prune trees for better air circulation."
        }
    }

    if pred_class in disease_info:
        info = disease_info[pred_class]
        col_cause, col_treat = st.columns(2)
        with col_cause:
            st.markdown("**🔬 Cause of Disease**")
            st.info(info["cause"])
        with col_treat:
            st.markdown("**🧪 Recommended Treatment**")
            st.success(info["treatment"])

    # Grad-CAM Visualization
        # Grad-CAM Visualization
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
        st.image(original, caption="📸 Original Image", width=450)
    with col2:
        st.image(overlay, caption="🔥 Grad-CAM: Areas the AI focused on (Red = High Attention)", width=450)

    st.caption("Red and yellow regions show where the model paid most attention while making its prediction.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Made with ❤️ for Farmers | Powered by Deep Learning & Explainable AI"
    "</p>",
    unsafe_allow_html=True
)
