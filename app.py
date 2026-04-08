import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json

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
</style>
""", unsafe_allow_html=True)

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
    
    if grads is None:
        st.warning("⚠️ Could not generate Grad-CAM heatmap")
        return np.zeros((7, 7)), int(pred_index)
    
    # Generate heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    
    return heatmap.numpy(), int(pred_index)

# Main Content
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
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    # Results in nice separate boxes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="result-box prediction-box">', unsafe_allow_html=True)
        st.markdown("**Predicted Disease**")
        st.markdown(f"<h2 style='color:#1b5e20;'>{pred_class.replace('_', ' ')}</h2>", unsafe_allow_html=True)
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

    # Grad-CAM Images
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
        st.image(original, caption="Original Image", width=450)
    with col2:
        st.image(overlay, caption="Grad-CAM: Areas the AI focused on (Red = High Attention)", width=450)

    st.caption("Red and yellow regions show where the model paid most attention while making its prediction.")
# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#666;'>Made with ❤️ for Farmers | AI-Powered Crop Disease Detection</p>", unsafe_allow_html=True)
