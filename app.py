import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json

# Load model and class names
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('stomascopes_model_v1.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_resources()

# Fixed Grad-CAM (gradients computed correctly)
def get_gradcam(img_array, model):
    base = model.layers[1]  # MobileNetV2 base

    with tf.GradientTape() as tape:
        conv_output = base(img_array)           # forward through base
        tape.watch(conv_output)

        # Forward through the rest of the model (GAP + Dense)
        gap = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        preds = model.layers[-1](gap)           # final Dense layer

        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_output)

    if grads is None:
        st.error("Gradients could not be computed. Try a different image.")
        return np.zeros((7, 7)), pred_index.numpy()  # fallback empty heatmap

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), int(pred_index)

# Streamlit App
st.set_page_config(page_title="StomaScope", layout="wide")

st.title("🌱 StomaScope - Crop Disease Detector")
st.markdown("Upload a leaf image to get prediction + Grad-CAM explanation")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model(img_array, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_class = class_names[pred_idx]

    st.success(f"**Predicted:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Grad-CAM
    with st.spinner("Generating Grad-CAM..."):
        heatmap, pred_idx = get_gradcam(img_array, model)

        original = (img_array[0] * 255).astype(np.uint8)
        h = cv2.resize(heatmap, (224, 224))
        h = np.uint8(255 * h)
        h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, h_color, 0.35, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", width=350)
    with col2:
        st.image(overlay, caption="Grad-CAM (Red/Yellow = Model Focus)", width=350)
