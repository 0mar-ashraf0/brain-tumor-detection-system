import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import HybridTumorModel, load_model
from config import MODEL_PATH, DEVICE, CLASSES, IMG_SIZE

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0a0e1a !important;
    }
    .main .block-container {
        background-color: #0a0e1a !important;
        padding: 2rem 3rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1226 !important;
        border-right: 1px solid #1e3a5f !important;
    }
    * { color: #e8eaf6 !important; }
    p, div, span, label { color: #cfd8dc !important; }
    h1 { color: #00e5ff !important; font-size: 2.2rem !important; letter-spacing: 1px; }
    h2 { color: #40c4ff !important; }
    h3 { color: #80d8ff !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1b2a, #1a2744) !important;
        border: 1px solid #00e5ff44 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    [data-testid="stMetricValue"] { color: #00e5ff !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #80d8ff !important; }
    [data-testid="stFileUploader"] {
        background-color: #0d1b2a !important;
        border: 2px dashed #00e5ff66 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    button {
        background: linear-gradient(90deg, #0077b6, #00b4d8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    [data-testid="stAlert"] { border-radius: 10px !important; font-weight: bold !important; }
    [data-testid="stArrowVegaLiteChart"] {
        background-color: #0d1b2a !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 1px solid #1e3a5f !important;
    }
    [data-testid="stImage"] img {
        border: 2px solid #00e5ff44 !important;
        border-radius: 10px !important;
    }
    hr { border-color: #1e3a5f !important; }
    [data-testid="stSidebar"] * { color: #b0bec5 !important; }
    [data-testid="stSidebar"] h1 { color: #00e5ff !important; }
    [data-testid="stSidebar"] strong { color: #40c4ff !important; }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Transforms ---
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Updated: Enhancement is now controlled by the user
def preprocess_mri(image, apply_enhance):
    img_array = np.array(image)
    avg_brightness = img_array.mean()
    if apply_enhance:
        image = ImageOps.autocontrast(image, cutoff=2)
    return image, avg_brightness

# Updated: Removed vertical flips to prevent confusing the model
def predict_tta(model, input_tensor):
    flips = [
        input_tensor,
        torch.flip(input_tensor, dims=[3]), # Horizontal flip only
    ]
    all_probs = []
    model.eval()
    with torch.no_grad():
        for x in flips:
            out = model(x)
            prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(prob)
    return np.mean(all_probs, axis=0)

def get_probs_from_image(model, image):
    img_array = np.array(image)
    augmented = test_transform(image=img_array)
    input_tensor = augmented['image'].float().unsqueeze(0).to(DEVICE)
    return predict_tta(model, input_tensor)

# --- Header ---
st.markdown("""
<div style='text-align:center; padding: 1rem 0 2rem 0;'>
    <h1>🧠 Brain Tumor Detection System</h1>
    <p style='color:#80d8ff; font-size:1.1rem;'>Hybrid Swin Transformer + EfficientNet-B0 | Deep Learning Medical Imaging</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

model = load_trained_model()

# --- Main UI ---
st.subheader("🩻 Upload MRI Scans")
st.caption("Upload 1 or 2 views (e.g. coronal + sagittal) for more accurate diagnosis")

# NEW: The Doctor's Control Toggle
apply_enhance = st.checkbox("⚡ Apply Auto-Contrast (Check this only if the uploaded scan is extremely dark)", value=False)

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    st.markdown("**View 1** (required)")
    file1 = st.file_uploader("Upload first MRI view", type=['jpg', 'jpeg', 'png'], key="file1")
with upload_col2:
    st.markdown("**View 2** (optional)")
    file2 = st.file_uploader("Upload second MRI view", type=['jpg', 'jpeg', 'png'], key="file2")

if file1 is not None:
    # Process view 1
    image1 = Image.open(file1).convert('RGB')
    image1, brightness1 = preprocess_mri(image1, apply_enhance)
    probs1 = get_probs_from_image(model, image1)

    # Process view 2 if uploaded
    if file2 is not None:
        image2 = Image.open(file2).convert('RGB')
        image2, brightness2 = preprocess_mri(image2, apply_enhance)
        probs2 = get_probs_from_image(model, image2)
        final_probs = np.mean([probs1, probs2], axis=0)
        num_views = 2
    else:
        final_probs = probs1
        num_views = 1

    pred_idx   = np.argmax(final_probs)
    pred_class = CLASSES[pred_idx]
    pred_conf  = final_probs[pred_idx]
    display_name = pred_class.title().replace('Notumor', 'No Tumor')

    st.markdown("---")

    # Show uploaded scans (Updated to use_container_width)
    st.markdown(f"#### 📋 Uploaded Scans ({num_views} view{'s' if num_views > 1 else ''})")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(image1, caption=f"View 1 {'⚡ Enhanced' if apply_enhance else '✅ Normal'}", use_column_width=True)
        st.caption(f"Avg brightness: {brightness1:.1f}")
    with preview_col2:
        if file2 is not None:
            st.image(image2, caption=f"View 2 {'⚡ Enhanced' if apply_enhance else '✅ Normal'}", use_column_width=True)
            st.caption(f"Avg brightness: {brightness2:.1f}")
        else:
            st.info("No second view uploaded — using single view prediction")

    st.markdown("---")

    # Results
    result_col1, result_col2 = st.columns([1, 1], gap="large")

    with result_col1:
        st.markdown(f"#### 🔬 AI Diagnosis — {num_views} View{'s' if num_views > 1 else ''}")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Predicted Class", display_name)
        with m2:
            st.metric("Confidence Score", f"{pred_conf:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)

        if pred_conf < 0.80:
            st.warning("⚠️ Low confidence — try adding a second view for better accuracy")

        if pred_class == 'notumor':
            st.success(f"✅ No Tumor Detected — Confidence: {pred_conf:.1%}")
        elif pred_conf > 0.85:
            st.error(f"🔴 {display_name} Tumor Detected — Confidence: {pred_conf:.1%}")
        else:
            st.warning(f"🟡 {display_name} Tumor Suspected — Confidence: {pred_conf:.1%}")

    with result_col2:
        st.markdown("#### 📊 Class Probability Breakdown")
        display_names = {
            'glioma':     'Glioma',
            'meningioma': 'Meningioma',
            'notumor':    'No Tumor',
            'pituitary':  'Pituitary'
        }
        prob_df = pd.DataFrame({
            'Class': [display_names[c] for c in CLASSES],
            'Probability (%)': final_probs * 100
        }).sort_values('Probability (%)', ascending=True)
        st.bar_chart(prob_df.set_index('Class'), height=250)

        if num_views == 2:
            st.markdown("#### 📐 Per-View Breakdown")
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                st.caption("View 1")
                for i, c in enumerate(CLASSES):
                    st.progress(float(probs1[i]), text=f"{display_names[c]}: {probs1[i]:.1%}")
            with view_col2:
                st.caption("View 2")
                for i, c in enumerate(CLASSES):
                    st.progress(float(probs2[i]), text=f"{display_names[c]}: {probs2[i]:.1%}")

st.markdown("---")

# --- Sidebar Info ---
st.sidebar.markdown("## 🧠 System Info")
st.sidebar.markdown("""
**Model Architecture:**
- Swin Transformer (Tiny)
- EfficientNet-B0
- Feature Fusion Classifier

**Training:**
- 4-Class Classification
- Glioma · Meningioma · No Tumor · Pituitary
- ImageNet Pretrained Backbones

**Tip:**
Upload coronal + sagittal views
together for best accuracy.

**Disclaimer:**
Research prototype only.
Not for clinical diagnosis.
Always consult a physician.
""")

st.markdown(
    "<p style='text-align:center; color:#546e7a; font-size:0.85rem;'>"
    "🩺 Brain Tumor AI Demo | Not for clinical use | Consult a medical professional</p>",
    unsafe_allow_html=True
)