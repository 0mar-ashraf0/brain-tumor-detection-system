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

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def preprocess_mri(image, apply_enhance):
    img_array = np.array(image)
    avg_brightness = img_array.mean()
    if apply_enhance:
        image = ImageOps.autocontrast(image, cutoff=2)
    return image, avg_brightness

def predict_tta(model, input_tensor):
    flips = [
        input_tensor,
        torch.flip(input_tensor, dims=[3]),
    ]
    all_probs = []
    model.eval()
    with torch.no_grad():
        for x in flips:
            out = model(x)
            # Added .detach() to prevent memory leaks
            prob = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
            all_probs.append(prob)
    return np.mean(all_probs, axis=0)

def get_probs_from_image(model, image):
    img_array = np.array(image)
    augmented = test_transform(image=img_array)
    input_tensor = augmented['image'].float().unsqueeze(0).to(DEVICE)
    return predict_tta(model, input_tensor)

def fuse_views(probs_list):
    """
    Smart fusion for 1, 2, or 3 views.
    - All agree → average
    - Majority detects tumor → weight tumor views by confidence
    - Only 1 detects tumor → trust it (clinical safety)
    - All say no tumor → average
    """
    n = len(probs_list)

    if n == 1:
        return probs_list[0], None, False

    preds = [CLASSES[np.argmax(p)] for p in probs_list]
    tumor_indices  = [i for i, p in enumerate(preds) if p != 'notumor']
    notumor_indices = [i for i, p in enumerate(preds) if p == 'notumor']

    # All agree on same class
    if len(set(preds)) == 1:
        final_probs = np.mean(probs_list, axis=0)
        reason = f"All {n} views agree — averaged"
        views_disagree = False

    # All detect tumor but disagree on type
    elif len(tumor_indices) == n:
        confs   = [np.max(probs_list[i]) for i in tumor_indices]
        total   = sum(confs)
        weights = [c / total for c in confs]
        final_probs = sum(w * probs_list[i] for w, i in zip(weights, tumor_indices))
        dominant = tumor_indices[int(np.argmax(confs))] + 1
        reason = f"All detect tumor but disagree on type — confidence weighted (View {dominant} dominant: {max(confs)/total:.0%})"
        views_disagree = True

    # Majority detects tumor
    elif len(tumor_indices) >= len(notumor_indices):
        confs   = [np.max(probs_list[i]) for i in tumor_indices]
        total   = sum(confs)
        weights = [c / total for c in confs]
        final_probs = sum(w * probs_list[i] for w, i in zip(weights, tumor_indices))
        dominant = tumor_indices[int(np.argmax(confs))] + 1
        reason = f"{len(tumor_indices)}/{n} views detect tumor — tumor views weighted (View {dominant} dominant)"
        views_disagree = True

    # Only 1 detects tumor — trust it clinically
    elif len(tumor_indices) == 1:
        idx = tumor_indices[0]
        final_probs = probs_list[idx]
        reason = f"Only View {idx+1} detects tumor — prioritized for clinical safety"
        views_disagree = True

    # All say no tumor
    else:
        final_probs = np.mean(probs_list, axis=0)
        reason = "All views agree — no tumor detected"
        views_disagree = False

    return final_probs, reason, views_disagree

# Header
st.markdown("""
<div style='text-align:center; padding: 1rem 0 2rem 0;'>
    <h1>🧠 Brain Tumor Detection System</h1>
    <p style='color:#80d8ff; font-size:1.1rem;'>Hybrid Swin Transformer + EfficientNet-B0 | Deep Learning Medical Imaging</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

model = load_trained_model()

st.subheader("🩻 Upload MRI Scans")
st.caption("Upload any combination of views (axial, coronal, sagittal). Uploading all 3 yields maximum accuracy.")

apply_enhance = st.checkbox("⚡ Apply Auto-Contrast (check only if the uploaded scan is extremely dark)", value=False)

# 3 upload slots - Removed "required" bias
upload_col1, upload_col2, upload_col3 = st.columns(3)
with upload_col1:
    st.markdown("**Coronal View**")
    file1 = st.file_uploader("Upload coronal view", type=['jpg', 'jpeg', 'png'], key="file1")
with upload_col2:
    st.markdown("**Sagittal View**")
    file2 = st.file_uploader("Upload sagittal view", type=['jpg', 'jpeg', 'png'], key="file2")
with upload_col3:
    st.markdown("**Axial View**")
    file3 = st.file_uploader("Upload axial view", type=['jpg', 'jpeg', 'png'], key="file3")

# Trigger inference if ANY file is uploaded
if any(f is not None for f in [file1, file2, file3]):
    uploaded = []
    
    # Process all uploaded views safely
    for f, label in [(file1, "Coronal"), (file2, "Sagittal"), (file3, "Axial")]:
        if f is not None:
            try:
                img, brightness = preprocess_mri(Image.open(f).convert('RGB'), apply_enhance)
                probs = get_probs_from_image(model, img)
                uploaded.append({
                    'image':      img,
                    'brightness': brightness,
                    'probs':      probs,
                    'label':      label,
                    'pred_class': CLASSES[np.argmax(probs)],
                    'pred_conf':  np.max(probs)
                })
            except Exception as e:
                st.error(f"⚠️ Error processing the {label} view. Please ensure it is a valid image file. Details: {e}")

    # Only proceed if we successfully processed at least one image
    if len(uploaded) > 0:
        num_views   = len(uploaded)
        probs_list  = [v['probs'] for v in uploaded]
        final_probs, reason, views_disagree = fuse_views(probs_list)

        pred_idx     = np.argmax(final_probs)
        pred_class   = CLASSES[pred_idx]
        pred_conf    = final_probs[pred_idx]
        display_name = pred_class.title().replace('Notumor', 'No Tumor')

        display_names = {
            'glioma':     'Glioma',
            'meningioma': 'Meningioma',
            'notumor':    'No Tumor',
            'pituitary':  'Pituitary'
        }

        st.markdown("---")

        # Show uploaded scans
        st.markdown(f"#### 📋 Uploaded Scans ({num_views} view{'s' if num_views > 1 else ''})")
        preview_cols = st.columns(3)
        for i, col in enumerate(preview_cols):
            with col:
                if i < len(uploaded):
                    v = uploaded[i]
                    st.image(v['image'], 
                        caption=f"{v['label']} {'⚡ Enhanced' if apply_enhance else '✅ Normal'}", 
                        use_column_width=True) # Updated to use_container_width (Streamlit 1.27+)
                    pred_display = v['pred_class'].title().replace('Notumor', 'No Tumor')
                    st.caption(f"Brightness: {v['brightness']:.1f} | {pred_display} ({v['pred_conf']:.1%})")
                else:
                    st.markdown(
                        "<div style='border:2px dashed #1e3a5f; border-radius:10px; "
                        "height:150px; display:flex; align-items:center; justify-content:center; "
                        "color:#546e7a;'>No view uploaded</div>",
                        unsafe_allow_html=True
                    )

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

            # Decision logic
            if reason:
                st.caption(f"🧠 Decision logic: {reason}")

            # Disagreement warning
            if views_disagree:
                st.warning("⚠️ Views disagree — please consult a radiologist for further evaluation.")
            
            # Low confidence warning
            if pred_conf < 0.80:
                st.warning("⚠️ Low confidence — try uploading additional views for better accuracy.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Diagnosis badge
            if pred_class == 'notumor':
                st.success(f"✅ No Tumor Detected — Confidence: {pred_conf:.1%}")
            elif pred_conf > 0.85:
                st.error(f"🔴 {display_name} Tumor Detected — Confidence: {pred_conf:.1%}")
            else:
                st.warning(f"🟡 {display_name} Tumor Suspected — Confidence: {pred_conf:.1%}")

        with result_col2:
            st.markdown("#### 📊 Final Probability Breakdown")
            prob_df = pd.DataFrame({
                'Class': [display_names[c] for c in CLASSES],
                'Probability (%)': final_probs * 100
            }).sort_values('Probability (%)', ascending=True)
            st.bar_chart(prob_df.set_index('Class'), height=250)

            # Per-view breakdown
            if num_views > 1:
                st.markdown("#### 📐 Per-View Breakdown")
                view_cols = st.columns(num_views)
                for i, col in enumerate(view_cols):
                    with col:
                        v = uploaded[i]
                        pred_display = v['pred_class'].title().replace('Notumor', 'No Tumor')
                        st.caption(f"{v['label']} — {pred_display} ({v['pred_conf']:.1%})")
                        for j, c in enumerate(CLASSES):
                            st.progress(float(v['probs'][j]), 
                                text=f"{display_names[c]}: {v['probs'][j]:.1%}")

st.markdown("---")

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

**Best Practice:**
Upload all 3 views for maximum
diagnostic accuracy:
- Coronal
- Sagittal 
- Axial 

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