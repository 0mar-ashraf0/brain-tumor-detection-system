# 🧠 Brain Tumor Detection System: Hybrid Swin-EfficientNet

A clinical-grade deep learning application designed to classify brain tumors from MRI scans with **99.3% accuracy**.  
This system leverages a hybrid architecture combining the global context modeling of **Swin Transformers** with the fine-grained feature extraction of **EfficientNet-B0**.

---

## 🚀 Key Features

- **Hybrid Architecture**  
  Dual-backbone system (Swin Transformer + EfficientNet-B0) for enhanced feature representation.

- **Multi-View Ensemble**  
  Supports multiple MRI views (Axial, Sagittal, Coronal) and averages predictions, simulating real radiology workflows.

- **Clinical UI**  
  Built with **Streamlit**, offering:
  - Real-time probability breakdowns  
  - "Doctor-in-the-loop" contrast enhancement toggle  

- **Test Time Augmentation (TTA)**  
  Applies horizontal flipping during inference to improve prediction stability.

---

## 🏗️ Model Architecture

The system processes **224×224 MRI images** through a parallel pipeline:

### 🔹 Swin Transformer (Tiny)
- Captures long-range dependencies  
- Understands global spatial relationships in brain tissue  

### 🔹 EfficientNet-B0
- Extracts fine-grained textures  
- Detects edges and tumor-specific patterns  

### 🔹 Feature Fusion
- Outputs from both models are concatenated  
- Passed through a custom classification head  

---

## 🧬 Supported Classes

| Class        | Description |
|-------------|------------|
| **Glioma**   | Intra-axial tumors with irregular borders |
| **Meningioma** | Extra-axial tumors, typically well-defined |
| **Pituitary** | Tumors in the sella turcica region |
| **No Tumor** | Normal brain MRI |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```
---

## 📊 Dataset
Dataset used: Brain Tumor MRI Dataset  
(Not included due to size)

---

## start server
```bash
cd brain_tumor_detector
streamlit run app_medical.py
```

---

## activate env
```bash
brain_env\Scripts\activate
```