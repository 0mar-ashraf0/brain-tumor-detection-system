# Brain Tumor Detector - System Architecture

## Overview
Hybrid deep learning system for 4-class brain tumor classification from MRI scans using **Swin Transformer + EfficientNet-B0** feature fusion.

---

## 1. Model Architecture (`model.py`)

```
Input Tensor: (batch_size, 3, 224, 224)
│
├─►┌─────────────────────────────┐
│  │  Swin Transformer Tiny      │  (timm - swin_tiny_patch4_window7_224)
│  │  pretrained=True            │
│  │  num_classes=0              │  <- feature extractor only
│  │  global_pool='avg'          │
│  └─────────────────────────────┘
│            │
│            ▼
│    768-D Feature Vector
│
├─►┌─────────────────────────────┐
│  │  EfficientNet-B0            │  (torchvision)
│  │  features()                 │  <- conv feature maps
│  │  AdaptiveAvgPool2d(1)       │
│  └─────────────────────────────┘
│            │
│            ▼
│   1280-D Feature Vector
│
│    ┌───────────────────────────┐
└───►│     Concatenate           │  [768 + 1280] = 2048-D
     └───────────────────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │  Fusion Head (MLP)        │
     │  ─────────────────────    │
     │  Linear(2048 → 512)       │
     │  ReLU()                   │
     │  Dropout(0.5)             │
     │  Linear(512 → 4)          │
     └───────────────────────────┘
                  │
                  ▼
          Output Logits (4 classes)
```

### Class: `HybridTumorModel`
| Component | Details |
|-----------|---------|
| `self.swin` | Swin Transformer Tiny, 768-D output |
| `self.effnet` | EfficientNet-B0 with `classifier = nn.Identity()` |
| `self.fusion_pool` | `nn.AdaptiveAvgPool2d(1)` for EffNet features |
| `self.fusion_fc` | MLP: 2048 → 512 → 4, with ReLU + Dropout(0.5) |

### Parameters Summary
| Source | Value |
|--------|-------|
| Swin Features | 768 |
| EfficientNet Features | 1280 |
| Fusion Hidden Dim | 512 |
| Output Classes | 4 |
| Total Input Size | 224 × 224 × 3 |

---

## 2. Configuration (`config.py`)

```python
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4
IMG_SIZE = 224
SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224'
EFFNET_MODEL_NAME = 'efficientnet_b0'
SWIN_FEATURES = 768
EFFNET_FEATURES = 1280

BATCH_SIZE = 16
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
CV_FOLDS = 4
TEST_SPLIT_SIZE = 0.1
DEVICE = 'cuda' | 'cpu'
```

---

## 3. Training Pipeline (`train.py`)

### Data Splitting Strategy
```
Total Dataset (Train Folder)
│
├─ 90% CV Pool ──┬── Fold 1 (train/val)
│                ├── Fold 2 (train/val)
│                ├── Fold 3 (train/val)
│                ├── Fold 4 (train/val)
│                └── Final Retrain on Full 90%
│
└─ 10% Holdout Test Set (never seen during CV)
```

### Augmentation Pipeline (Albumentations)
```
A.Resize(224, 224)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomRotate90(p=0.5)
A.Transpose(p=0.3)
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5)
A.RandomBrightnessContrast(p=0.3)
A.GaussNoise(p=0.2)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

### Training Configuration
| Setting | Implementation |
|---------|----------------|
| Optimizer | AdamW with **Differential Learning Rates** |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss with **inverse class weights** |
| Early Stopping | Patience = 5 epochs |
| Logging | TensorBoard per-fold metrics |
| Evaluation | Confusion Matrix + ROC curves per fold |

### Optimizer Structure
```python
optim.AdamW([
    {'params': model.swin.parameters(),      'lr': LR_BACKBONE},
    {'params': model.effnet.parameters(),    'lr': LR_BACKBONE},
    {'params': model.fusion_fc.parameters(), 'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)
```

---

## 4. Data Pipeline (`data_utils.py`)

```
ImageFolder(root_dir)
    │
    ▼
TumorDataset
    │
    ├─ Transform: Albumentations Compose
    │       ├── Resize, Normalize
    │       └── Augmentations (train/test variants)
    │
    ▼
DataLoader(batch_size=16, shuffle=True/False)
```

### Transform Variants
| Mode | Augmentations |
|------|--------------|
| Train | Flip, Rotate90, Transpose, ShiftScaleRotate, BrightnessContrast, GaussNoise |
| Test | Resize + Normalize only |

---

## 5. Inference / Application (`app_medical.py`)

```
User Uploads MRI (JPG/PNG)
    │
    ▼
PIL.Image → np.array
    │
    ▼
test_transform (Albumentations)
    │
    ▼
Input Tensor (1, 3, 224, 224) → DEVICE
    │
    ▼
HybridTumorModel.eval() → forward pass
    │
    ▼
Softmax → Class Probabilities
    │
    ▼
Prediction Display (Streamlit UI)
    ├── Predicted Class Metric
    ├── Confidence Score Metric
    ├── Probability Bar Chart
    └── Diagnosis Badge (Success/Warning/Error)
```

### UI Features
- Dark navy medical theme (custom CSS)
- Confidence thresholds:
  - `notumor` → Success (green)
  - `> 0.85` → Error (red - tumor detected)
  - `<= 0.85` → Warning (yellow - suspected)

---

## 6. Directory Structure

```
brain_tumor_detector/
├── model.py              # HybridTumorModel definition
├── train.py              # 4-Fold CV training + final retrain
├── config.py             # Hyperparameters & paths
├── data_utils.py         # Dataset + transforms + loaders
├── app_medical.py        # Streamlit inference app
├── requirements.txt      # Python dependencies
├── best_model.pth        # Saved model weights
└── results/
    ├── cm_fold_*.png     # Confusion matrices
    ├── roc_fold_*.png    # ROC curves
    ├── cv_fold_*_curves.png  # Train/val loss & accuracy
    └── logs/             # TensorBoard logs
```

---

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | PyTorch 2.4.1, torchvision 0.19.1 |
| Models | timm 1.0.7 (Swin), torchvision (EfficientNet) |
| Augmentation | albumentations 1.4.18 |
| ML Utilities | scikit-learn 1.5.2, numpy 2.1.1, pandas 2.2.2 |
| Visualization | matplotlib 3.9.2, seaborn 0.13.2 |
| App Framework | streamlit 1.38.0 |
| Logging | tensorboard 2.17.1 |
| Progress | tqdm 4.66.5 |

---

## 8. Key Design Decisions

1. **Dual-Backbone Fusion**: Combines Swin Transformer's global attention features with EfficientNet's local convolutional features.
2. **View-Invariant Augmentation**: Heavy augmentation (flip, rotate90, transpose) ensures robustness to different MRI scan orientations (axial/sagittal/coronal).
3. **Differential Learning Rates**: Protects pretrained backbone weights with lower LR while allowing faster convergence in the fusion head.
4. **Inverse Class Weighting**: Addresses dataset imbalance by weighting loss inversely proportional to class frequency.
5. **Stratified 4-Fold CV + Holdout**: Ensures reliable performance estimation with a final unbiased test evaluation.
6. **Albumentations over Torchvision**: Better performance and more augmentation options for medical imaging.

---

*Generated from project analysis - reflects current state after edits.*

