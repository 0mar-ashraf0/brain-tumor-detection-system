import os
import torch
import numpy as np

# Paths
BASE_DIR = r'c:\Users\ashra\OneDrive\Desktop\MRI'
DATA_DIR = DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'Epic and CSCR hospital Dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'Train')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
PROJECT_DIR = os.path.join(BASE_DIR, 'brain_tumor_detector')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
MODEL_PATH = os.path.join(PROJECT_DIR, 'best_model.pth')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Classes
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

# Model config
IMG_SIZE = 224
SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224'
EFFNET_MODEL_NAME = 'efficientnet_b0'
SWIN_FEATURES = 768
EFFNET_FEATURES = 1280

# Training config
BATCH_SIZE = 16
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

LR_BACKBONE = 1e-5  # Slower rate to protect pre-trained weights
LR_HEAD = 1e-3      # Faster rate to train the new fusion layer
WEIGHT_DECAY = 1e-4 # Regularization

CV_FOLDS = 4
TEST_SPLIT_SIZE = 0.1  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


