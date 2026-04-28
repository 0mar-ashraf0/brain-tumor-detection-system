import torch
import torch.nn as nn
import timm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import SWIN_MODEL_NAME, EFFNET_MODEL_NAME, SWIN_FEATURES, EFFNET_FEATURES, NUM_CLASSES, DEVICE

class HybridTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Swin Transformer (tiny for speed)
        self.swin = timm.create_model(SWIN_MODEL_NAME, pretrained=True, num_classes=0, global_pool='avg')
        
        # EfficientNet-B0
        self.effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.effnet.classifier = nn.Identity()
        
        # Fusion
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion_fc = nn.Sequential(
            nn.Linear(SWIN_FEATURES + EFFNET_FEATURES, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x):
        swin_feat = self.swin(x)
        eff_feat = self.effnet.features(x)
        eff_feat = self.fusion_pool(eff_feat).flatten(1)
        fused = torch.cat([swin_feat, eff_feat], dim=1)
        return self.fusion_fc(fused)

def load_model(model_path):
    model = HybridTumorModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

