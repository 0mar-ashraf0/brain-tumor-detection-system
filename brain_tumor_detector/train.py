import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from model import HybridTumorModel
from data_utils import TumorDataset, test_transform
from sklearn.model_selection import train_test_split
from collections import Counter
from config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

# View-invariant augmentation
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'checkpoint.pth')  # ✅ checkpoint path

writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, 'logs'))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    return running_loss / len(loader), acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def get_optimizer(model):
    return optim.AdamW([
        {'params': model.swin.parameters(),      'lr': LR_BACKBONE},
        {'params': model.effnet.parameters(),    'lr': LR_BACKBONE},
        {'params': model.fusion_fc.parameters(), 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, patience_counter, fold, phase):
    torch.save({
        'phase': phase,           # 'cv' or 'final'
        'fold': fold,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'patience_counter': patience_counter,
    }, CHECKPOINT_PATH)

def plot_metrics(train_losses, val_losses, train_accs, val_accs, fold, prefix='CV_Fold'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'{prefix} {fold} Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title(f'{prefix} {fold} Acc')
    plt.savefig(os.path.join(RESULTS_DIR, f'{prefix.lower()}_{fold}_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_fold_{fold}.png'))
    plt.close()

def plot_roc(y_true, y_probs, classes, fold):
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc_score(y_true == i, y_probs[:, i]):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title(f'ROC Fold {fold}')
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_fold_{fold}.png'))
    plt.close()

# 90/10 split
print('Creating 90/10 train/test split from TRAIN_DIR...')
train_ds_no_transform = TumorDataset(TRAIN_DIR, transform=None)
all_labels = np.array(train_ds_no_transform.dataset.targets)

cv_train_idx, test_idx, _, test_labels = train_test_split(
    np.arange(len(all_labels)), all_labels,
    test_size=TEST_SPLIT_SIZE, stratify=all_labels, random_state=42
)

print(f'Total: {len(all_labels)} | CV train: {len(cv_train_idx)} | Test holdout: {len(test_idx)}')

label_counts = Counter(all_labels)
print('Class distribution:', dict(sorted(label_counts.items())))
class_weights = torch.tensor([1.0 / label_counts[i] for i in range(NUM_CLASSES)]).float().to(DEVICE)
print('Class weights:', class_weights.tolist())

class WrappedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = np.array(img)
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        return img.float(), torch.tensor(label)

# ✅ Check for existing checkpoint
start_fold = 0
cv_accs = []
cv_losses = []

if os.path.exists(CHECKPOINT_PATH):
    print(f'\n✅ Checkpoint found! Auto-resuming...')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    print(f'   Phase: {ckpt["phase"]} | Fold: {ckpt["fold"]} | Epoch: {ckpt["epoch"]}')
    resume = 'y'
else:
    resume = 'n'
    ckpt = None

# 4-Fold CV
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(cv_train_idx)), all_labels[cv_train_idx])):

    # ✅ Skip completed folds if resuming
    if resume == 'y' and ckpt['phase'] == 'cv' and fold < ckpt['fold']:
        print(f'Skipping completed fold {fold + 1}')
        continue

    print(f'\n=== Fold {fold + 1} ===')
    cv_train_sub_idx = cv_train_idx[train_idx]
    cv_val_sub_idx   = cv_train_idx[val_idx]

    train_wrapped = WrappedSubset(Subset(train_ds_no_transform.dataset, cv_train_sub_idx), train_transform)
    val_wrapped   = WrappedSubset(Subset(train_ds_no_transform.dataset, cv_val_sub_idx),   test_transform)

    train_loader = DataLoader(train_wrapped, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_wrapped,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = HybridTumorModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ✅ Restore checkpoint state for current fold
    start_epoch = 0
    if resume == 'y' and ckpt['phase'] == 'cv' and fold == ckpt['fold']:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resuming fold {fold+1} from epoch {start_epoch}')

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, y_true, y_pred, y_probs = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar(f'Fold{fold}/Train_Loss', train_loss, epoch)
        writer.add_scalar(f'Fold{fold}/Val_Loss',   val_loss,   epoch)
        writer.add_scalar(f'Fold{fold}/Train_Acc',  train_acc,  epoch)
        writer.add_scalar(f'Fold{fold}/Val_Acc',    val_acc,    epoch)

        print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}')

        # ✅ Save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, scheduler, val_acc, 0, fold, 'cv')

    plot_metrics(train_losses, val_losses, train_accs, val_accs, fold + 1)
    plot_confusion_matrix(y_true, y_pred, CLASSES, f'Fold {fold+1} Confusion Matrix', fold + 1)
    plot_roc(y_true, y_probs, CLASSES, fold + 1)

    cv_accs.append(val_acc)
    cv_losses.append(val_loss)

# Final training on full 90%
print('\nFinal training on full 90% CV train set...')
cv_train_full = Subset(train_ds_no_transform.dataset, cv_train_idx)
test_sub      = Subset(train_ds_no_transform.dataset, test_idx)

train_loader_final = DataLoader(WrappedSubset(cv_train_full, train_transform), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader        = DataLoader(WrappedSubset(test_sub,      test_transform),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model     = HybridTumorModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = get_optimizer(model)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_acc    = 0.0
patience_counter = 0
start_epoch     = 0

# ✅ Resume final training phase if checkpoint exists
if resume == 'y' and ckpt['phase'] == 'final':
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    best_val_acc     = ckpt['best_val_acc']
    patience_counter = ckpt['patience_counter']
    start_epoch      = ckpt['epoch'] + 1
    print(f'Resuming final training from epoch {start_epoch}, best acc: {best_val_acc:.4f}')

for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader_final, optimizer, criterion, DEVICE)
    val_loss, val_acc, _, _, _ = validate(model, test_loader, criterion, DEVICE)
    scheduler.step()

    print(f'Final Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Holdout Acc {val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'✅ New best model saved! Val Acc {val_acc:.4f}')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # ✅ Save checkpoint every epoch
    save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, patience_counter, 0, 'final')

# Final evaluation
test_loss, test_acc, test_y_true, test_y_pred, test_y_probs = validate(model, test_loader, criterion, DEVICE)

print(f'\n=== FINAL RESULTS ===')
print(f'CV Avg Val Acc:   {np.mean(cv_accs):.4f} (+/- {np.std(cv_accs)*2:.4f})')
print(f'Test Holdout Acc: {test_acc:.4f}')
print('\nTest Classification Report:')
print(classification_report(test_y_true, test_y_pred, target_names=CLASSES))

plot_confusion_matrix(test_y_true, test_y_pred, CLASSES, '10% Test Holdout Confusion Matrix', 'holdout')
plot_roc(test_y_true, test_y_probs, CLASSES, 'holdout')

writer.close()

# ✅ Delete checkpoint after successful completion
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print('Checkpoint deleted after successful training.')

print(f'\nModel saved to {MODEL_PATH}')
print('Training complete! Check results/ folder.')