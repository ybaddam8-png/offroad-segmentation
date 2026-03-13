"""
SegFormer-B4 — FINAL PUSH SCRIPT
=================================
- Continues from best checkpoint (val mIoU 0.5935)
- CLAHE + HueSaturationValue + RandomShadow augmentations
- Boosted weights: Lush Bushes 2.0, Rocks 2.5
- Lower lr: 1e-5 for precise fine-tuning
- Prints per-class IoU every epoch especially Lush Bushes + Rocks
- TTA validation at the end for best possible score
- 5 epochs — fits in 4 hours
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

plt.switch_backend('Agg')


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'train_dir':  r'C:\Users\YASHWANTH\offroad_segmentation\Offroad_Segmentation_Training_Dataset\train',
    'val_dir':    r'C:\Users\YASHWANTH\offroad_segmentation\Offroad_Segmentation_Training_Dataset\val',
    'pretrained': 'nvidia/segformer-b4-finetuned-ade-512-512',

    'num_epochs':  10,
    'batch_size':  2,
    'lr':          1e-5,       # lower lr for precise fine-tuning
    'img_size':    512,
    'num_classes': 10,

    'focal_weight': 0.5,
    'dice_weight':  0.5,
    'focal_gamma':  2.0,

    'output_dir':      r'C:\Users\YASHWANTH\offroad_segmentation\segformer_output',
    'model_save_path': r'C:\Users\YASHWANTH\offroad_segmentation\segformer_output\best_model.pth',
}


# ============================================================================
# Class Definitions
# ============================================================================

VALUE_MAP = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    700:   6,
    800:   7,
    7100:  8,
    10000: 9
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = np.array([
    [0,   0,   0  ],
    [34,  139, 34 ],
    [0,   255, 0  ],
    [210, 180, 140],
    [139, 90,  43 ],
    [128, 128, 0  ],
    [139, 69,  19 ],
    [128, 128, 128],
    [160, 82,  45 ],
    [135, 206, 235],
], dtype=np.uint8)

CLASS_WEIGHTS = torch.tensor([
    0.5,   # Background
    1.0,   # Trees
    2.0,   # Lush Bushes  — boosted
    1.0,   # Dry Grass
    1.0,   # Dry Bushes
    2.0,   # Ground Clutter
    3.0,   # Logs
    2.5,   # Rocks        — boosted
    0.8,   # Landscape
    0.5,   # Sky
], dtype=torch.float32)


# ============================================================================
# Mask Conversion
# ============================================================================

def convert_mask(mask_array):
    new_arr = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_value, class_id in VALUE_MAP.items():
        new_arr[mask_array == raw_value] = class_id
    return new_arr


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(CLASS_NAMES)):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


# ============================================================================
# Augmentation — FULL pipeline with CLAHE + HSV + RandomShadow
# ============================================================================

def get_train_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ], p=0.8),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.RandomShadow(p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Dataset
# ============================================================================

class OffRoadDataset(Dataset):
    def __init__(self, data_dir, img_size=512, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.transform = get_train_augmentation(img_size) if augment else get_val_augmentation(img_size)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image   = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, data_id)), cv2.COLOR_BGR2RGB)
        mask    = convert_mask(np.array(Image.open(os.path.join(self.mask_dir, data_id))))
        t       = self.transform(image=image, mask=mask)
        return t['image'], t['mask'].long()


# ============================================================================
# Losses
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes=10, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        if targets.shape[1] != H or targets.shape[2] != W:
            targets = F.interpolate(
                targets.unsqueeze(1).float(), size=(H, W), mode='nearest'
            ).squeeze(1).long()
        probs           = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        intersection    = (probs * targets_one_hot).sum(dim=(2, 3))
        cardinality     = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score      = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss    = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=10, focal_weight=0.5, dice_weight=0.5,
                 gamma=2.0, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal        = FocalLoss(gamma=gamma, weight=class_weights)
        self.dice         = DiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        H, W = targets.shape[1], targets.shape[2]
        if logits.shape[2] != H or logits.shape[3] != W:
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return self.focal_weight * self.focal(logits, targets) + \
               self.dice_weight  * self.dice(logits, targets)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(logits, targets, num_classes=10):
    H, W = targets.shape[1], targets.shape[2]
    if logits.shape[2] != H or logits.shape[3] != W:
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    pred   = torch.argmax(logits, dim=1).view(-1)
    target = targets.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        iou_per_class.append(float('nan') if union == 0 else (inter / union).item())
    return np.nanmean(iou_per_class), iou_per_class


def compute_pixel_accuracy(logits, targets):
    H, W = targets.shape[1], targets.shape[2]
    if logits.shape[2] != H or logits.shape[3] != W:
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    return (torch.argmax(logits, dim=1) == targets).float().mean().item()


# ============================================================================
# TTA Prediction
# ============================================================================

def tta_predict(model, imgs, num_classes, device):
    H, W = imgs.shape[2], imgs.shape[3]
    with torch.no_grad():
        # Original
        logits = model(pixel_values=imgs).logits
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs  = F.softmax(logits.float(), dim=1)

        # Horizontal flip
        imgs_hf = torch.flip(imgs, dims=[3])
        logits  = model(pixel_values=imgs_hf).logits
        logits  = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs  += torch.flip(F.softmax(logits.float(), dim=1), dims=[3])

        # Vertical flip
        imgs_vf = torch.flip(imgs, dims=[2])
        logits  = model(pixel_values=imgs_vf).logits
        logits  = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs  += torch.flip(F.softmax(logits.float(), dim=1), dims=[2])

        # Both flips
        imgs_hvf = torch.flip(imgs, dims=[2, 3])
        logits   = model(pixel_values=imgs_hvf).logits
        logits   = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        probs   += torch.flip(F.softmax(logits.float(), dim=1), dims=[2, 3])

    return probs / 4.0


# ============================================================================
# Plotting & Saving
# ============================================================================

def save_training_plots(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(history['train_loss'], label='Train')
    axes[0,0].plot(history['val_loss'],   label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(history['train_iou'], label='Train')
    axes[0,1].plot(history['val_iou'],   label='Val')
    axes[0,1].set_title('Mean IoU'); axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,0].plot(history['train_acc'], label='Train')
    axes[1,0].plot(history['val_acc'],   label='Val')
    axes[1,0].set_title('Pixel Accuracy'); axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].plot(history['lr'], color='orange')
    axes[1,1].set_title('Learning Rate'); axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print("Saved training curves")


def save_per_class_iou(class_iou_list, mean_iou, output_dir):
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in class_iou_list]
    colors    = [COLOR_PALETTE[i] / 255.0 for i in range(len(CLASS_NAMES))]
    fig, ax   = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(CLASS_NAMES)), valid_iou, color=colors, edgecolor='black')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
    ax.set_title(f'Per-Class IoU — Mean: {mean_iou:.4f}')
    ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.4f}')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
    plt.close()
    print("Saved per-class IoU chart")


def save_metrics_to_file(history, best_val_iou, best_epoch,
                         best_class_iou, tta_iou, tta_class_iou, output_dir):
    path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SEGFORMER-B4 — FINAL PUSH RUN\n")
        f.write("CLAHE + HSV + RandomShadow | Lush Bushes 2.0 | Rocks 2.5\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Val mIoU (no TTA) : {best_val_iou:.4f}  (Epoch {best_epoch})\n")
        f.write(f"Best Val mIoU (TTA)    : {tta_iou:.4f}\n")
        f.write(f"Baseline mIoU          : 0.2478\n")
        f.write(f"Improvement            : +{tta_iou - 0.2478:.4f}\n\n")
        f.write("Per-Class IoU (TTA):\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(CLASS_NAMES, tta_class_iou):
            f.write(f"  {name:<20}: {f'{iou:.4f}' if not np.isnan(iou) else 'N/A'}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-Epoch History:\n")
        f.write(f"{'Epoch':<8}{'TrainLoss':<12}{'ValLoss':<12}"
                f"{'TrainIoU':<12}{'ValIoU':<12}{'TrainAcc':<12}{'ValAcc':<12}\n")
        f.write("-" * 78 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8}"
                    f"{history['train_loss'][i]:<12.4f}"
                    f"{history['val_loss'][i]:<12.4f}"
                    f"{history['train_iou'][i]:<12.4f}"
                    f"{history['val_iou'][i]:<12.4f}"
                    f"{history['train_acc'][i]:<12.4f}"
                    f"{history['val_acc'][i]:<12.4f}\n")
    print(f"Saved metrics to {path}")


# ============================================================================
# Train / Validate
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, ious, accs = [], [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values=imgs).logits
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            iou, _ = compute_iou(logits.detach(), masks)
            acc    = compute_pixel_accuracy(logits.detach(), masks)

        losses.append(loss.item()); ious.append(iou); accs.append(acc)
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    return np.mean(losses), np.nanmean(ious), np.mean(accs)


def validate(model, loader, criterion, device):
    model.eval()
    losses, ious, accs, all_class_iou = [], [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            logits      = model(pixel_values=imgs).logits
            loss        = criterion(logits, masks)
            iou, c_iou  = compute_iou(logits, masks)
            acc         = compute_pixel_accuracy(logits, masks)
            losses.append(loss.item()); ious.append(iou)
            accs.append(acc); all_class_iou.append(c_iou)
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    return np.mean(losses), np.nanmean(ious), np.mean(accs), np.nanmean(all_class_iou, axis=0)


def validate_with_tta(model, loader, device):
    model.eval()
    ious, all_class_iou = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="TTA Validation", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            probs       = tta_predict(model, imgs, CONFIG['num_classes'], device)
            iou, c_iou  = compute_iou(probs, masks)
            ious.append(iou); all_class_iou.append(c_iou)
            pbar.set_postfix(iou=f"{iou:.4f}")

    return np.nanmean(ious), np.nanmean(all_class_iou, axis=0)


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)} GB")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("\nLoading datasets...")
    train_dataset = OffRoadDataset(CONFIG['train_dir'], img_size=CONFIG['img_size'], augment=True)
    val_dataset   = OffRoadDataset(CONFIG['val_dir'],   img_size=CONFIG['img_size'], augment=False)
    train_loader  = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True,  num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                               shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    print("\nLoading SegFormer-B4...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        CONFIG['pretrained'],
        num_labels=CONFIG['num_classes'],
        ignore_mismatched_sizes=True,
    )

    if os.path.exists(CONFIG['model_save_path']):
        print(f"Checkpoint found — loading from {CONFIG['model_save_path']}")
        model.load_state_dict(torch.load(CONFIG['model_save_path'], weights_only=True))
        print("Checkpoint loaded! Continuing from val mIoU 0.5935...")
    else:
        print("No checkpoint — starting from ADE20K weights")

    model = model.to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — Total: {total:,} | Trainable: {trainable:,}")

    criterion = CombinedLoss(
        num_classes=CONFIG['num_classes'],
        focal_weight=CONFIG['focal_weight'],
        dice_weight=CONFIG['dice_weight'],
        gamma=CONFIG['focal_gamma'],
        class_weights=CLASS_WEIGHTS.to(device),
    )

    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-7)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_acc':  [], 'val_acc':  [],
        'lr': []
    }
    best_val_iou   = 0.5935   # start from known best
    best_epoch     = 0
    best_class_iou = None

    print(f"\nStarting FINAL PUSH — {CONFIG['num_epochs']} epochs | lr={CONFIG['lr']}")
    print("=" * 70)

    for epoch in range(CONFIG['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]  LR: {current_lr:.2e}")

        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_iou, val_acc, class_iou = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Train — Loss: {train_loss:.4f}  mIoU: {train_iou:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}  mIoU: {val_iou:.4f}  Acc: {val_acc:.4f}")
        print(f"  ── Per-Class Highlights ──")
        print(f"     Lush Bushes : {class_iou[2]:.4f}")
        print(f"     Rocks       : {class_iou[7]:.4f}")
        print(f"     Trees       : {class_iou[1]:.4f}")
        print(f"     Sky         : {class_iou[9]:.4f}")

        if val_iou > best_val_iou:
            best_val_iou   = val_iou
            best_epoch     = epoch + 1
            best_class_iou = class_iou
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"  ✅ New best! Saved — Val mIoU: {best_val_iou:.4f}")
        else:
            print(f"  ── No improvement (best: {best_val_iou:.4f})")

    # ---- TTA on best model ----
    print(f"\nRunning TTA on best model...")
    if best_class_iou is not None:
        model.load_state_dict(torch.load(CONFIG['model_save_path'], weights_only=True))
        model = model.to(device)
    tta_iou, tta_class_iou = validate_with_tta(model, val_loader, device)

    # ---- Final Results ----
    print("\n" + "=" * 70)
    print("FINAL PUSH COMPLETE")
    print("=" * 70)
    if best_class_iou is not None:
        print(f"Best Val mIoU (no TTA) : {best_val_iou:.4f}  (Epoch {best_epoch})")
    else:
        print(f"No improvement over 0.5935 during this run")
    print(f"Best Val mIoU (TTA)    : {tta_iou:.4f}")
    print(f"Baseline mIoU          : 0.2478")
    print(f"Total Improvement      : +{tta_iou - 0.2478:.4f}")
    print("\nPer-Class IoU (TTA):")
    print("-" * 40)
    for name, iou in zip(CLASS_NAMES, tta_class_iou):
        status = "✅" if not np.isnan(iou) and iou > 0.3 else "⚠️"
        print(f"  {status} {name:<20}: {f'{iou:.4f}' if not np.isnan(iou) else 'N/A'}")

    if best_class_iou is None:
        best_class_iou = tta_class_iou

    save_training_plots(history, CONFIG['output_dir'])
    save_per_class_iou(tta_class_iou, tta_iou, CONFIG['output_dir'])
    save_metrics_to_file(history, best_val_iou, best_epoch,
                         best_class_iou, tta_iou, tta_class_iou, CONFIG['output_dir'])
    print(f"\nAll outputs saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()