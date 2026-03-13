"""
SegFormer-B4 Test Script — TTA + Multi-Scale Inference
=======================================================
- Test Time Augmentation (original + h-flip + v-flip + both flips)
- Multi-scale inference (448, 512, 576)
- Averages all predictions for best possible test mIoU
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    'test_dir':    r'C:\Users\YASHWANTH\offroad_segmentation\Offroad_Segmentation_testImages',
    'pretrained':  'nvidia/segformer-b4-finetuned-ade-512-512',
    'model_path':  r'C:\Users\YASHWANTH\offroad_segmentation\segformer_output\best_model.pth',
    'output_dir':  r'C:\Users\YASHWANTH\offroad_segmentation\segformer_output\test_results',
    'num_classes': 10,
    'batch_size':  1,           # batch 1 for multi-scale TTA
    'scales':      [448, 512, 576],  # multi-scale inference
    'num_samples': 10,
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
# Dataset
# ============================================================================

def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class TestDataset(Dataset):
    def __init__(self, test_dir, img_size=512):
        self.image_dir = os.path.join(test_dir, 'Color_Images')
        self.mask_dir  = os.path.join(test_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.img_size  = img_size
        self.transform = get_transform(img_size)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id  = self.data_ids[idx]
        image    = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, data_id)), cv2.COLOR_BGR2RGB)
        mask     = convert_mask(np.array(Image.open(os.path.join(self.mask_dir, data_id))))
        t        = self.transform(image=image, mask=mask)
        return t['image'], t['mask'].long(), data_id


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred   = pred.view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        iou_per_class.append(float('nan') if union == 0 else (inter / union).item())
    return np.nanmean(iou_per_class), iou_per_class


def compute_pixel_accuracy(pred, target):
    return (pred == target).float().mean().item()


# ============================================================================
# TTA + Multi-Scale Prediction
# ============================================================================

def predict_with_tta_multiscale(model, image_path, device, scales, num_classes):
    """
    For a single image:
    - Run at each scale in scales list
    - At each scale run 4 TTA augmentations (original, hflip, vflip, hvflip)
    - Average all predictions
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W  = image.shape[:2]

    all_probs = None
    count     = 0

    for scale in scales:
        transform = get_transform(scale)
        img_t     = transform(image=image)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            # Original
            logits = model(pixel_values=img_t).logits
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs  = F.softmax(logits.float(), dim=1)

            # H-flip
            img_hf  = torch.flip(img_t, dims=[3])
            logits  = model(pixel_values=img_hf).logits
            logits  = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs  += torch.flip(F.softmax(logits.float(), dim=1), dims=[3])

            # V-flip
            img_vf  = torch.flip(img_t, dims=[2])
            logits  = model(pixel_values=img_vf).logits
            logits  = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs  += torch.flip(F.softmax(logits.float(), dim=1), dims=[2])

            # HV-flip
            img_hvf  = torch.flip(img_t, dims=[2, 3])
            logits   = model(pixel_values=img_hvf).logits
            logits   = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            probs   += torch.flip(F.softmax(logits.float(), dim=1), dims=[2, 3])

        # 4 augmentations per scale
        if all_probs is None:
            all_probs = probs
        else:
            all_probs += probs
        count += 4

    return all_probs / count  # average over all scales x augmentations


# ============================================================================
# Saving Results
# ============================================================================

def save_per_class_iou_chart(class_iou, mean_iou, output_dir):
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in class_iou]
    colors    = [COLOR_PALETTE[i] / 255.0 for i in range(len(CLASS_NAMES))]
    fig, ax   = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(CLASS_NAMES)), valid_iou, color=colors, edgecolor='black')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
    ax.set_title(f'Per-Class IoU (TTA + Multi-Scale) — Mean: {mean_iou:.4f}')
    ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.4f}')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_per_class_iou.png'), dpi=150)
    plt.close()


def save_metrics(mean_iou, pixel_acc, class_iou, output_dir):
    # Corrected 7-class mIoU (removing absent classes: Background, Ground Clutter, Logs)
    present_classes = [1, 2, 3, 4, 7, 8, 9]  # Trees, Lush Bushes, Dry Grass, Dry Bushes, Rocks, Landscape, Sky
    corrected_iou   = np.nanmean([class_iou[i] for i in present_classes])

    path = os.path.join(output_dir, 'test_evaluation_metrics.txt')
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEST RESULTS — TTA + MULTI-SCALE INFERENCE\n")
        f.write(f"Scales: {CONFIG['scales']} | TTA: 4 augmentations per scale\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Raw Test mIoU (10 classes)    : {mean_iou:.4f}\n")
        f.write(f"Corrected mIoU (7 classes)    : {corrected_iou:.4f}\n")
        f.write(f"Pixel Accuracy                : {pixel_acc:.4f}\n")
        f.write(f"Baseline mIoU                 : 0.2478\n")
        f.write(f"Improvement (corrected)       : +{corrected_iou - 0.2478:.4f}\n\n")
        f.write("Note: Background(0), Ground Clutter(550), Logs(700)\n")
        f.write("      are absent from test set — excluded from corrected mIoU\n\n")
        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(CLASS_NAMES, class_iou):
            present = "" if name in ['Background', 'Ground Clutter', 'Logs'] else "✓"
            f.write(f"  {present} {name:<20}: {f'{iou:.4f}' if not np.isnan(iou) else 'N/A (absent)'}\n")

    print(f"\nSaved metrics to {path}")
    return corrected_iou


def save_comparison_samples(model, test_dataset, device, output_dir, num_samples=10):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)

    image_dir = os.path.join(CONFIG['test_dir'], 'Color_Images')

    for i in range(min(num_samples, len(test_dataset))):
        img_tensor, mask_tensor, data_id = test_dataset[i]
        image_path = os.path.join(image_dir, data_id)

        probs = predict_with_tta_multiscale(
            model, image_path, device, CONFIG['scales'], CONFIG['num_classes']
        )
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

        # Resize mask to match pred
        H, W = pred.shape
        mask_resized = cv2.resize(
            mask_tensor.numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img);                              axes[0].set_title('Input');        axes[0].axis('off')
        axes[1].imshow(mask_to_color(mask_resized));      axes[1].set_title('Ground Truth'); axes[1].axis('off')
        axes[2].imshow(mask_to_color(pred.astype(np.uint8))); axes[2].set_title('Prediction (TTA+MS)'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparisons', f'sample_{i+1}.png'), dpi=150)
        plt.close()

    print(f"Saved {num_samples} comparison samples")


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Scales : {CONFIG['scales']}")
    print(f"TTA    : 4 augmentations per scale")
    print(f"Total  : {len(CONFIG['scales']) * 4} predictions averaged per image")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Load model
    print(f"\nLoading SegFormer-B4 from {CONFIG['model_path']}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        CONFIG['pretrained'],
        num_labels=CONFIG['num_classes'],
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(CONFIG['model_path'], weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Load test dataset at base scale 512
    test_dataset = TestDataset(CONFIG['test_dir'], img_size=512)
    print(f"Test images: {len(test_dataset)}")

    # Run TTA + Multi-Scale inference
    print(f"\nRunning TTA + Multi-Scale inference...")
    all_ious, all_accs, all_class_iou = [], [], []

    image_dir = os.path.join(CONFIG['test_dir'], 'Color_Images')

    pbar = tqdm(test_dataset, desc="Testing")
    for img_tensor, mask_tensor, data_id in pbar:
        image_path = os.path.join(image_dir, data_id)

        probs = predict_with_tta_multiscale(
            model, image_path, device, CONFIG['scales'], CONFIG['num_classes']
        )

        # Resize mask to match prediction size
        H, W  = mask_tensor.shape
        pred  = torch.argmax(probs, dim=1).squeeze(0)
        pred  = F.interpolate(
            pred.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze().long()

        iou, c_iou = compute_iou(pred.cpu(), mask_tensor.cpu())
        acc        = compute_pixel_accuracy(pred.cpu(), mask_tensor.cpu())
        all_ious.append(iou)
        all_accs.append(acc)
        all_class_iou.append(c_iou)
        pbar.set_postfix(iou=f"{iou:.4f}")

    mean_iou      = np.nanmean(all_ious)
    pixel_acc     = np.mean(all_accs)
    class_iou_avg = np.nanmean(all_class_iou, axis=0)

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS — TTA + MULTI-SCALE")
    print("=" * 60)
    print(f"Raw Test mIoU (10 classes) : {mean_iou:.4f}")

    present_classes = [1, 2, 3, 4, 7, 8, 9]
    corrected_iou   = np.nanmean([class_iou_avg[i] for i in present_classes])
    print(f"Corrected mIoU (7 classes) : {corrected_iou:.4f}")
    print(f"Pixel Accuracy             : {pixel_acc:.4f}")
    print(f"Baseline mIoU              : 0.2478")
    print(f"Improvement                : +{corrected_iou - 0.2478:.4f}")
    print("\nPer-Class IoU:")
    print("-" * 40)
    for name, iou in zip(CLASS_NAMES, class_iou_avg):
        print(f"  {name:<20}: {f'{iou:.4f}' if not np.isnan(iou) else 'N/A'}")

    save_per_class_iou_chart(class_iou_avg, mean_iou, CONFIG['output_dir'])
    save_metrics(mean_iou, pixel_acc, class_iou_avg, CONFIG['output_dir'])
    save_comparison_samples(model, test_dataset, device, CONFIG['output_dir'])

    print(f"\nAll outputs saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()


