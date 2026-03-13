"""
Push SegFormer-B4 Offroad Segmentation Model to HuggingFace Hub
"""

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from huggingface_hub import HfApi
import json
import os

# ============================================================
# CHANGE THIS to your HuggingFace username
HF_USERNAME = "rohan9977"
REPO_NAME   = "segformer-b4-offroad-segmentation"
# ============================================================

MODEL_PATH  = r'C:\Users\YASHWANTH\offroad_segmentation\segformer_output\best_model.pth'
PRETRAINED  = 'nvidia/segformer-b4-finetuned-ade-512-512'
NUM_CLASSES = 10

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = [
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
]

id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location='cpu'))
model.eval()
print("Model loaded!")

repo_id = f"{HF_USERNAME}/{REPO_NAME}"
print(f"\nPushing to: https://huggingface.co/{repo_id}")

model.push_to_hub(repo_id)
print("Model pushed!")

processor = SegformerImageProcessor(
    image_scale=(512, 512),
    do_resize=True,
    do_normalize=True,
)
processor.push_to_hub(repo_id)
print("Processor pushed!")

model_card = f"""---
license: apache-2.0
tags:
- segformer
- semantic-segmentation
- offroad
- autonomous-vehicles
- desert
- pytorch
datasets:
- duality-ai-falcon-simulation
metrics:
- mean_iou
---

# SegFormer-B4 — Offroad Semantic Segmentation

Fine-tuned SegFormer-B4 for semantic segmentation of offroad desert scenes,
trained on synthetic data from Duality AI's Falcon simulation platform.

## Model Details
- **Architecture:** SegFormer-B4 (MiT-B4 backbone)
- **Parameters:** 64M
- **Pretrained on:** ADE20K (150 classes)
- **Fine-tuned on:** Duality AI Falcon synthetic desert images
- **Input size:** 512×512
- **Classes:** {NUM_CLASSES}

## Performance
| Metric | Score |
|--------|-------|
| Val mIoU | 0.5935 |
| Baseline mIoU | 0.2478 |
| Improvement | +0.3457 |
| Val Pixel Accuracy | 86.17% |

## Classes
| ID | Class | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Trees | Dark Green |
| 2 | Lush Bushes | Bright Green |
| 3 | Dry Grass | Tan |
| 4 | Dry Bushes | Brown |
| 5 | Ground Clutter | Olive |
| 6 | Logs | Dark Brown |
| 7 | Rocks | Gray |
| 8 | Landscape | Sandy Brown |
| 9 | Sky | Light Blue |

## Training Details
- **Loss:** Combined Dice + Focal Loss
- **Optimizer:** AdamW (lr=3e-5, wd=0.01)
- **Scheduler:** CosineAnnealingLR
- **Augmentation:** HorizontalFlip, ColorJitter, CLAHE, HueSaturationValue, RandomFog
- **Epochs:** 10 (Phase 1) + 10 (Phase 2)
- **Hardware:** NVIDIA RTX 4050 6GB

## Usage
```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import torch

processor = SegformerImageProcessor.from_pretrained("{repo_id}")
model = SegformerForSemanticSegmentation.from_pretrained("{repo_id}")

image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_mask = outputs.logits.argmax(dim=1)
```
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)
print("Model card pushed!")

print(f"\n✅ DONE! Your model is live at:")
print(f"   https://huggingface.co/{repo_id}")
