import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# =========================================================
# 1. CONFIGURATION
# =========================================================
# IMPORTANT: Use 'val' if you want to see a score. Use 'test' only if it has ground truth masks.
TEST_ROOT = r'C:\Users\DANIYAL\Desktop\study\iba datathon\Offroad_Segmentation_Training_Dataset\test'
MODEL_WEIGHTS = 'segformer_big_model-Copy1.pth' 
SAVE_VIS_DIR = './segformer_big_results'
os.makedirs(SAVE_VIS_DIR, exist_ok=True)

# MUST MATCH YOUR TRAINING BACKBONE: "nvidia/mit-b2" or "nvidia/mit-b5"
MODEL_NAME = "nvidia/mit-b2" 

CLASS_NAMES = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", 
               "Flowers", "Logs", "Rocks", "Landscape", "Sky"]

COLOR_MAP = np.array([
    [0, 255, 0], [0, 128, 0], [255, 255, 0], [139, 69, 19], [128, 128, 128],
    [255, 0, 255], [160, 82, 45], [105, 105, 105], [210, 180, 140], [135, 206, 235]
])

# =========================================================
# 2. DATASET & HELPERS
# =========================================================
class SegformerEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, 'color_images')
        self.msk_dir = os.path.join(root_dir, 'segmentation')
        self.filenames = sorted(os.listdir(self.img_dir))
        self.transform = transform
        self.class_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.filenames[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.msk_dir, self.filenames[idx]), cv2.IMREAD_UNCHANGED)
        target = np.zeros(mask.shape[:2], dtype=np.int64)
        for old_id, new_id in self.class_map.items(): target[mask == old_id] = new_id
        if self.transform:
            aug = self.transform(image=img, mask=target)
            img, target = aug['image'], aug['mask']
        return img, target, self.filenames[idx]

def denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    return np.clip(std * img + mean, 0, 1)

# =========================================================
# 3. EVALUATION FUNCTION
# =========================================================
def evaluate_segformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Architecture (Updated to Big Model)
    print(f"Initializing SegFormer Architecture: {MODEL_NAME}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME, num_labels=10, ignore_mismatched_sizes=True
    )
    
    # 2. Load Weights
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"Loaded weights from {MODEL_WEIGHTS}")
    else:
        print(f"Error: {MODEL_WEIGHTS} not found!")
        return

    model.to(device).eval()

    test_tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    loader = DataLoader(SegformerEvalDataset(TEST_ROOT, test_tf), batch_size=1)
    
    tp_l, fp_l, fn_l, tn_l = [], [], [], []

    print(f"🔬 Evaluating Big Model on {len(loader)} images...")

    with torch.no_grad():
        for i, (image, mask, filename) in enumerate(loader):
            image, mask = image.to(device), mask.to(device).long()
            
            # Prediction
            outputs = model(pixel_values=image)
            
            # UPSAMPLE: Segformer-B2/B5 still outputs at 1/4 resolution (128x128)
            logits = nn.functional.interpolate(
                outputs.logits, 
                size=mask.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            pred = logits.argmax(dim=1)

            # Metrics calculation per image
            s = smp.metrics.get_stats(pred, mask, mode='multiclass', num_classes=10)
            tp_l.append(s[0]); fp_l.append(s[1]); fn_l.append(s[2]); tn_l.append(s[3])

            # Save Visual Comparison (first 10)
            if i < 10:
                plt.figure(figsize=(15, 5))
                plt.subplot(1,3,1); plt.imshow(denormalize(image[0])); plt.title("Original Photo"); plt.axis('off')
                plt.subplot(1,3,2); plt.imshow(COLOR_MAP[mask[0].cpu()]); plt.title("Ground Truth"); plt.axis('off')
                plt.subplot(1,3,3); plt.imshow(COLOR_MAP[pred[0].cpu()]); plt.title(f"{MODEL_NAME} Prediction"); plt.axis('off')
                plt.savefig(f"{SAVE_VIS_DIR}/eval_{filename[0]}.png", bbox_inches='tight')
                plt.close()

    # 4. Final Dataset-wide Calculation
    tp = torch.cat(tp_l, dim=0).sum(dim=0)
    fp = torch.cat(fp_l, dim=0).sum(dim=0)
    fn = torch.cat(fn_l, dim=0).sum(dim=0)
    tn = torch.cat(tn_l, dim=0).sum(dim=0)

    per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

    print("\n" + f"BIG SEGFORMER ({MODEL_NAME}) RESULTS " + "="*20)
    print(f"{'CLASS NAME':<25} | {'IOU SCORE':<10}")
    print("-" * 40)
    for name, score in zip(CLASS_NAMES, per_class_iou):
        print(f"{name:<25} | {score.item():.4f}")
    print("-" * 40)
    print(f"{'FINAL MEAN IoU SCORE':<25} | {per_class_iou.mean().item():.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_segformer()