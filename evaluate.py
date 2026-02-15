import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pickle

# =========================================================
# 1. CONFIGURATION (Matches Boosted Training)
# =========================================================
TEST_ROOT = r'C:\Users\DANIYAL\Desktop\study\iba datathon\Offroad_Segmentation_Training_Dataset\test'
MODEL_PATH = './model.pkl'
SAVE_VIS_DIR = './boosted_eval_results'

CLASS_NAMES = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", 
               "Flowers", "Logs", "Rocks", "Landscape", "Sky"]

# Professional Color Map for Report Visuals
COLOR_MAP = np.array([
    [0, 255, 0],    # Trees
    [0, 128, 0],    # Lush Bushes
    [255, 255, 0],  # Dry Grass
    [139, 69, 19],  # Dry Bushes
    [128, 128, 128],# Ground Clutter
    [255, 0, 255],  # Flowers
    [160, 82, 45],  # Logs
    [105, 105, 105],# Rocks
    [210, 180, 140],# Landscape
    [135, 206, 235] # Sky
])

# =========================================================
# 2. DATASET & HELPERS
# =========================================================
class DualityEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images_dir = os.path.join(root_dir, 'color_images')
        self.masks_dir = os.path.join(root_dir, 'segmentation')
        self.filenames = sorted(os.listdir(self.images_dir))
        self.transform = transform
        self.class_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.images_dir, self.filenames[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, self.filenames[idx]), cv2.IMREAD_UNCHANGED)
        target = np.zeros(mask.shape[:2], dtype=np.int64)
        for old_id, new_id in self.class_map.items():
            target[mask == old_id] = new_id
        if self.transform:
            aug = self.transform(image=img, mask=target)
            img, target = aug['image'], aug['mask']
        return img, target, self.filenames[idx]

def denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    return np.clip(std * img + mean, 0, 1)

# =========================================================
# 3. EVALUATION RUN
# =========================================================
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_VIS_DIR, exist_ok=True)

    # CRITICAL: Architecture must match training (UnetPlusPlus + ResNet50)
    model = smp.UnetPlusPlus(encoder_name="resnet50", classes=10).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found. Train the model first!")
        return
    
    with open(MODEL_PATH, "rb") as f:
        state_dict = pickle.load(f)

    model.load_state_dict(state_dict)
    model.eval()

    test_tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    loader = DataLoader(DualityEvalDataset(TEST_ROOT, test_tf), batch_size=1)
    
    tp_list, fp_list, fn_list, tn_list = [], [], [], []

    print(f"🔬 Evaluating Boosted Model on {len(loader)} images...")

    with torch.no_grad():
        for i, (image, mask, filename) in enumerate(loader):
            image, mask = image.to(device), mask.to(device).long()
            pred = model(image).argmax(dim=1)

            # Store stats for global calculation
            s = smp.metrics.get_stats(pred, mask, mode='multiclass', num_classes=10)
            tp_list.append(s[0]); fp_list.append(s[1]); fn_list.append(s[2]); tn_list.append(s[3])

            # Save Visual Comparison for first 15 images
            if i < 15:
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                ax[0].imshow(denormalize(image[0])); ax[0].set_title("Actual Photo")
                ax[1].imshow(COLOR_MAP[mask[0].cpu()]); ax[1].set_title("Target (Ground Truth)")
                ax[2].imshow(COLOR_MAP[pred[0].cpu()]); ax[2].set_title("AI Prediction")
                for a in ax: a.axis('off')
                plt.savefig(f"{SAVE_VIS_DIR}/sample_{i}.png", bbox_inches='tight')
                plt.close()

    # Calculate Global Metrics
    tp = torch.cat(tp_list, dim=0).sum(dim=0)
    fp = torch.cat(fp_list, dim=0).sum(dim=0)
    fn = torch.cat(fn_list, dim=0).sum(dim=0)
    tn = torch.cat(tn_list, dim=0).sum(dim=0)

    per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

    print("\n" + "🚀 BOOSTED COMPETITION RESULTS " + "="*20)
    print(f"{'CLASS NAME':<25} | {'IOU SCORE':<10}")
    print("-" * 40)
    for name, score in zip(CLASS_NAMES, per_class_iou):
        print(f"{name:<25} | {score.item():.4f}")
    print("-" * 40)
    print(f"{'FINAL MEAN IoU SCORE':<25} | {per_class_iou.mean().item():.4f}")
    print("="*50)
    print(f"Visual results saved to: {os.path.abspath(SAVE_VIS_DIR)}")

if __name__ == "__main__":
    run_evaluation()