import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================================================
# 1. SETUP
# =========================================================
TRAIN_ROOT = r'C:\Users\DANIYAL\Desktop\study\iba datathon\Offroad_Segmentation_Training_Dataset\train'
VAL_ROOT = r'C:\Users\DANIYAL\Desktop\study\iba datathon\Offroad_Segmentation_Training_Dataset\val'
MODEL_PATH = 'offroad_segmentation_model.pth'

CLASS_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

class DualityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images_dir = os.path.join(root_dir, 'color_images')
        self.masks_dir = os.path.join(root_dir, 'segmentation')
        img_f = set(os.listdir(self.images_dir))
        mask_f = set(os.listdir(self.masks_dir))
        self.filenames = sorted(list(img_f.intersection(mask_f)))
        self.transform = transform

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.images_dir, self.filenames[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, self.filenames[idx]), cv2.IMREAD_UNCHANGED)
        target = np.zeros(mask.shape[:2], dtype=np.int64)
        for old_id, new_id in CLASS_MAP.items():
            target[mask == old_id] = new_id
        if self.transform:
            aug = self.transform(image=img, mask=target)
            img, target = aug['image'], aug['mask']
        return img, target

# =========================================================
# 2. TRANSFORMATIONS
# =========================================================
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])

# =========================================================
# 3. TRAINING WITH VALIDATION LOSS & IOU
# =========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model: Unet++ is better for capturing small objects (Logs/Flowers)
    model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="imagenet", classes=10).to(device)

    train_loader = DataLoader(DualityDataset(TRAIN_ROOT, train_transform), batch_size=4, shuffle=True)
    val_loader = DataLoader(DualityDataset(VAL_ROOT, val_transform), batch_size=4, shuffle=False)

    # Combined Loss: Focal for hard pixels, Dice for overlap
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')
    criterion_focal = smp.losses.FocalLoss(mode='multiclass')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    print(f" Training started on {device}...")
    best_val_iou = 0.0

    for epoch in range(50):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device).long()
            optimizer.zero_grad()
            output = model(imgs)
            loss = 0.5 * criterion_dice(output, msks) + 0.5 * criterion_focal(output, msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        tp, fp, fn, tn = [], [], [], []
        
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device).long()
                output = model(imgs)
                
                # Calculate Validation Loss
                loss = 0.5 * criterion_dice(output, msks) + 0.5 * criterion_focal(output, msks)
                val_loss += loss.item()

                # Calculate stats for IoU
                pred = output.argmax(dim=1)
                stats = smp.metrics.get_stats(pred, msks, mode='multiclass', num_classes=10)
                tp.append(stats[0]); fp.append(stats[1]); fn.append(stats[2]); tn.append(stats[3])

        # Finalize Metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        
        # Calculate Mean IoU across classes
        tp_cat = torch.cat(tp, dim=0).sum(dim=0)
        fp_cat = torch.cat(fp, dim=0).sum(dim=0)
        fn_cat = torch.cat(fn, dim=0).sum(dim=0)
        tn_cat = torch.cat(tn, dim=0).sum(dim=0)
        val_iou = smp.metrics.iou_score(tp_cat, fp_cat, fn_cat, tn_cat, reduction="micro").item()

        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save the best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New Best Model Saved (IoU: {best_val_iou:.4f})")
        
        scheduler.step()

if __name__ == "__main__":
    train()