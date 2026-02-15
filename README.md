# 🌿 Offroad Scene Semantic Segmentation

IBA Datathon Project

## 📌 Overview

This project performs **multi-class semantic segmentation** on off-road environmental images using **Unet++ with a ResNet50 encoder**.

The model segments each image into 10 semantic classes:

* Trees
* Lush Bushes
* Dry Grass
* Dry Bushes
* Ground Clutter
* Flowers
* Logs
* Rocks
* Landscape
* Sky

The system is implemented using **PyTorch** and **segmentation_models_pytorch**, with Dice + Focal loss for improved class balance handling.

---

## 🧠 Model Architecture

* **Architecture:** Unet++
* **Encoder:** ResNet50
* **Classes:** 10
* **Loss Function:**

  * Dice Loss (multiclass)
  * Focal Loss (multiclass)
* **Optimizer:** AdamW
* **Scheduler:** CosineAnnealingLR

---

## 📂 Project Structure

```
project/
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation & visualization script
├── model.pkl                 # Saved trained model (state_dict)
├── requirements.txt

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If running on GPU, install CUDA-enabled PyTorch from:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🚀 Training

Run:

```bash
python train.py
```

Training features:

* Image resizing to 512×512
* Data augmentation (flip, brightness/contrast, hue/saturation)
* Validation IoU calculation
* Best model saved automatically

The best model is saved as:

```
model.pkl
```

## 🔬 Evaluation

Run:

```bash
python evaluate.py
```

Evaluation performs:

* Per-class IoU calculation
* Mean IoU calculation
* Visualization of first 15 predictions
* Saves visual comparison images

Output example:

```
CLASS NAME               | IOU SCORE
----------------------------------------
Trees                    | 0.8123
Lush Bushes              | 0.7431
...
FINAL MEAN IoU SCORE     | 0.7568
```

Visualization images are saved in:

```
./boosted_eval_results/
```

---

## 🎨 Visualization

Each saved visualization includes:

1. Original image
2. Ground truth mask
3. Model prediction

A professional color map is used for clear reporting.

---

## 📊 Metrics

The evaluation computes:

* True Positives (TP)
* False Positives (FP)
* False Negatives (FN)
* True Negatives (TN)
* Per-class IoU
* Mean IoU

IoU is computed using `segmentation_models_pytorch.metrics`.

---

## Download trained model from:
https://drive.google.com/file/d/16chig0coBFXnaVDjt4odhd94Q7LnqoYA/view?usp=sharing


## 🔐 Model Format

The trained model is stored as a **state_dict** serialized using pickle:

```
model.pkl
```

During evaluation:

```python
with open(MODEL_PATH, "rb") as f:
    state_dict = pickle.load(f)

model.load_state_dict(state_dict)
```

---

## 💻 Hardware Compatibility

The code automatically detects device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Works on:

* CPU-only systems
* GPU systems

No CUDA-specific configuration required.

---

## 📈 Reproducibility Notes

* Resize: 512 × 512
* Normalization: ImageNet mean & std
* Batch size: 4
* Epochs: 50

---

## 🏁 Final Output

The system produces:

* Quantitative evaluation (Mean IoU)
* Per-class IoU breakdown
* Saved prediction visualizations

---

## 📌 Notes for Judges

If running on GPU, please ensure CUDA-compatible PyTorch is installed.
Otherwise, CPU execution works by default.