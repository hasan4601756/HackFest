# 🌿 Offroad Scene Semantic Segmentation with SegFormer

IBA Datathon Project

## 📌 Overview

This project performs **multi-class semantic segmentation** on off-road environmental images using **SegFormer (B2/B5)**.

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

The system is implemented using **PyTorch** and **Transformers**, with optional class weighting to handle imbalanced classes.

---

## 🧠 Model Architecture

* **Architecture:** SegFormer
* **Backbone:** MIT-B2 or MIT-B5
* **Classes:** 10
* **Loss Function:** Built-in CrossEntropyLoss (optionally weighted)
* **Optimizer:** AdamW
* **Mixed Precision Training:** Enabled via `torch.amp` for faster training and lower memory usage

---

## 📂 Project Structure

```
project/
│
├── train.py                  # Training script for SegFormer
├── evaluate.py               # Evaluation & visualization script
├── segformer_big_model.pth   # Saved trained model (state_dict)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If running on GPU, install CUDA-enabled PyTorch:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🚀 Training

Run:

```bash
python train.py
```

Training features:

* Image resizing to 512×512
* Data augmentation (flip, brightness/contrast)
* Mixed precision (automatic) for GPU efficiency
* Optional class-weighted CrossEntropyLoss for imbalanced classes
* Model checkpoint saved after each epoch

The best model is saved as:

```
segformer_big_model.pth
```

---

## 🔬 Evaluation

Run:

```bash
python evaluate.py
```

Evaluation performs:

* Per-class IoU calculation
* Mean IoU calculation
* Visualization of first 10 predictions
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
./segformer_big_results/
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
https://drive.google.com/file/d/14LUjD-1mzeFRvBpqCfcoaHS3Ap-NARXB/view?usp=sharing


## 🔐 Model Format

The trained model is stored as a **state_dict** serialized with PyTorch:

```
segformer_big_model.pth
```

During evaluation:

```python
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, num_labels=10)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
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

No additional CUDA configuration is required.

---

## 📈 Reproducibility Notes

* Resize: 512 × 512
* Normalization: ImageNet mean & std
* Batch size: 4 (B2) / 1–2 (B5 on small GPU)
* Epochs: 15
* Optimizer: AdamW, LR = 6e-5

---

## 🏁 Final Output

The system produces:

* Quantitative evaluation (Mean IoU)
* Per-class IoU breakdown
* Saved prediction visualizations

---

## 📌 Notes for Judges

* If running on GPU, ensure a CUDA-compatible PyTorch installation.
* CPU execution works by default but will be slower.

* Model backbone must match training (`B2` or `B5`) to load weights correctly.
