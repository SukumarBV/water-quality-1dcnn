#  Water Quality CNN – Risk Scoring Model

A lightweight **1D Convolutional Neural Network** that predicts a continuous **contamination risk score** `[0.0 → 1.0]` from four water quality sensor readings. Designed to run on embedded hardware via **TensorFlow Lite Micro on ESP32-S3**.

---

## 🧠 What This Model Does

| Risk Score | Meaning |
|------------|---------|
| 0.0 – 0.2  | ✅ Safe – water is clean |
| 0.2 – 0.4  | 🟡 Low Risk – generally acceptable |
| 0.4 – 0.6  | 🟠 Moderate Risk – treatment recommended |
| 0.6 – 0.8  | 🔴 High Risk – do not consume |
| 0.8 – 1.0  | ☠️ Critical – severely contaminated |

This is **regression**, not classification. The model outputs a smooth, continuous score.

---

## 📁 Project Structure

```
water_quality_cnn/
│
├── train.py                        ← Main training pipeline
├── requirements.txt
├── README.md
│
├── data/
│   └── water_quality_dataset.csv  ← Your CSV (auto-generated if absent)
│
├── models/
│   ├── water_quality_cnn.h5       ← Keras model (best checkpoint)
│   ├── water_quality_cnn.tflite   ← Float32 TFLite model
│   ├── water_quality_cnn_quantized.tflite  ← INT8 quantized TFLite
│   └── feature_scaler.npy         ← Saved scaler parameters
│
├── plots/
│   └── training_report.png        ← Loss curves + predictions + errors
│
├── scripts/
│   ├── generate_dataset.py        ← Synthetic dataset generator
│   └── infer.py                   ← Single-sample inference script
│
└── esp32/
    └── water_quality_inference.ino ← Arduino sketch for ESP32-S3
```

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate a synthetic dataset (optional)

```bash
python scripts/generate_dataset.py
# → Creates data/water_quality_dataset.csv (2,000 samples)
```

> If you skip this step, `train.py` auto-generates the dataset on first run.

### 3. Train the model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Build and train the 1D-CNN
- Print MAE / MSE / RMSE / R² on the test set
- Save the Keras `.h5` model and both TFLite models
- Generate `plots/training_report.png`

### 4. Run inference on a single sample

```bash
# Using the Keras model
python scripts/infer.py --pH 7.1 --turbidity 3.2 --ec 210 --temp 27

# Using the TFLite model (mirrors ESP32 behaviour)
python scripts/infer.py --pH 6.2 --turbidity 9.1 --ec 450 --temp 31 --tflite

# Run all 5 built-in demo samples
python scripts/infer.py --demo
```

---

##  Model Architecture

```
Input (4 features, 1 channel)
│
├─ Conv1D(32 filters, kernel=2, padding=same) + ReLU
├─ MaxPooling1D(pool=2, padding=same)
│
├─ Conv1D(64 filters, kernel=2, padding=same) + ReLU
├─ GlobalAveragePooling1D          ← compact, embedded-friendly
│
├─ Dropout(0.25)
├─ Dense(32, ReLU)
└─ Dense(1, sigmoid)               ← output ∈ [0, 1]
```

**Why GlobalAveragePooling1D?**
Replaces `Flatten` → dramatically fewer parameters → smaller model → better for TFLite Micro.

---

## ⚙️ Recommended Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Epochs | 100 (w/ early stopping) | Usually stops around 30–60 |
| Batch size | 32 | Balance between speed and gradient quality |
| Learning rate | 1e-3 | Adam default; auto-reduced on plateau |
| Patience | 15 | Epochs to wait before early stopping |
| Dropout | 0.25 | Light regularisation – increase if overfitting |
| Loss | MSE | Standard for regression |
| Optimiser | Adam | Adaptive learning rate – robust |

---

##  Expected Performance

On the synthetic dataset (2,000 samples):

| Metric | Typical Value |
|--------|--------------|
| MAE    | ~0.02 – 0.04 |
| RMSE   | ~0.03 – 0.06 |
| R²     | ~0.92 – 0.97 |

Real-world datasets with sensor noise will show higher error; consider collecting ≥5,000 labelled samples.

---

## 📦 TFLite Models

| Model | Size | Use Case |
|-------|------|----------|
| `water_quality_cnn.tflite` | ~80–120 KB | ESP32-S3 with FPU |
| `water_quality_cnn_quantized.tflite` | ~20–30 KB | ESP32-S3 TFLite Micro (recommended) |

---

## 🔧 ESP32-S3 Deployment

### Step 1 – Convert model to C header

```bash
xxd -i models/water_quality_cnn_quantized.tflite > esp32/model_data.h
```

### Step 2 – Install Arduino library

In Arduino IDE: **Library Manager → search "EloquentTinyML" → Install**

### Step 3 – Update scaler values

After training, the console prints the MinMax scaler parameters. Copy them into `water_quality_inference.ino`:

```cpp
const float SCALER_MIN[4]   = { ... };   // printed during train.py
const float SCALER_SCALE[4] = { ... };
```

### Step 4 – Flash and monitor

Upload `esp32/water_quality_inference.ino` to your ESP32-S3.  
Open Serial Monitor at **115200 baud**.

### ESP32 Optimisation Tips

- ✅ Use the **INT8 quantized** model – 4× smaller, marginal accuracy loss
- ✅ Set `TENSOR_ARENA` to the minimum that works (saves RAM)
- ✅ Use `esp_light_sleep_start()` between readings for battery life
- ✅ Cache inference results; don't run every loop iteration
- ⚠️ If inference fails: increase `TENSOR_ARENA` by 1 KB and retry

---

## 📈 CSV Dataset Format

```csv
pH,Turbidity,EC,Temperature,RiskScore
7.1,3.2,210,27,0.12
6.2,9.1,450,31,0.84
5.0,35.0,1200,38,0.91
8.5,1.0,180,22,0.05
```

| Column | Unit | Safe Range |
|--------|------|------------|
| pH | — | 6.5 – 8.5 |
| Turbidity | NTU | < 4 |
| EC | µS/cm | < 300 |
| Temperature | °C | 20 – 30 |
| RiskScore | [0, 1] | < 0.2 |

---

## 🔬 Bringing Your Own Data

1. Collect labelled water samples with known contamination levels
2. Format as CSV with the columns above
3. Place at `data/water_quality_dataset.csv`
4. Run `python train.py` – it will use your real data automatically

For best results: **≥1,000 samples**, diverse contamination levels, balanced across risk categories.
