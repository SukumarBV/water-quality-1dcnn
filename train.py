"""
=============================================================================
Water Quality Risk Scoring – 1D CNN Training Script
=============================================================================
MODEL  : Lightweight 1D-CNN regression (TFLite / ESP32-S3 ready)
OUTPUT : Continuous risk score ∈ [0, 1]
         0.0 → Completely safe   |   1.0 → Extremely unsafe

USAGE  : python train.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for servers / CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HYPERPARAMETERS  (all in one place – easy to tune)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Data
    "csv_path"        : "data/water_quality_dataset.csv",
    "feature_cols"    : ["pH", "Turbidity", "EC", "Temperature"],
    "target_col"      : "RiskScore",
    "test_size"       : 0.20,     # 20 % held out for evaluation
    "val_size"        : 0.15,     # 15 % of training set used for validation
    "random_seed"     : 42,

    # Model
    "conv1_filters"   : 32,
    "conv2_filters"   : 64,
    "kernel_size"     : 2,        # small kernel → fewer params
    "dense_units"     : 32,
    "dropout_rate"    : 0.25,     # light regularisation

    # Training
    "epochs"          : 100,
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "patience"        : 15,       # early-stopping patience

    # Paths
    "model_h5"        : "models/water_quality_cnn.h5",
    "model_tflite"    : "models/water_quality_cnn.tflite",
    "model_tflite_q"  : "models/water_quality_cnn_quantized.tflite",
    "plots_dir"       : "plots",
    "scaler_path"     : "models/feature_scaler.npy",   # save scaler params
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(cfg: dict):
    """
    Load CSV → handle NaNs → normalise features → split into train/val/test.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, features, 1)
    y_train, y_val, y_test  : np.ndarray, shape (N,)
    scaler                  : fitted MinMaxScaler
    """
    print("\n" + "="*60)
    print("  STEP 1 – DATA LOADING & PREPROCESSING")
    print("="*60)

    # ── 2a. Load CSV ──────────────────────────────────────────────────────────
    if not os.path.exists(cfg["csv_path"]):
        print(f"[WARN] '{cfg['csv_path']}' not found. Generating synthetic data …")
        _generate_fallback_dataset(cfg["csv_path"])

    df = pd.read_csv(cfg["csv_path"])
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} cols from '{cfg['csv_path']}'")

    # ── 2b. Handle missing values ─────────────────────────────────────────────
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} missing values found – filling with column medians")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("[INFO] No missing values detected ✓")

    # ── 2c. Separate features and target ──────────────────────────────────────
    X = df[cfg["feature_cols"]].values.astype(np.float32)
    y = df[cfg["target_col"]].values.astype(np.float32)

    print(f"[INFO] Features shape: {X.shape}  |  Target range: [{y.min():.3f}, {y.max():.3f}]")

    # ── 2d. Normalise features to [0, 1] using MinMaxScaler ──────────────────
    #        NOTE: fit ONLY on training data to prevent data leakage
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size   = cfg["test_size"],
        random_state= cfg["random_seed"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size   = cfg["val_size"],
        random_state= cfg["random_seed"],
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)   # fit here
    X_val   = scaler.transform(X_val)         # transform only
    X_test  = scaler.transform(X_test)        # transform only

    # Save scaler min/scale for embedded deployment reference
    np.save(cfg["scaler_path"], {
        "min_"  : scaler.data_min_,
        "scale_": scaler.scale_,
        "cols"  : cfg["feature_cols"],
    })
    print(f"[INFO] Scaler params saved → '{cfg['scaler_path']}'")

    # ── 2e. Reshape for Conv1D: (N, timesteps, channels) ─────────────────────
    #        We treat each feature as a "timestep" → shape (N, 4, 1)
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val   = X_val.reshape(-1, X_val.shape[1], 1)
    X_test  = X_test.reshape(-1, X_test.shape[1], 1)

    print(f"[INFO] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def build_model(input_shape: tuple, cfg: dict) -> keras.Model:
    """
    Lightweight 1D-CNN regression model.

    Architecture
    ────────────
    Input(4,1)
      → Conv1D(32, k=2) + ReLU
      → MaxPooling1D(pool=2)
      → Conv1D(64, k=2) + ReLU
      → GlobalAveragePooling1D            ← replaces Flatten → fewer weights
      → Dropout(0.25)
      → Dense(32, ReLU)
      → Dense(1, sigmoid)                 ← output ∈ (0, 1)

    Why GlobalAveragePooling1D?
      • Dramatically reduces parameter count vs Flatten
      • Acts as structural regularisation
      • Preferred for embedded / TFLite deployment
    """
    print("\n" + "="*60)
    print("  STEP 2 – BUILDING MODEL")
    print("="*60)

    inp = keras.Input(shape=input_shape, name="sensor_input")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters     = cfg["conv1_filters"],
        kernel_size = cfg["kernel_size"],
        padding     = "same",             # keep spatial size
        activation  = "relu",
        name        = "conv1",
    )(inp)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="pool1")(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters     = cfg["conv2_filters"],
        kernel_size = cfg["kernel_size"],
        padding     = "same",
        activation  = "relu",
        name        = "conv2",
    )(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)   # (batch, 64)

    # ── Head ──────────────────────────────────────────────────────────────────
    x = layers.Dropout(cfg["dropout_rate"], name="dropout")(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", name="dense1")(x)
    out = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    #                              ↑ sigmoid → output always in [0, 1]

    model = keras.Model(inputs=inp, outputs=out, name="WaterQualityCNN")

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss      = "mse",          # MSE for regression
        metrics   = ["mae"],        # track MAE during training
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Approx model size (fp32): {total_params * 4 / 1024:.1f} KB")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val, cfg: dict):
    """
    Train the model with:
      • Early stopping  – halts when val_loss stops improving
      • Model checkpoint – saves best weights automatically
    """
    print("\n" + "="*60)
    print("  STEP 3 – TRAINING")
    print("="*60)

    os.makedirs("models", exist_ok=True)

    cb_list = [
        # Stop training when validation loss plateaus
        callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = cfg["patience"],
            restore_best_weights = True,    # revert to best epoch weights
            verbose              = 1,
        ),
        # Save best model checkpoint during training
        callbacks.ModelCheckpoint(
            filepath         = cfg["model_h5"],
            monitor          = "val_loss",
            save_best_only   = True,
            verbose          = 1,
        ),
        # Reduce LR on plateau (optional but helps convergence)
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 7,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = cfg["epochs"],
        batch_size      = cfg["batch_size"],
        callbacks       = cb_list,
        verbose         = 1,
    )

    print(f"\n[✓] Training complete. Best model saved → '{cfg['model_h5']}'")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Compute and print regression metrics:
      MAE  – mean absolute error (average magnitude of errors)
      MSE  – mean squared error  (penalises large errors more)
      RMSE – square root of MSE  (same units as target)
      R²   – coefficient of determination (1.0 = perfect fit)
    """
    print("\n" + "="*60)
    print("  STEP 4 – EVALUATION")
    print("="*60)

    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  {'Metric':<10} {'Value':>10}")
    print(f"  {'-'*22}")
    print(f"  {'MAE':<10} {mae:>10.4f}")
    print(f"  {'MSE':<10} {mse:>10.4f}")
    print(f"  {'RMSE':<10} {rmse:>10.4f}")
    print(f"  {'R²':<10} {r2:>10.4f}")

    return y_pred, {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(history, y_test, y_pred, metrics: dict, cfg: dict):
    """Generate and save three diagnostic plots."""

    print("\n" + "="*60)
    print("  STEP 5 – PLOTTING")
    print("="*60)

    os.makedirs(cfg["plots_dir"], exist_ok=True)

    # ── Colour palette ────────────────────────────────────────────────────────
    C = {
        "bg"      : "#0d1117",
        "panel"   : "#161b22",
        "accent1" : "#58a6ff",
        "accent2" : "#3fb950",
        "accent3" : "#f78166",
        "text"    : "#c9d1d9",
        "subtext" : "#8b949e",
        "grid"    : "#21262d",
    }

    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    fig.suptitle(
        "Water Quality CNN – Training Report",
        fontsize=20, color=C["text"], fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    def _style_ax(ax, title):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["subtext"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["grid"])
        ax.set_title(title, color=C["text"], fontsize=12, pad=10)
        ax.grid(True, color=C["grid"], linewidth=0.6)
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])

    # ── Plot 1: Loss curves ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_ran = range(1, len(history.history["loss"]) + 1)
    ax1.plot(epochs_ran, history.history["loss"],
             color=C["accent1"], linewidth=2, label="Train Loss")
    ax1.plot(epochs_ran, history.history["val_loss"],
             color=C["accent2"], linewidth=2, linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend(facecolor=C["panel"], labelcolor=C["text"], fontsize=9)
    _style_ax(ax1, "Training vs Validation Loss")

    # ── Plot 2: MAE curves ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_ran, history.history["mae"],
             color=C["accent1"], linewidth=2, label="Train MAE")
    ax2.plot(epochs_ran, history.history["val_mae"],
             color=C["accent2"], linewidth=2, linestyle="--", label="Val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend(facecolor=C["panel"], labelcolor=C["text"], fontsize=9)
    _style_ax(ax2, "Training vs Validation MAE")

    # ── Plot 3: Predicted vs Actual ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_test, y_pred, alpha=0.45, s=18,
                color=C["accent1"], edgecolors="none", label="Predictions")
    lims = [0, 1]
    ax3.plot(lims, lims, color=C["accent3"], linewidth=1.5,
             linestyle="--", label="Perfect fit")
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.set_xlabel("Actual Risk Score")
    ax3.set_ylabel("Predicted Risk Score")
    ax3.text(0.05, 0.87,
             f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}",
             transform=ax3.transAxes, color=C["text"],
             fontsize=10, verticalalignment="top",
             bbox=dict(facecolor=C["bg"], alpha=0.7, edgecolor=C["grid"]))
    ax3.legend(facecolor=C["panel"], labelcolor=C["text"], fontsize=9)
    _style_ax(ax3, "Predicted vs Actual Risk Scores")

    # ── Plot 4: Error distribution ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    errors = y_pred - y_test
    ax4.hist(errors, bins=40, color=C["accent2"], edgecolor=C["bg"],
             alpha=0.85, density=True)
    ax4.axvline(0, color=C["accent3"], linewidth=1.8, linestyle="--",
                label="Zero error")
    ax4.axvline(errors.mean(), color=C["accent1"], linewidth=1.5,
                linestyle=":", label=f"Mean = {errors.mean():.4f}")
    ax4.set_xlabel("Prediction Error (pred − actual)")
    ax4.set_ylabel("Density")
    ax4.legend(facecolor=C["panel"], labelcolor=C["text"], fontsize=9)
    _style_ax(ax4, "Error Distribution")

    out_path = os.path.join(cfg["plots_dir"], "training_report.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print(f"[✓] Plot saved → '{out_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TFLite CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_tflite(model, X_train, cfg: dict):
    """
    Convert the trained Keras model to two TFLite formats:

    1. Standard float32 TFLite  – for devices with FPU (ESP32-S3)
    2. INT8 quantized TFLite    – for ultra-low memory (TFLite Micro)

    Quantization reduces model size by ~4× with minimal accuracy drop.
    """
    print("\n" + "="*60)
    print("  STEP 6 – TFLITE CONVERSION")
    print("="*60)

    os.makedirs("models", exist_ok=True)

    # ── 7a. Standard float32 conversion ──────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(cfg["model_tflite"], "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"[✓] Float32 TFLite → '{cfg['model_tflite']}'  ({size_kb:.1f} KB)")

    # ── 7b. INT8 post-training quantization ──────────────────────────────────
    def representative_dataset():
        """
        Feed ~100 representative samples so TFLite can calibrate
        integer scale factors for each layer's activations.
        """
        for i in range(min(100, len(X_train))):
            sample = X_train[i:i+1].astype(np.float32)
            yield [sample]

    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_dataset
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type  = tf.int8
    converter_q.inference_output_type = tf.int8

    try:
        tflite_q_model = converter_q.convert()
        with open(cfg["model_tflite_q"], "wb") as f:
            f.write(tflite_q_model)
        size_q_kb = len(tflite_q_model) / 1024
        print(f"[✓] INT8 Quantized TFLite → '{cfg['model_tflite_q']}'  ({size_q_kb:.1f} KB)")
        print(f"[INFO] Size reduction: {size_kb:.1f} KB → {size_q_kb:.1f} KB  "
              f"({100*(1 - size_q_kb/size_kb):.0f}% smaller)")
    except Exception as e:
        print(f"[WARN] Quantized conversion failed: {e}")
        print("[WARN] Float32 model still usable on ESP32-S3")

    # ── 7c. Verify TFLite inference (smoke test) ──────────────────────────────
    print("\n[INFO] Verifying TFLite float32 model …")
    interp = tf.lite.Interpreter(model_path=cfg["model_tflite"])
    interp.allocate_tensors()

    inp_details  = interp.get_input_details()
    out_details  = interp.get_output_details()
    test_sample  = X_train[0:1].astype(np.float32)

    interp.set_tensor(inp_details[0]["index"], test_sample)
    interp.invoke()
    tflite_out = interp.get_tensor(out_details[0]["index"])
    keras_out  = model.predict(test_sample, verbose=0)[0][0]

    print(f"  Keras output  : {keras_out:.6f}")
    print(f"  TFLite output : {tflite_out[0][0]:.6f}")
    print(f"  Difference    : {abs(keras_out - tflite_out[0][0]):.2e}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HELPER – FALLBACK DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _generate_fallback_dataset(path: str, n: int = 2000):
    """Inline fallback dataset generator (no external script needed)."""
    np.random.seed(42)
    pH          = np.random.uniform(3.5, 10.5, n)
    turbidity   = np.clip(np.random.exponential(8.0, n), 0.1, 80)
    ec          = np.random.uniform(50, 2000, n)
    temperature = np.clip(np.random.normal(27, 8, n), 5, 45)

    pH_risk   = np.clip(np.abs(pH - 7.0) / 3.5, 0, 1)
    turb_risk = np.clip(turbidity / 50.0, 0, 1)
    ec_risk   = np.clip((ec - 100) / 1900.0, 0, 1)
    temp_risk = np.clip(np.abs(temperature - 25) / 20.0, 0, 1)
    risk      = np.clip(0.35*pH_risk + 0.30*turb_risk +
                        0.25*ec_risk + 0.10*temp_risk +
                        np.random.normal(0, 0.03, n), 0, 1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "pH": np.round(pH, 2), "Turbidity": np.round(turbidity, 2),
        "EC": np.round(ec, 1), "Temperature": np.round(temperature, 1),
        "RiskScore": np.round(risk, 4),
    }).to_csv(path, index=False)
    print(f"[✓] Fallback dataset generated → '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print("  WATER QUALITY CNN – RISK SCORE REGRESSION PIPELINE")
    print("█"*60)
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  NumPy version      : {np.__version__}")

    # Set seed everywhere for reproducibility
    np.random.seed(CFG["random_seed"])
    tf.random.set_seed(CFG["random_seed"])

    # Step 1 – Load & preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        load_and_preprocess(CFG)

    # Step 2 – Build model
    input_shape = (X_train.shape[1], 1)   # (4, 1)
    model = build_model(input_shape, CFG)

    # Step 3 – Train
    history = train_model(model, X_train, y_train, X_val, y_val, CFG)

    # Step 4 – Evaluate on held-out test set
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    # Step 5 – Plots
    plot_results(history, y_test, y_pred, metrics, CFG)

    # Step 6 – TFLite export
    convert_to_tflite(model, X_train, CFG)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  PIPELINE COMPLETE")
    print("█"*60)
    print(f"  Keras model   : {CFG['model_h5']}")
    print(f"  TFLite model  : {CFG['model_tflite']}")
    print(f"  Quant TFLite  : {CFG['model_tflite_q']}")
    print(f"  Training plot : {CFG['plots_dir']}/training_report.png")
    print(f"  MAE  = {metrics['mae']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  R²   = {metrics['r2']:.4f}")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
