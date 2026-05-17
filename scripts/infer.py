"""
=============================================================================
Water Quality – Single-Sample Inference Script
=============================================================================
Test the trained model with one water quality reading.

USAGE:
  # Keras model (default)
  python scripts/infer.py --pH 7.1 --turbidity 3.2 --ec 210 --temp 27

  # TFLite model (embedded-style, closer to ESP32 behaviour)
  python scripts/infer.py --pH 6.2 --turbidity 9.1 --ec 450 --temp 31 --tflite

  # Run all built-in demo samples
  python scripts/infer.py --demo
=============================================================================
"""

import argparse
import os
import sys
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
KERAS_PATH   = "models/water_quality_cnn.h5"
TFLITE_PATH  = "models/water_quality_cnn.tflite"
SCALER_PATH  = "models/feature_scaler.npy"

# ─────────────────────────────────────────────────────────────────────────────
# Risk label mapping
# ─────────────────────────────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.20, "✅ SAFE",          "Water is safe to use."),
    (0.40, "🟡 LOW RISK",      "Generally acceptable; monitor parameters."),
    (0.60, "🟠 MODERATE RISK", "Treatment recommended before consumption."),
    (0.80, "🔴 HIGH RISK",     "Do NOT consume; requires treatment."),
    (1.01, "☠️  CRITICAL",     "Severely contaminated – immediate action required!"),
]

DEMO_SAMPLES = [
    {"pH": 7.1,  "Turbidity": 3.2,  "EC": 210,  "Temperature": 27,
     "label": "Clean tap water"},
    {"pH": 6.2,  "Turbidity": 9.1,  "EC": 450,  "Temperature": 31,
     "label": "Slightly polluted"},
    {"pH": 5.0,  "Turbidity": 35.0, "EC": 1200, "Temperature": 38,
     "label": "Heavily polluted"},
    {"pH": 8.5,  "Turbidity": 1.0,  "EC": 180,  "Temperature": 22,
     "label": "High-quality spring water"},
    {"pH": 4.1,  "Turbidity": 60.0, "EC": 1900, "Temperature": 42,
     "label": "Extremely contaminated"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_scaler():
    """Load the fitted scaler parameters saved during training."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'. Run train.py first."
        )
    params = np.load(SCALER_PATH, allow_pickle=True).item()
    return params["min_"], params["scale_"]


def normalise(features: np.ndarray, min_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    """Apply the same MinMax normalisation used during training."""
    return (features - min_) * scale_


def get_risk_label(score: float) -> tuple:
    for threshold, label, description in RISK_LEVELS:
        if score < threshold:
            return label, description
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def print_result(score: float, label: str, desc: str, sample_info: str = ""):
    bar_len  = int(score * 30)
    bar      = "█" * bar_len + "░" * (30 - bar_len)
    colour   = ""  # terminal colours optional

    print("\n" + "─"*55)
    if sample_info:
        print(f"  Sample : {sample_info}")
    print(f"  Score  : {score:.4f}  [{bar}]")
    print(f"  Level  : {label}")
    print(f"  Detail : {desc}")
    print("─"*55)


# ─────────────────────────────────────────────────────────────────────────────
# Inference engines
# ─────────────────────────────────────────────────────────────────────────────
def infer_keras(features: np.ndarray) -> float:
    """Run inference using the full Keras/H5 model."""
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    inp = features.reshape(1, -1, 1).astype(np.float32)
    return float(model.predict(inp, verbose=0)[0][0])


def infer_tflite(features: np.ndarray) -> float:
    """
    Run inference using the TFLite flat-buffer model.
    This closely mirrors how the ESP32-S3 TFLite Micro runtime behaves.
    """
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    inp_data = features.reshape(1, -1, 1).astype(np.float32)
    interp.set_tensor(inp_details[0]["index"], inp_data)
    interp.invoke()

    return float(interp.get_tensor(out_details[0]["index"])[0][0])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Water Quality Risk Score – Single Sample Inference"
    )
    parser.add_argument("--pH",         type=float, default=7.0)
    parser.add_argument("--turbidity",  type=float, default=3.0)
    parser.add_argument("--ec",         type=float, default=250.0)
    parser.add_argument("--temp",       type=float, default=25.0)
    parser.add_argument("--tflite",     action="store_true",
                        help="Use TFLite model instead of Keras H5")
    parser.add_argument("--demo",       action="store_true",
                        help="Run all built-in demo samples")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  WATER QUALITY CNN – INFERENCE")
    print("="*55)

    # Check model files exist
    model_path = TFLITE_PATH if args.tflite else KERAS_PATH
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: '{model_path}'")
        print("[ERROR] Please run train.py first.")
        sys.exit(1)

    min_, scale_ = load_scaler()
    engine = "TFLite" if args.tflite else "Keras"
    print(f"[INFO] Using {engine} inference engine")

    infer_fn = infer_tflite if args.tflite else infer_keras

    if args.demo:
        print(f"\n[INFO] Running {len(DEMO_SAMPLES)} demo samples …")
        for s in DEMO_SAMPLES:
            raw = np.array([s["pH"], s["Turbidity"], s["EC"], s["Temperature"]],
                           dtype=np.float32)
            normed = normalise(raw, min_, scale_)
            score  = infer_fn(normed)
            label, desc = get_risk_label(score)
            print_result(score, label, desc, s["label"])
    else:
        raw    = np.array([args.pH, args.turbidity, args.ec, args.temp],
                          dtype=np.float32)
        normed = normalise(raw, min_, scale_)
        score  = infer_fn(normed)
        label, desc = get_risk_label(score)

        print(f"\n  pH          : {args.pH}")
        print(f"  Turbidity   : {args.turbidity} NTU")
        print(f"  EC          : {args.ec} µS/cm")
        print(f"  Temperature : {args.temp} °C")
        print_result(score, label, desc)

    print()


if __name__ == "__main__":
    main()
