"""
=============================================================================
Water Quality Dataset Generator
=============================================================================
This script generates a synthetic dataset that mimics real water quality
sensor readings. It creates realistic relationships between water parameters
and contamination risk scores.

RUN: python scripts/generate_dataset.py
OUTPUT: data/water_quality_dataset.csv
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES = 2000          # Total samples to generate
OUTPUT_PATH = "data/water_quality_dataset.csv"

def compute_risk_score(pH, turbidity, ec, temperature):
    """
    Compute a continuous risk score [0, 1] from sensor readings.

    Physics-based heuristic rules (approximating real water quality standards):
      - WHO safe pH range: 6.5 – 8.5  (outside → risk increases)
      - Turbidity: < 4 NTU safe, > 25 NTU very risky
      - EC (Electrical Conductivity): < 300 µS/cm safe, > 1500 risky
      - Temperature: 20–30°C optimal; extremes increase microbial risk
    """
    # pH risk: penalise deviation from neutral (7.0)
    pH_risk = np.clip(np.abs(pH - 7.0) / 3.5, 0, 1)        # max risk at pH 3.5 or 10.5

    # Turbidity risk: log-scale feel
    turbidity_risk = np.clip(turbidity / 50.0, 0, 1)

    # EC risk: linear scale
    ec_risk = np.clip((ec - 100) / 1900.0, 0, 1)

    # Temperature risk: bell-shaped penalty away from 25 °C
    temp_risk = np.clip(np.abs(temperature - 25) / 20.0, 0, 1)

    # Weighted combination
    risk = (0.35 * pH_risk +
            0.30 * turbidity_risk +
            0.25 * ec_risk +
            0.10 * temp_risk)

    # Add small Gaussian noise to simulate sensor imprecision
    noise = np.random.normal(0, 0.03, size=risk.shape)
    risk = np.clip(risk + noise, 0.0, 1.0)

    return np.round(risk, 4)


def generate_dataset(n_samples: int) -> pd.DataFrame:
    """
    Generate synthetic water quality samples with realistic distributions.

    Sensor ranges (typical field deployments):
      pH           : 3.5 – 10.5
      Turbidity    : 0.1 – 80 NTU
      EC           : 50 – 2000 µS/cm
      Temperature  : 5 – 45 °C
    """
    print(f"[INFO] Generating {n_samples} synthetic water quality samples …")

    # Sample each feature from a realistic distribution
    pH          = np.random.uniform(3.5, 10.5, n_samples)
    turbidity   = np.random.exponential(scale=8.0, size=n_samples)   # most readings low
    turbidity   = np.clip(turbidity, 0.1, 80.0)
    ec          = np.random.uniform(50, 2000, n_samples)
    temperature = np.random.normal(loc=27, scale=8, size=n_samples)
    temperature = np.clip(temperature, 5, 45)

    risk_score  = compute_risk_score(pH, turbidity, ec, temperature)

    df = pd.DataFrame({
        "pH"          : np.round(pH, 2),
        "Turbidity"   : np.round(turbidity, 2),
        "EC"          : np.round(ec, 1),
        "Temperature" : np.round(temperature, 1),
        "RiskScore"   : risk_score,
    })

    return df


def main():
    os.makedirs("data", exist_ok=True)

    df = generate_dataset(N_SAMPLES)

    # ── Quick sanity checks ───────────────────────────────────────────────────
    print(f"\n[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Risk score distribution:\n{df['RiskScore'].describe().round(4)}")
    print(f"\n[INFO] First 5 rows:\n{df.head()}")

    # Verify no nulls
    assert df.isnull().sum().sum() == 0, "Unexpected nulls in generated data!"

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[✓] Dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
