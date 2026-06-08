"""
monitor.py — Basic model monitoring: data drift + prediction drift detection
Run this periodically to check if the model needs retraining.
"""

import os, sys, json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_prepare

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices_srilanka.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "monitor_report.json")

# Baseline stats computed from training data
BASELINE = {
    "perch_mean": 11.51, "perch_std": 8.52,
    "bedrooms_mean": 3.90, "bathrooms_mean": 2.50,
    "price_mean": 15981420, "price_std": 13381560,
}

DRIFT_THRESHOLD = 0.15  # 15% deviation triggers warning


def check_feature_drift(df: pd.DataFrame) -> dict:
    """Compare current data stats against baseline."""
    results = {}
    checks = [
        ("perch", BASELINE["perch_mean"], BASELINE["perch_std"]),
        ("bedrooms", BASELINE["bedrooms_mean"], None),
        ("bathrooms", BASELINE["bathrooms_mean"], None),
    ]
    for col, base_mean, base_std in checks:
        current_mean = df[col].mean()
        deviation = abs(current_mean - base_mean) / base_mean
        results[col] = {
            "baseline_mean": round(base_mean, 2),
            "current_mean": round(current_mean, 2),
            "deviation_pct": round(deviation * 100, 2),
            "drift_detected": deviation > DRIFT_THRESHOLD
        }
    return results


def check_prediction_drift(model, X: pd.DataFrame) -> dict:
    """Run model on sample and check if prediction distribution shifted."""
    sample = X.sample(min(500, len(X)), random_state=42)
    preds = model.predict(sample)
    current_mean = np.mean(preds)
    current_std  = np.std(preds)
    mean_dev = abs(current_mean - BASELINE["price_mean"]) / BASELINE["price_mean"]
    return {
        "baseline_price_mean": BASELINE["price_mean"],
        "current_price_mean": round(current_mean, 0),
        "deviation_pct": round(mean_dev * 100, 2),
        "drift_detected": mean_dev > DRIFT_THRESHOLD,
        "current_std": round(current_std, 0),
    }


def run_monitoring():
    print("=" * 55)
    print("  MLOps Monitor — SL House Price Model")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Load model
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    cols_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    if not os.path.exists(model_path):
        print("❌ Model not found. Run src/train.py first.")
        return

    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)

    # Load data
    df_raw = pd.read_csv(DATA_PATH)
    X, y, _ = load_and_prepare(DATA_PATH)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Feature drift
    print("\n📊 Feature Drift Check:")
    feature_results = check_feature_drift(df_raw)
    for col, stats in feature_results.items():
        status = "⚠️  DRIFT" if stats["drift_detected"] else "✅ OK"
        print(f"  {col:12} baseline={stats['baseline_mean']:6.2f} | current={stats['current_mean']:6.2f} | dev={stats['deviation_pct']:5.1f}% | {status}")

    # Prediction drift
    print("\n🔮 Prediction Drift Check:")
    pred_results = check_prediction_drift(model, X)
    status = "⚠️  DRIFT" if pred_results["drift_detected"] else "✅ OK"
    print(f"  Price mean  baseline={pred_results['baseline_price_mean']:,.0f} | current={pred_results['current_price_mean']:,.0f} | dev={pred_results['deviation_pct']:.1f}% | {status}")

    # Overall recommendation
    any_drift = any(v["drift_detected"] for v in feature_results.values()) or pred_results["drift_detected"]
    recommendation = "⚠️  RETRAIN RECOMMENDED" if any_drift else "✅ Model is healthy — no retraining needed"
    print(f"\n🏁 Result: {recommendation}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "feature_drift": feature_results,
        "prediction_drift": pred_results,
        "retrain_recommended": any_drift
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: int(x) if hasattr(x, "item") else str(x))
    print(f"\n📄 Report saved to {REPORT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    run_monitoring()
