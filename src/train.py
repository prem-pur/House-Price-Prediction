"""
train.py — Train models and log experiments to MLflow
"""

import sys
import os
import joblib
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_prepare, HouseFeatureEngineer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices_srilanka.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_and_log(model_name, model, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("SL-House-Price-Prediction")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"[{model_name}] R2={metrics['R2']:.4f} | MAE={metrics['MAE']:,.0f} | RMSE={metrics['RMSE']:,.0f} | MAPE={metrics['MAPE']:.2f}% | run_id={run_id}")
        return metrics, run_id


def main():
    print("Loading and preparing data...")
    X, y, _ = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    }

    best_r2 = -999
    best_model = None
    best_name = ""

    for name, model in models.items():
        metrics, run_id = train_and_log(name, model, X_train, X_test, y_train, y_test)
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_name = name

    # Save best model + feature columns
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    feature_cols = X.columns.tolist()
    cols_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"\nBest model: {best_name} (R2={best_r2:.4f}) — saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")


if __name__ == "__main__":
    main()
