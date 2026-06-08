# Sri Lanka House Price Prediction — MLOps Project

> A production-grade MLOps pipeline that predicts Sri Lankan house prices using Gradient Boosting, tracks experiments with MLflow, serves predictions via a FastAPI REST API with a built-in web UI, and is fully containerized with Docker.

---

## Project Structure

```
mlops-sl-house-price/
├── data/
│   └── house_prices_srilanka.csv      # 20,000 SL property records
├── src/
│   ├── preprocess.py                  # Custom sklearn transformer
│   ├── train.py                       # Train 3 models + log to MLflow
│   └── monitor.py                     # Data & prediction drift detection
├── api/
│   └── main.py                        # FastAPI app + embedded web UI
├── tests/
│   ├── test_preprocess.py             # 17 unit tests for feature engineering
│   └── test_api.py                    # 13 unit tests for API endpoints
├── models/
│   ├── best_model.pkl                 # Saved best model (GradientBoosting)
│   ├── feature_columns.json           # Training feature schema
│   └── monitor_report.json            # Latest monitoring report
├── .github/
│   └── workflows/ci.yml               # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
python src/train.py
```
Trains Ridge, RandomForest, and GradientBoosting — logs all to MLflow.

### 3. View MLflow dashboard
```bash
python -m mlflow ui
# → http://localhost:5000
```

### 4. Run the web app
```bash
python -m uvicorn api.main:app --reload
# → http://localhost:8000
```

### 5. Run tests
```bash
python -m pytest tests/ -v
# 30 tests — all passing ✅
```

### 6. Run monitoring
```bash
python src/monitor.py
```

---

## Docker

```bash
# Build and start everything (API + MLflow)
docker-compose up --build

# API  → http://localhost:8000
# MLflow → http://localhost:5000
```

---

## Model Results

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Ridge Regression | 0.877 | LKR 2.9M | 26.4% |
| Random Forest | 0.955 | LKR 1.8M | 13.1% |
| **Gradient Boosting** | **0.966** | **LKR 1.6M** | **11.3%** |

---

## API Usage

### Predict house price
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Colombo", "area": "Borella",
    "perch": 10, "bedrooms": 3, "bathrooms": 2,
    "kitchen_area_sqft": 150, "parking_spots": 1,
    "has_garden": true, "has_ac": false,
    "water_supply": "Pipe-borne", "electricity": "Single phase",
    "floors": 2, "year_built": 2010
  }'
```
**Response:**
```json
{
  "predicted_price_lkr": 24500000.0,
  "predicted_price_lkr_formatted": "LKR 24,500,000"
}
```

### Retrain model
```bash
curl -X POST http://localhost:8000/retrain
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Tests (30 total)

| Test File | Tests | Coverage |
|---|---|---|
| `test_preprocess.py` | 17 | Feature engineering, encoding, data loading |
| `test_api.py` | 13 | Health, predict, validation, edge cases |

---

## MLOps Components

| Component | Tool | Purpose |
|---|---|---|
| Experiment Tracking | MLflow | Log metrics, params, artifacts per run |
| Model Serving | FastAPI | REST API + web UI |
| Containerization | Docker | Reproducible deployment |
| CI/CD | GitHub Actions | Auto-test on every push |
| Monitoring | Custom script | Drift detection, retrain alerts |
| Feature Engineering | sklearn Pipeline | Prevent data leakage |


## Dataset

**Source:** [House Prices in Sri Lanka (Kaggle)](https://www.kaggle.com/datasets/dewminimnaadi/house-prices-in-sri-lanka)