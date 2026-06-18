"""
main.py — FastAPI 4-page app: Home, Predict, Market Insights, Model Performance
"""
import os, sys, json, joblib
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import HouseFeatureEngineer

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices_srilanka.csv")

model = None
feature_columns = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_columns
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    cols_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        with open(cols_path) as f:
            feature_columns = json.load(f)
    yield
    model = None
    feature_columns = []

app = FastAPI(title="SL House Price Prediction", version="1.0.0", lifespan=lifespan)

class HouseInput(BaseModel):
    district: str
    area: str
    perch: int = Field(..., ge=1, le=200)
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=10)
    kitchen_area_sqft: int = Field(..., ge=50, le=2000)
    parking_spots: int = Field(..., ge=0, le=10)
    has_garden: bool
    has_ac: bool
    water_supply: str
    electricity: str
    floors: int = Field(..., ge=1, le=10)
    year_built: int = Field(..., ge=1950, le=2025)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(input: HouseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw = pd.DataFrame([input.model_dump()])
    engineer = HouseFeatureEngineer()
    raw = engineer.transform(raw)
    raw = pd.get_dummies(raw, columns=["district","area","water_supply","electricity"], drop_first=True)
    bool_cols = raw.select_dtypes(include="bool").columns
    raw[bool_cols] = raw[bool_cols].astype(int)
    raw = raw.reindex(columns=feature_columns, fill_value=0)
    price = float(model.predict(raw)[0])
    price = max(price, 0)
    return {"predicted_price_lkr": round(price, 2), "predicted_price_lkr_formatted": f"LKR {price:,.0f}"}

@app.post("/retrain")
def retrain():
    import subprocess
    global model, feature_columns
    train_script = os.path.join(os.path.dirname(__file__), "..", "src", "train.py")
    result = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    cols_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    return {"status": "retrained"}

@app.get("/api/market-data")
def market_data():
    df = pd.read_csv(DATA_PATH)
    dist = df.groupby("district")["price_lkr"].agg(["mean","median","count"]).round(0).reset_index()
    dist.columns = ["district","mean","median","count"]
    dist = dist.sort_values("mean", ascending=False)
    beds = df.groupby("bedrooms")["price_lkr"].mean().round(0).reset_index()
    beds.columns = ["bedrooms","avg_price"]
    price_ranges = [
        {"range": "< 5M", "count": int((df["price_lkr"] < 5e6).sum())},
        {"range": "5M–10M", "count": int(((df["price_lkr"] >= 5e6) & (df["price_lkr"] < 10e6)).sum())},
        {"range": "10M–25M", "count": int(((df["price_lkr"] >= 10e6) & (df["price_lkr"] < 25e6)).sum())},
        {"range": "25M–50M", "count": int(((df["price_lkr"] >= 25e6) & (df["price_lkr"] < 50e6)).sum())},
        {"range": "> 50M", "count": int((df["price_lkr"] >= 50e6).sum())},
    ]
    return {
        "districts": dist.to_dict(orient="records"),
        "by_bedrooms": beds.to_dict(orient="records"),
        "price_distribution": price_ranges,
        "summary": {
            "total": len(df),
            "avg_price": round(float(df["price_lkr"].mean()), 0),
            "median_price": round(float(df["price_lkr"].median()), 0),
            "max_price": round(float(df["price_lkr"].max()), 0),
            "min_price": round(float(df["price_lkr"].min()), 0),
        }
    }

NAV = """
<nav>
  <div class="nav-inner">
    <a href="/" class="nav-logo">Lanka<span>Home</span> Value</a>
    <div class="nav-links">
      <a href="/" class="nav-link" id="nl-home">Home</a>
      <a href="/predict" class="nav-link" id="nl-predict">Predict</a>
      <a href="/insights" class="nav-link" id="nl-insights">Market Insights</a>
      <a href="/model" class="nav-link" id="nl-model">Model Performance</a>
    </div>
    <div class="nav-badge">AI Powered</div>
  </div>
</nav>
"""

BASE_CSS = """
<style>
:root {
  --cream:#f5f0e8; --dark:#1a1410; --gold:#c9974a; --gold-light:#e8c07a;
  --sage:#5a7a5a; --card:#fffdf8; --border:#e0d8c8; --shadow:0 4px 40px rgba(26,20,16,0.10);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--cream);color:var(--dark);font-family:'DM Sans',sans-serif;min-height:100vh;}
nav{background:var(--dark);border-bottom:2px solid var(--gold);position:sticky;top:0;z-index:100;}
.nav-inner{max-width:1200px;margin:0 auto;padding:0 32px;height:68px;display:flex;align-items:center;gap:32px;}
.nav-logo{font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:900;color:var(--cream);text-decoration:none;letter-spacing:-0.5px;margin-right:auto;}
.nav-logo span{color:var(--gold);}
.nav-links{display:flex;gap:4px;}
.nav-link{color:#a09880;text-decoration:none;font-size:0.88rem;font-weight:500;padding:7px 14px;border-radius:3px;transition:color 0.2s,background 0.2s;}
.nav-link:hover{color:var(--cream);background:rgba(255,255,255,0.06);}
.nav-link.active{color:var(--gold);}
.nav-badge{background:var(--gold);color:var(--dark);font-size:0.68rem;font-weight:500;padding:4px 12px;border-radius:20px;letter-spacing:1px;text-transform:uppercase;white-space:nowrap;}
</style>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
"""

def active(page, current):
    return ' class="nav-link active"' if page == current else ' class="nav-link"'

# ── PAGE 1: HOME ─────────────────────────────────────────────────────────────
HOME_HTML = BASE_CSS + NAV + """
<style>
.hero{background:var(--dark);padding:88px 48px 100px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(201,151,74,0.12) 0%,transparent 70%);pointer-events:none;}
.hero-tag{display:inline-block;border:1px solid var(--gold);color:var(--gold);font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;padding:6px 16px;border-radius:2px;margin-bottom:28px;}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2.8rem,5vw,4.5rem);font-weight:900;color:var(--cream);line-height:1.08;max-width:700px;margin-bottom:20px;}
.hero h1 em{color:var(--gold);font-style:normal;}
.hero p{color:#a09880;font-size:1.1rem;font-weight:300;max-width:520px;line-height:1.75;margin-bottom:40px;}
.hero-cta{display:flex;gap:16px;flex-wrap:wrap;}
.btn-primary{display:inline-flex;align-items:center;gap:8px;background:var(--gold);color:var(--dark);padding:14px 28px;border-radius:3px;font-weight:500;font-size:0.95rem;text-decoration:none;transition:background 0.2s;}
.btn-primary:hover{background:var(--gold-light);}
.btn-secondary{display:inline-flex;align-items:center;gap:8px;background:transparent;color:var(--cream);padding:14px 28px;border-radius:3px;font-weight:500;font-size:0.95rem;text-decoration:none;border:1px solid rgba(255,255,255,0.2);transition:border-color 0.2s,color 0.2s;}
.btn-secondary:hover{border-color:var(--gold);color:var(--gold);}
.stats-bar{background:var(--gold);display:flex;}
.stat{flex:1;padding:20px 28px;border-right:1px solid rgba(26,20,16,0.2);text-align:center;}
.stat:last-child{border-right:none;}
.stat-num{font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:700;color:var(--dark);display:block;}
.stat-label{font-size:0.72rem;font-weight:500;color:rgba(26,20,16,0.65);text-transform:uppercase;letter-spacing:1px;}
.features{max-width:1100px;margin:0 auto;padding:80px 32px;display:grid;grid-template-columns:repeat(3,1fr);gap:32px;}
.feat-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:36px 28px;box-shadow:var(--shadow);transition:transform 0.2s,box-shadow 0.2s;}
.feat-card:hover{transform:translateY(-4px);box-shadow:0 8px 50px rgba(26,20,16,0.14);}
.feat-title{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;margin-bottom:10px;}
.feat-desc{font-size:0.88rem;color:#7a6a55;line-height:1.7;}
.feat-link{display:inline-block;margin-top:18px;color:var(--gold);font-size:0.82rem;font-weight:500;text-decoration:none;letter-spacing:0.5px;}
.feat-link:hover{text-decoration:underline;}
.how-section{background:var(--dark);padding:80px 32px;}
.how-inner{max-width:900px;margin:0 auto;text-align:center;}
.section-tag{display:inline-block;border:1px solid var(--gold);color:var(--gold);font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;padding:5px 14px;border-radius:2px;margin-bottom:20px;}
.section-title{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--cream);margin-bottom:48px;}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:32px;text-align:left;}
.step{position:relative;padding-left:48px;}
.step-num{position:absolute;left:0;top:0;width:32px;height:32px;background:var(--gold);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Playfair Display',serif;font-size:0.9rem;font-weight:700;color:var(--dark);}
.step-title{font-size:0.95rem;font-weight:500;color:var(--cream);margin-bottom:8px;}
.step-desc{font-size:0.82rem;color:#7a6a55;line-height:1.65;}
@media(max-width:800px){.features{grid-template-columns:1fr;}.steps{grid-template-columns:1fr;}.hero{padding:56px 24px 72px;}.stats-bar{flex-wrap:wrap;}.stat{min-width:50%;}}
</style>
<script>document.getElementById('nl-home').className='nav-link active';</script>

<section class="hero">
  <div class="hero-tag">Sri Lanka Real Estate Intelligence</div>
  <h1>Discover Your Property's <em>True Value</em></h1>
  <p>An end-to-end MLOps pipeline trained on 20,000 Sri Lankan properties. Instantly predict house prices, explore market trends, and understand model performance.</p>
  <div class="hero-cta">
    <a href="/predict" class="btn-primary">Get Price Estimate</a>
    <a href="/insights" class="btn-secondary">Explore Market Data</a>
  </div>
</section>

<div class="stats-bar">
  <div class="stat"><span class="stat-num">20,000</span><span class="stat-label">Properties Trained</span></div>
  <div class="stat"><span class="stat-num">96.6%</span><span class="stat-label">Model Accuracy (R²)</span></div>
  <div class="stat"><span class="stat-num">25</span><span class="stat-label">Districts Covered</span></div>
  <div class="stat"><span class="stat-num">LKR 1.6M</span><span class="stat-label">Avg Error (MAE)</span></div>
</div>

<div class="features">
  <div class="feat-card">
    <div class="feat-title">AI Price Prediction</div>
    <div class="feat-desc">Enter your property details and get an instant price estimate powered by Gradient Boosting — the best-performing model across 20,000 Sri Lankan properties.</div>
    <a href="/predict" class="feat-link">Predict now →</a>
  </div>
  <div class="feat-card">
    <div class="feat-title">Market Insights</div>
    <div class="feat-desc">Explore real estate price trends across all 25 districts, price distribution by bedroom count, and market heat maps sourced from the training dataset.</div>
    <a href="/insights" class="feat-link">View insights →</a>
  </div>
  <div class="feat-card">
    <div class="feat-title">Model Performance</div>
    <div class="feat-desc">Compare all three trained models (Ridge, Random Forest, Gradient Boosting) by R², MAE, and MAPE. See feature importances and understand what drives prices.</div>
    <a href="/model" class="feat-link">View performance →</a>
  </div>
</div>

<section class="how-section">
  <div class="how-inner">
    <div class="section-tag">How It Works</div>
    <div class="section-title">From Raw Data to Prediction</div>
    <div class="steps">
      <div class="step"><div class="step-num">1</div><div class="step-title">Data & Feature Engineering</div><div class="step-desc">20,000 property records are cleaned and engineered — house age, amenity score, and room ratios are derived to improve model signal.</div></div>
      <div class="step"><div class="step-num">2</div><div class="step-title">Experiment Tracking</div><div class="step-desc">Three models are trained and compared. Every run is logged to MLflow with metrics, parameters, and artifacts for full reproducibility.</div></div>
      <div class="step"><div class="step-num">3</div><div class="step-title">Live API Serving</div><div class="step-desc">The best model is saved and served via FastAPI. New predictions are returned in milliseconds with a ±8% confidence range.</div></div>
    </div>
  </div>
</section>
"""

# ── PAGE 2: PREDICT ──────────────────────────────────────────────────────────
PREDICT_HTML = BASE_CSS + NAV + """
<style>
.page-header{background:var(--dark);padding:48px 48px 56px;}
.page-header h1{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;color:var(--cream);margin-bottom:10px;}
.page-header p{color:#a09880;font-size:0.95rem;}
.main{max-width:1100px;margin:0 auto;padding:48px 32px;display:grid;grid-template-columns:1fr 380px;gap:36px;align-items:start;}
.form-card{background:var(--card);border:1px solid var(--border);border-radius:4px;overflow:hidden;box-shadow:var(--shadow);}
.form-section{padding:24px 28px;border-bottom:1px solid var(--border);}
.section-title{font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:700;color:var(--dark);margin-bottom:18px;display:flex;align-items:center;gap:10px;}
.section-title::before{content:'';display:block;width:4px;height:16px;background:var(--gold);border-radius:2px;}
.field-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.field-grid.cols3{grid-template-columns:1fr 1fr 1fr;}
.field{display:flex;flex-direction:column;gap:5px;}
label{font-size:0.72rem;font-weight:500;color:#6b5f50;text-transform:uppercase;letter-spacing:0.8px;}
select,input[type="number"]{width:100%;padding:10px 12px;background:var(--cream);border:1px solid var(--border);border-radius:3px;font-family:'DM Sans',sans-serif;font-size:0.9rem;color:var(--dark);appearance:none;outline:none;transition:border-color 0.2s,box-shadow 0.2s;}
select:focus,input[type="number"]:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(201,151,74,0.15);}
.select-wrap{position:relative;}.select-wrap::after{content:'▾';position:absolute;right:12px;top:50%;transform:translateY(-50%);color:var(--gold);pointer-events:none;font-size:0.78rem;}
.toggle-row{display:flex;gap:10px;}
.toggle-item{flex:1;display:flex;align-items:center;gap:9px;padding:9px 12px;background:var(--cream);border:1px solid var(--border);border-radius:3px;cursor:pointer;transition:border-color 0.2s,background 0.2s;user-select:none;}
.toggle-item:has(input:checked){border-color:var(--gold);background:rgba(201,151,74,0.08);}
.toggle-item input{display:none;}
.toggle-dot{width:16px;height:16px;border-radius:50%;border:2px solid var(--border);display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:border-color 0.2s,background 0.2s;}
.toggle-item:has(input:checked) .toggle-dot{border-color:var(--gold);background:var(--gold);}
.toggle-dot::after{content:'✓';color:var(--dark);font-size:0.6rem;font-weight:700;opacity:0;transition:opacity 0.15s;}
.toggle-item:has(input:checked) .toggle-dot::after{opacity:1;}
.toggle-label{font-size:0.86rem;color:var(--dark);}
.form-footer{padding:20px 28px;background:var(--dark);}
.btn-predict{width:100%;padding:15px;background:var(--gold);border:none;border-radius:3px;font-family:'DM Sans',sans-serif;font-size:0.95rem;font-weight:500;color:var(--dark);cursor:pointer;transition:background 0.2s;display:flex;align-items:center;justify-content:center;gap:10px;}
.btn-predict:hover{background:var(--gold-light);}
.result-panel{display:flex;flex-direction:column;gap:18px;}
.result-card{background:var(--card);border:1px solid var(--border);border-radius:4px;overflow:hidden;box-shadow:var(--shadow);opacity:0;transform:translateY(10px);transition:opacity 0.4s,transform 0.4s;}
.result-card.visible{opacity:1;transform:none;}
.result-header{background:var(--dark);padding:16px 22px;font-family:'Playfair Display',serif;font-size:0.82rem;color:var(--gold);letter-spacing:1px;text-transform:uppercase;}
.result-body{padding:24px 22px;}
.price-display{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:var(--dark);line-height:1;margin-bottom:6px;}
.price-sub{font-size:0.8rem;color:#8a7a65;margin-bottom:18px;}
.price-bar{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:18px;}
.price-fill{height:100%;background:linear-gradient(90deg,var(--sage),var(--gold));border-radius:3px;width:0%;transition:width 1s ease;}
.breakdown-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.breakdown-item{background:var(--cream);border-radius:3px;padding:10px;border:1px solid var(--border);}
.breakdown-label{font-size:0.65rem;color:#8a7a65;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px;}
.breakdown-value{font-size:0.9rem;font-weight:500;color:var(--dark);}
.history-card{background:var(--card);border:1px solid var(--border);border-radius:4px;overflow:hidden;box-shadow:var(--shadow);}
.history-header{background:var(--dark);padding:12px 18px;font-size:0.76rem;color:var(--gold);letter-spacing:1px;text-transform:uppercase;font-family:'Playfair Display',serif;}
.history-list{max-height:260px;overflow-y:auto;}
.history-item{display:flex;justify-content:space-between;align-items:center;padding:11px 18px;border-bottom:1px solid var(--border);}
.history-item:last-child{border-bottom:none;}
.history-loc{font-size:0.8rem;color:var(--dark);font-weight:500;}
.history-loc span{font-size:0.7rem;color:#8a7a65;display:block;font-weight:400;}
.history-price{font-family:'Playfair Display',serif;font-size:0.92rem;color:var(--gold);font-weight:700;}
.empty-history{padding:28px;text-align:center;color:#b0a090;font-size:0.82rem;}
.spinner{width:16px;height:16px;border:2px solid rgba(26,20,16,0.3);border-top-color:var(--dark);border-radius:50%;animation:spin 0.7s linear infinite;display:none;}
@keyframes spin{to{transform:rotate(360deg);}}
.error-msg{background:#fff0ee;border:1px solid #e0a090;border-radius:3px;padding:10px 14px;font-size:0.82rem;color:#8b3a2a;display:none;margin-top:10px;}
@media(max-width:800px){.main{grid-template-columns:1fr;}.field-grid.cols3{grid-template-columns:1fr 1fr;}}
</style>
<script>document.getElementById('nl-predict').className='nav-link active';</script>

<div class="page-header">
  <h1>Price Estimator</h1>
  <p>Fill in your property details below to get an AI-powered price estimate</p>
</div>

<div class="main">
  <div class="form-card">
    <div class="form-section">
      <div class="section-title">Location</div>
      <div class="field-grid">
        <div class="field"><label>District</label><div class="select-wrap"><select id="district" onchange="updateAreas()"><option value="">Select district…</option></select></div></div>
        <div class="field"><label>Area / City</label><div class="select-wrap"><select id="area"><option value="">Select area…</option></select></div></div>
      </div>
    </div>
    <div class="form-section">
      <div class="section-title">Property Details</div>
      <div class="field-grid cols3">
        <div class="field"><label>Land (Perch)</label><input type="number" id="perch" min="1" max="200" value="10"/></div>
        <div class="field"><label>Bedrooms</label><input type="number" id="bedrooms" min="1" max="10" value="3"/></div>
        <div class="field"><label>Bathrooms</label><input type="number" id="bathrooms" min="1" max="10" value="2"/></div>
        <div class="field"><label>Kitchen (sqft)</label><input type="number" id="kitchen_area_sqft" min="50" max="2000" value="150"/></div>
        <div class="field"><label>Parking Spots</label><input type="number" id="parking_spots" min="0" max="10" value="1"/></div>
        <div class="field"><label>Floors</label><input type="number" id="floors" min="1" max="10" value="2"/></div>
        <div class="field"><label>Year Built</label><input type="number" id="year_built" min="1950" max="2025" value="2010"/></div>
      </div>
    </div>
    <div class="form-section">
      <div class="section-title">Utilities & Amenities</div>
      <div class="field-grid" style="margin-bottom:14px;">
        <div class="field"><label>Water Supply</label><div class="select-wrap"><select id="water_supply"><option value="Pipe-borne">Pipe-borne</option><option value="Well">Well</option><option value="Both">Both</option></select></div></div>
        <div class="field"><label>Electricity</label><div class="select-wrap"><select id="electricity"><option value="Single phase">Single Phase</option><option value="Three phase">Three Phase</option></select></div></div>
      </div>
      <div class="toggle-row">
        <label class="toggle-item"><input type="checkbox" id="has_garden"/><div class="toggle-dot"></div><span class="toggle-label">Garden</span></label>
        <label class="toggle-item"><input type="checkbox" id="has_ac"/><div class="toggle-dot"></div><span class="toggle-label">Air Conditioning</span></label>
      </div>
    </div>
    <div class="form-footer">
      <button class="btn-predict" onclick="predict()"><span id="btn-text">Estimate Property Value</span><div class="spinner" id="spinner"></div></button>
      <div class="error-msg" id="error-msg"></div>
    </div>
  </div>

  <div class="result-panel">
    <div class="result-card" id="result-card">
      <div class="result-header">Estimated Market Value</div>
      <div class="result-body">
        <div class="price-display" id="price-display">—</div>
        <div class="price-sub" id="price-sub">Fill the form and click estimate</div>
        <div class="price-bar"><div class="price-fill" id="price-fill"></div></div>
        <div class="breakdown-grid" id="breakdown-grid"></div>
      </div>
    </div>
    <div class="history-card">
      <div class="history-header">Recent Estimates</div>
      <div class="history-list" id="history-list"><div class="empty-history">Your estimates will appear here</div></div>
    </div>
  </div>
</div>

<script>
const AREA_MAP={"Ampara":["Ampara Central"],"Anuradhapura":["Madawachchiya","New Town","Nuwaragam Palatha"],"Badulla":["Badulla Town","Bandarawela","Hali Ela"],"Batticaloa":["Batticaloa Town","Eravur","Kallady"],"Colombo":["Bambalapitiya","Borella","Dehiwala","Kollupitiya","Mount Lavinia","Narahenpita","Nugegoda","Rajagiriya","Wellawatte"],"Galle":["Galle Fort","Hikkaduwa","Karapitiya","Unawatuna"],"Gampaha":["Gampaha Town","Ja-Ela","Kadawatha","Negombo","Ragama","Wattala"],"Hambantota":["Ambalantota","Hambantota Town","Tangalle"],"Jaffna":["Chunnakam","Jaffna Town","Kokuvil","Nallur"],"Kalutara":["Beruwala","Kalutara North","Panadura","Wadduwa"],"Kandy":["Gatambe","Kandy City","Katugastota","Peradeniya","Tennekumbura"],"Kegalle":["Kegalle Central"],"Kilinochchi":["Kilinochchi Central"],"Kurunegala":["Kurunegala Town","Melsiripura","Pannala","Polgahawela"],"Mannar":["Mannar Central"],"Matale":["Matale Central"],"Matara":["Matara Town","Mirissa","Weligama"],"Monaragala":["Monaragala Central"],"Mullaitivu":["Mullaitivu Central"],"Nuwara Eliya":["Nuwara Eliya Central"],"Polonnaruwa":["Polonnaruwa Central"],"Puttalam":["Puttalam Central"],"Ratnapura":["Kuruwita","Ratnapura Town"],"Trincomalee":["Nilaveli","Trincomalee Town"],"Vavuniya":["Vavuniya Central"]};
const distSel=document.getElementById('district');
Object.keys(AREA_MAP).sort().forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;distSel.appendChild(o);});
function updateAreas(){const d=distSel.value,a=document.getElementById('area');a.innerHTML='<option value="">Select area…</option>';if(d&&AREA_MAP[d]){AREA_MAP[d].forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;a.appendChild(o);});a.value=AREA_MAP[d][0];}}
distSel.value='Colombo';updateAreas();
const history=[];
async function predict(){
  const btn=document.querySelector('.btn-predict'),sp=document.getElementById('spinner'),bt=document.getElementById('btn-text'),er=document.getElementById('error-msg');
  const district=distSel.value,area=document.getElementById('area').value;
  if(!district||!area){showError('Please select a district and area.');return;}
  const payload={district,area,perch:+document.getElementById('perch').value,bedrooms:+document.getElementById('bedrooms').value,bathrooms:+document.getElementById('bathrooms').value,kitchen_area_sqft:+document.getElementById('kitchen_area_sqft').value,parking_spots:+document.getElementById('parking_spots').value,has_garden:document.getElementById('has_garden').checked,has_ac:document.getElementById('has_ac').checked,water_supply:document.getElementById('water_supply').value,electricity:document.getElementById('electricity').value,floors:+document.getElementById('floors').value,year_built:+document.getElementById('year_built').value};
  btn.disabled=true;sp.style.display='block';bt.textContent='Estimating…';er.style.display='none';
  try{const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!res.ok)throw new Error();showResult(await res.json(),payload);}
  catch{showError('Prediction failed. Is the model loaded?');}
  finally{btn.disabled=false;sp.style.display='none';bt.textContent='Estimate Property Value';}
}
function showResult(data,p){
  const card=document.getElementById('result-card'),pe=document.getElementById('price-display'),sub=document.getElementById('price-sub'),fill=document.getElementById('price-fill'),grid=document.getElementById('breakdown-grid');
  const price=data.predicted_price_lkr,m=(price/1e6).toFixed(2);
  let s=0;const step=ts=>{if(!s)s=ts;const prog=Math.min((ts-s)/900,1),e=1-Math.pow(1-prog,3);pe.textContent='LKR '+(price*e/1e6).toFixed(2)+'M';if(prog<1)requestAnimationFrame(step);else pe.textContent=`LKR ${m}M`;};requestAnimationFrame(step);
  sub.textContent=`Range: LKR ${(price*.92/1e6).toFixed(1)}M – ${(price*1.08/1e6).toFixed(1)}M`;
  setTimeout(()=>fill.style.width=Math.min(((price-1500000)/(150000000-1500000))*100,100)+'%',100);
  const age=2025-p.year_built;
  grid.innerHTML=`<div class="breakdown-item"><div class="breakdown-label">District</div><div class="breakdown-value">${p.district}</div></div><div class="breakdown-item"><div class="breakdown-label">Area</div><div class="breakdown-value">${p.area}</div></div><div class="breakdown-item"><div class="breakdown-label">Land Size</div><div class="breakdown-value">${p.perch} perch</div></div><div class="breakdown-item"><div class="breakdown-label">House Age</div><div class="breakdown-value">${age} years</div></div><div class="breakdown-item"><div class="breakdown-label">Bedrooms</div><div class="breakdown-value">${p.bedrooms} bed / ${p.bathrooms} bath</div></div><div class="breakdown-item"><div class="breakdown-label">Amenities</div><div class="breakdown-value">${[p.has_garden&&'Garden',p.has_ac&&'AC'].filter(Boolean).join(', ')||'None'}</div></div>`;
  card.classList.add('visible');
  history.unshift({district:p.district,area:p.area,price:m});if(history.length>8)history.pop();renderHistory();
}
function renderHistory(){const el=document.getElementById('history-list');el.innerHTML=history.length?history.map(h=>`<div class="history-item"><div class="history-loc">${h.district}<span>${h.area}</span></div><div class="history-price">LKR ${h.price}M</div></div>`).join(''):'<div class="empty-history">Your estimates will appear here</div>';}
function showError(msg){const e=document.getElementById('error-msg');e.textContent=msg;e.style.display='block';}
</script>
"""

# ── PAGE 3: MARKET INSIGHTS ──────────────────────────────────────────────────
INSIGHTS_HTML = BASE_CSS + NAV + """
<style>
.page-header{background:var(--dark);padding:48px 48px 56px;}
.page-header h1{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;color:var(--cream);margin-bottom:10px;}
.page-header p{color:#a09880;font-size:0.95rem;}
.insights-body{max-width:1200px;margin:0 auto;padding:48px 32px;display:flex;flex-direction:column;gap:40px;}
.summary-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;}
.sum-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:24px;box-shadow:var(--shadow);text-align:center;}
.sum-num{font-family:'Playfair Display',serif;font-size:1.7rem;font-weight:700;color:var(--gold);display:block;margin-bottom:4px;}
.sum-label{font-size:0.72rem;color:#8a7a65;text-transform:uppercase;letter-spacing:1px;}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:28px;}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:28px;box-shadow:var(--shadow);}
.chart-card.full{grid-column:1/-1;}
.chart-title{font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;color:var(--dark);margin-bottom:4px;}
.chart-sub{font-size:0.75rem;color:#8a7a65;margin-bottom:22px;}
.bar-row{display:flex;align-items:center;gap:12px;margin-bottom:10px;}
.bar-label{font-size:0.78rem;color:var(--dark);width:120px;flex-shrink:0;text-overflow:ellipsis;overflow:hidden;white-space:nowrap;}
.bar-track{flex:1;height:26px;background:var(--cream);border-radius:3px;overflow:hidden;border:1px solid var(--border);}
.bar-fill{height:100%;background:linear-gradient(90deg,var(--sage),var(--gold));border-radius:3px;transition:width 1s ease;display:flex;align-items:center;padding-left:8px;}
.bar-val{font-size:0.7rem;font-weight:500;color:var(--dark);white-space:nowrap;}
.dist-table{width:100%;border-collapse:collapse;}
.dist-table th{text-align:left;font-size:0.7rem;color:#8a7a65;text-transform:uppercase;letter-spacing:0.8px;padding:8px 12px;border-bottom:2px solid var(--border);}
.dist-table td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:0.84rem;}
.dist-table tr:last-child td{border-bottom:none;}
.dist-table tr:hover td{background:var(--cream);}
.rank{font-size:0.7rem;color:#a09880;font-weight:600;}
.price-badge{display:inline-block;background:rgba(201,151,74,0.12);color:var(--gold);padding:3px 8px;border-radius:2px;font-weight:600;font-size:0.82rem;}
.donut-wrap{display:flex;flex-direction:column;gap:10px;}
.donut-row{display:flex;align-items:center;gap:10px;}
.donut-label{font-size:0.8rem;color:var(--dark);width:80px;}
.donut-bar{flex:1;height:22px;background:var(--cream);border-radius:2px;overflow:hidden;border:1px solid var(--border);}
.donut-fill{height:100%;border-radius:2px;transition:width 1.2s ease;}
.donut-val{font-size:0.75rem;color:#8a7a65;width:60px;text-align:right;}
@media(max-width:900px){.summary-cards{grid-template-columns:1fr 1fr;}.chart-row{grid-template-columns:1fr;}.chart-card.full{grid-column:1;}}
</style>
<script>document.getElementById('nl-insights').className='nav-link active';</script>

<div class="page-header">
  <h1>Market Insights</h1>
  <p>Real estate price trends across Sri Lanka — drawn from 20,000 property records</p>
</div>

<div class="insights-body">
  <div class="summary-cards" id="summary-cards">
    <div class="sum-card"><span class="sum-num" id="s-total">—</span><span class="sum-label">Properties</span></div>
    <div class="sum-card"><span class="sum-num" id="s-avg">—</span><span class="sum-label">Avg Price</span></div>
    <div class="sum-card"><span class="sum-num" id="s-median">—</span><span class="sum-label">Median Price</span></div>
    <div class="sum-card"><span class="sum-num" id="s-max">—</span><span class="sum-label">Highest Price</span></div>
  </div>

  <div class="chart-row">
    <div class="chart-card">
      <div class="chart-title">Average Price by District</div>
      <div class="chart-sub">Top 10 districts ranked by mean property price</div>
      <div id="district-chart">Loading…</div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Price Distribution</div>
      <div class="chart-sub">Number of properties in each price range</div>
      <div class="donut-wrap" id="dist-chart">Loading…</div>
    </div>
  </div>

  <div class="chart-row">
    <div class="chart-card">
      <div class="chart-title">Average Price by Bedroom Count</div>
      <div class="chart-sub">How price scales with number of bedrooms</div>
      <div id="bed-chart">Loading…</div>
    </div>
    <div class="chart-card full" style="grid-column:2">
      <div class="chart-title">Full District Price Table</div>
      <div class="chart-sub">All 25 districts ranked by average property price</div>
      <table class="dist-table">
        <thead><tr><th>#</th><th>District</th><th>Avg Price</th><th>Median Price</th><th>Properties</th></tr></thead>
        <tbody id="dist-table-body">Loading…</tbody>
      </table>
    </div>
  </div>
</div>

<script>
const COLORS=['#c9974a','#5a7a5a','#8b3a2a','#4a7a9b','#7a5a9b','#9b7a3a'];
async function loadData(){
  const r=await fetch('/api/market-data');const d=await r.json();
  // Summary
  document.getElementById('s-total').textContent=d.summary.total.toLocaleString();
  document.getElementById('s-avg').textContent='LKR '+(d.summary.avg_price/1e6).toFixed(1)+'M';
  document.getElementById('s-median').textContent='LKR '+(d.summary.median_price/1e6).toFixed(1)+'M';
  document.getElementById('s-max').textContent='LKR '+(d.summary.max_price/1e6).toFixed(0)+'M';
  // District bar chart (top 10)
  const top10=d.districts.slice(0,10);const maxVal=top10[0].mean;
  document.getElementById('district-chart').innerHTML=top10.map((x,i)=>`
    <div class="bar-row">
      <div class="bar-label" title="${x.district}">${x.district}</div>
      <div class="bar-track"><div class="bar-fill" style="width:0%;background:linear-gradient(90deg,${COLORS[i%COLORS.length]}88,${COLORS[i%COLORS.length]})" data-w="${(x.mean/maxVal*100).toFixed(1)}">
        <span class="bar-val">LKR ${(x.mean/1e6).toFixed(1)}M</span></div></div>
    </div>`).join('');
  setTimeout(()=>document.querySelectorAll('#district-chart .bar-fill').forEach(b=>b.style.width=b.dataset.w+'%'),100);
  // Price distribution
  const maxC=Math.max(...d.price_distribution.map(x=>x.count));
  const distColors=['#5a7a5a','#c9974a','#4a7a9b','#8b6a3a','#8b3a2a'];
  document.getElementById('dist-chart').innerHTML=d.price_distribution.map((x,i)=>`
    <div class="donut-row">
      <div class="donut-label">${x.range}</div>
      <div class="donut-bar"><div class="donut-fill" style="width:0%;background:${distColors[i]}" data-w="${(x.count/maxC*100).toFixed(1)}"></div></div>
      <div class="donut-val">${x.count.toLocaleString()}</div>
    </div>`).join('');
  setTimeout(()=>document.querySelectorAll('#dist-chart .donut-fill').forEach(b=>b.style.width=b.dataset.w+'%'),200);
  // Bedroom chart
  const maxB=Math.max(...d.by_bedrooms.map(x=>x.avg_price));
  document.getElementById('bed-chart').innerHTML=d.by_bedrooms.map((x,i)=>`
    <div class="bar-row">
      <div class="bar-label">${x.bedrooms} bedroom${x.bedrooms>1?'s':''}</div>
      <div class="bar-track"><div class="bar-fill" style="width:0%;background:linear-gradient(90deg,var(--sage),var(--gold))" data-w="${(x.avg_price/maxB*100).toFixed(1)}">
        <span class="bar-val">LKR ${(x.avg_price/1e6).toFixed(1)}M</span></div></div>
    </div>`).join('');
  setTimeout(()=>document.querySelectorAll('#bed-chart .bar-fill').forEach(b=>b.style.width=b.dataset.w+'%'),300);
  // Full table
  document.getElementById('dist-table-body').innerHTML=d.districts.map((x,i)=>`
    <tr><td><span class="rank">#${i+1}</span></td><td>${x.district}</td>
    <td><span class="price-badge">LKR ${(x.mean/1e6).toFixed(1)}M</span></td>
    <td>LKR ${(x.median/1e6).toFixed(1)}M</td><td>${x.count.toLocaleString()}</td></tr>`).join('');
}
loadData();
</script>
"""

# ── PAGE 4: MODEL PERFORMANCE ────────────────────────────────────────────────
MODEL_HTML = BASE_CSS + NAV + """
<style>
.page-header{background:var(--dark);padding:48px 48px 56px;}
.page-header h1{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:900;color:var(--cream);margin-bottom:10px;}
.page-header p{color:#a09880;font-size:0.95rem;}
.model-body{max-width:1200px;margin:0 auto;padding:48px 32px;display:flex;flex-direction:column;gap:36px;}
.model-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;}
.model-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:28px;box-shadow:var(--shadow);position:relative;overflow:hidden;}
.model-card.best{border-color:var(--gold);box-shadow:0 4px 40px rgba(201,151,74,0.2);}
.best-badge{position:absolute;top:16px;right:16px;background:var(--gold);color:var(--dark);font-size:0.65rem;font-weight:700;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:1px;}
.model-name{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:var(--dark);margin-bottom:18px;}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border);}
.metric-row:last-child{border-bottom:none;}
.metric-name{font-size:0.75rem;color:#8a7a65;text-transform:uppercase;letter-spacing:0.8px;}
.metric-val{font-size:0.92rem;font-weight:600;color:var(--dark);}
.metric-val.good{color:var(--sage);}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:28px;}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:28px;box-shadow:var(--shadow);}
.chart-title{font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;color:var(--dark);margin-bottom:4px;}
.chart-sub{font-size:0.75rem;color:#8a7a65;margin-bottom:22px;}
.bar-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.bar-label{font-size:0.8rem;color:var(--dark);width:160px;flex-shrink:0;}
.bar-track{flex:1;height:28px;background:var(--cream);border-radius:3px;overflow:hidden;border:1px solid var(--border);}
.bar-fill{height:100%;border-radius:3px;transition:width 1.1s ease;display:flex;align-items:center;padding-left:10px;}
.bar-val{font-size:0.72rem;font-weight:500;color:white;white-space:nowrap;}
.model-compare{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--border);border-radius:4px;overflow:hidden;}
.compare-col{display:flex;flex-direction:column;}
.compare-head{background:var(--dark);padding:16px;text-align:center;}
.compare-head.gold-head{background:var(--gold);}
.compare-model{font-family:'Playfair Display',serif;font-size:0.9rem;font-weight:700;color:var(--cream);}
.compare-head.gold-head .compare-model{color:var(--dark);}
.compare-cell{padding:14px 16px;text-align:center;border-bottom:1px solid var(--border);font-size:0.88rem;}
.compare-cell:last-child{border-bottom:none;}
.compare-cell.best-cell{background:rgba(201,151,74,0.08);font-weight:600;color:var(--gold);}
.info-box{background:var(--dark);border-radius:4px;padding:28px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;}
.info-item .info-title{font-size:0.75rem;color:var(--gold);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.info-item .info-text{font-size:0.84rem;color:#a09880;line-height:1.65;}
@media(max-width:900px){.model-cards{grid-template-columns:1fr;}.chart-row{grid-template-columns:1fr;}.model-compare{grid-template-columns:1fr;}.info-box{grid-template-columns:1fr;}}
</style>
<script>document.getElementById('nl-model').className='nav-link active';</script>

<div class="page-header">
  <h1>Model Performance</h1>
  <p>Compare all three trained models — see what works, why, and what drives property prices</p>
</div>

<div class="model-body">

  <!-- Model cards -->
  <div class="model-cards">
    <div class="model-card">
      <div class="model-name">Ridge Regression</div>
      <div class="metric-row"><span class="metric-name">R² Score</span><span class="metric-val">0.877</span></div>
      <div class="metric-row"><span class="metric-name">MAE</span><span class="metric-val">LKR 2.9M</span></div>
      <div class="metric-row"><span class="metric-name">RMSE</span><span class="metric-val">LKR 4.7M</span></div>
      <div class="metric-row"><span class="metric-name">MAPE</span><span class="metric-val">26.4%</span></div>
      <div class="metric-row"><span class="metric-name">Training Time</span><span class="metric-val">~8s</span></div>
    </div>
    <div class="model-card">
      <div class="model-name">Random Forest</div>
      <div class="metric-row"><span class="metric-name">R² Score</span><span class="metric-val">0.955</span></div>
      <div class="metric-row"><span class="metric-name">MAE</span><span class="metric-val">LKR 1.8M</span></div>
      <div class="metric-row"><span class="metric-name">RMSE</span><span class="metric-val">LKR 2.8M</span></div>
      <div class="metric-row"><span class="metric-name">MAPE</span><span class="metric-val">13.1%</span></div>
      <div class="metric-row"><span class="metric-name">Training Time</span><span class="metric-val">~5s</span></div>
    </div>
    <div class="model-card best">
      <div class="best-badge">Best Model</div>
      <div class="model-name">Gradient Boosting</div>
      <div class="metric-row"><span class="metric-name">R² Score</span><span class="metric-val good">0.966</span></div>
      <div class="metric-row"><span class="metric-name">MAE</span><span class="metric-val good">LKR 1.6M</span></div>
      <div class="metric-row"><span class="metric-name">RMSE</span><span class="metric-val good">LKR 2.5M</span></div>
      <div class="metric-row"><span class="metric-name">MAPE</span><span class="metric-val good">11.3%</span></div>
      <div class="metric-row"><span class="metric-name">Training Time</span><span class="metric-val">~12s</span></div>
    </div>
  </div>

  <!-- Feature importance + R2 comparison -->
  <div class="chart-row">
    <div class="chart-card">
      <div class="chart-title">Feature Importance (Correlation with Price)</div>
      <div class="chart-sub">How strongly each feature correlates with house price</div>
      <div id="feat-chart">
        <div class="bar-row"><div class="bar-label">Land Size (Perch)</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:var(--gold)" data-w="100"><span class="bar-val">0.825</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Kitchen Area (sqft)</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:var(--gold)" data-w="98"><span class="bar-val">0.808</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Bedrooms</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#5a7a5a" data-w="77"><span class="bar-val">0.632</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Bathrooms</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#5a7a5a" data-w="74"><span class="bar-val">0.607</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Floors</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#4a7a9b" data-w="58"><span class="bar-val">0.475</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Parking Spots</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#4a7a9b" data-w="42"><span class="bar-val">0.348</span></div></div></div>
        <div class="bar-row"><div class="bar-label">Year Built</div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#8a7a65" data-w="7"><span class="bar-val">0.060</span></div></div></div>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">R² Score Comparison</div>
      <div class="chart-sub">Higher is better — how much variance the model explains</div>
      <div class="bar-row" style="margin-top:8px;">
        <div class="bar-label">Ridge</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:#8b3a2a" data-w="87.7"><span class="bar-val">0.877 (87.7%)</span></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Random Forest</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:#4a7a9b" data-w="95.5"><span class="bar-val">0.955 (95.5%)</span></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Gradient Boosting</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:var(--gold)" data-w="96.6"><span class="bar-val">0.966 (96.6%)</span></div></div>
      </div>

      <div class="chart-title" style="margin-top:28px;">MAE Comparison</div>
      <div class="chart-sub">Lower is better — average prediction error in LKR</div>
      <div class="bar-row" style="margin-top:8px;">
        <div class="bar-label">Ridge</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:#8b3a2a" data-w="100"><span class="bar-val">LKR 2.9M</span></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Random Forest</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:#4a7a9b" data-w="62"><span class="bar-val">LKR 1.8M</span></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Gradient Boosting</div>
        <div class="bar-track"><div class="bar-fill" style="width:0%;background:var(--gold)" data-w="55"><span class="bar-val">LKR 1.6M</span></div></div>
      </div>
    </div>
  </div>

  <!-- MLOps info -->
  <div class="info-box">
    <div class="info-item"><div class="info-title">Experiment Tracking</div><div class="info-text">Every training run is logged to MLflow with full metrics, hyperparameters, and model artifacts. View all runs at <strong style="color:var(--gold)">localhost:5000</strong>.</div></div>
    <div class="info-item"><div class="info-title">Feature Engineering</div><div class="info-text">Raw features are enriched with derived signals: house_age (2025 - year_built), amenity_score (garden + AC + parking), and bed_bath_ratio — reducing MAPE from 32% to 11%.</div></div>
    <div class="info-item"><div class="info-title">Live Retraining</div><div class="info-text">Call <strong style="color:var(--gold)">POST /retrain</strong> to retrain all 3 models on fresh data and hot-reload the best one without restarting the server. Monitor drift with <strong style="color:var(--gold)">src/monitor.py</strong>.</div></div>
  </div>
</div>

<script>
setTimeout(()=>{
  document.querySelectorAll('.bar-fill[data-w]').forEach(b=>b.style.width=b.dataset.w+'%');
},200);
</script>
"""

# ── ROUTES ───────────────────────────────────────────────────────────────────
def wrap(body, active_id):
    return body.replace(f'id="{active_id}"', f'id="{active_id}" class="nav-link active"')

@app.get("/", response_class=HTMLResponse)
def home(): return f"<!DOCTYPE html><html><head><title>LankaHome Value</title></head><body>{HOME_HTML}</body></html>"

@app.get("/predict", response_class=HTMLResponse)
def predict_page(): return f"<!DOCTYPE html><html><head><title>Predict — LankaHome Value</title></head><body>{PREDICT_HTML}</body></html>"

@app.get("/insights", response_class=HTMLResponse)
def insights_page(): return f"<!DOCTYPE html><html><head><title>Market Insights — LankaHome Value</title></head><body>{INSIGHTS_HTML}</body></html>"

@app.get("/model", response_class=HTMLResponse)
def model_page(): return f"<!DOCTYPE html><html><head><title>Model Performance — LankaHome Value</title></head><body>{MODEL_HTML}</body></html>"