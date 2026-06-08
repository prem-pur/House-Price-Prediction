"""
main.py — FastAPI + embedded frontend for SL House Price Prediction
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
    return {
        "predicted_price_lkr": round(price, 2),
        "predicted_price_lkr_formatted": f"LKR {price:,.0f}"
    }

@app.post("/retrain")
def retrain():
    """Retrain the model with fresh data and reload into memory."""
    import subprocess, sys
    global model, feature_columns
    train_script = os.path.join(os.path.dirname(__file__), "..", "src", "train.py")
    result = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")
    # Reload model
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    cols_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    return {"status": "retrained", "message": "Model retrained and reloaded successfully"}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Lanka Home Value — AI Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --cream: #f5f0e8;
    --dark: #1a1410;
    --gold: #c9974a;
    --gold-light: #e8c07a;
    --rust: #8b3a2a;
    --sage: #5a7a5a;
    --card: #fffdf8;
    --border: #e0d8c8;
    --shadow: 0 4px 40px rgba(26,20,16,0.10);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--cream); color: var(--dark); font-family: 'DM Sans', sans-serif; min-height: 100vh; overflow-x: hidden; }
  header { background: var(--dark); padding: 0 48px; display: flex; align-items: center; justify-content: space-between; height: 72px; position: sticky; top: 0; z-index: 100; border-bottom: 2px solid var(--gold); }
  .logo { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 900; color: var(--cream); letter-spacing: -0.5px; }
  .logo span { color: var(--gold); }
  .badge { background: var(--gold); color: var(--dark); font-size: 0.7rem; font-weight: 500; padding: 4px 12px; border-radius: 20px; letter-spacing: 1px; text-transform: uppercase; }
  .hero { background: var(--dark); padding: 72px 48px 80px; position: relative; overflow: hidden; }
  .hero::before { content: ''; position: absolute; top: -60px; right: -60px; width: 400px; height: 400px; border-radius: 50%; background: radial-gradient(circle, rgba(201,151,74,0.15) 0%, transparent 70%); pointer-events: none; }
  .hero-tag { display: inline-block; border: 1px solid var(--gold); color: var(--gold); font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; padding: 6px 16px; border-radius: 2px; margin-bottom: 24px; }
  .hero h1 { font-family: 'Playfair Display', serif; font-size: clamp(2.4rem, 5vw, 4rem); font-weight: 900; color: var(--cream); line-height: 1.1; max-width: 640px; margin-bottom: 18px; }
  .hero h1 em { color: var(--gold); font-style: normal; }
  .hero p { color: #a09880; font-size: 1.05rem; font-weight: 300; max-width: 480px; line-height: 1.7; }
  .stats-bar { background: var(--gold); display: flex; }
  .stat { flex: 1; padding: 18px 28px; border-right: 1px solid rgba(26,20,16,0.2); text-align: center; }
  .stat:last-child { border-right: none; }
  .stat-num { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: var(--dark); display: block; }
  .stat-label { font-size: 0.72rem; font-weight: 500; color: rgba(26,20,16,0.65); text-transform: uppercase; letter-spacing: 1px; }
  .main { max-width: 1100px; margin: 0 auto; padding: 56px 32px; display: grid; grid-template-columns: 1fr 380px; gap: 40px; align-items: start; }
  .form-card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; box-shadow: var(--shadow); }
  .form-section { padding: 28px 32px; border-bottom: 1px solid var(--border); }
  .form-section:last-of-type { border-bottom: none; }
  .section-title { font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 700; color: var(--dark); margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
  .section-title::before { content: ''; display: block; width: 4px; height: 18px; background: var(--gold); border-radius: 2px; flex-shrink: 0; }
  .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .field-grid.cols3 { grid-template-columns: 1fr 1fr 1fr; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 0.75rem; font-weight: 500; color: #6b5f50; text-transform: uppercase; letter-spacing: 0.8px; }
  select, input[type="number"] { width: 100%; padding: 11px 14px; background: var(--cream); border: 1px solid var(--border); border-radius: 3px; font-family: 'DM Sans', sans-serif; font-size: 0.92rem; color: var(--dark); appearance: none; transition: border-color 0.2s, box-shadow 0.2s; outline: none; }
  select:focus, input[type="number"]:focus { border-color: var(--gold); box-shadow: 0 0 0 3px rgba(201,151,74,0.15); }
  .select-wrap { position: relative; }
  .select-wrap::after { content: '▾'; position: absolute; right: 14px; top: 50%; transform: translateY(-50%); color: var(--gold); pointer-events: none; font-size: 0.8rem; }
  .toggle-row { display: flex; gap: 12px; }
  .toggle-item { flex: 1; display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: var(--cream); border: 1px solid var(--border); border-radius: 3px; cursor: pointer; transition: border-color 0.2s, background 0.2s; user-select: none; }
  .toggle-item:has(input:checked) { border-color: var(--gold); background: rgba(201,151,74,0.08); }
  .toggle-item input { display: none; }
  .toggle-dot { width: 18px; height: 18px; border-radius: 50%; border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: border-color 0.2s, background 0.2s; }
  .toggle-item:has(input:checked) .toggle-dot { border-color: var(--gold); background: var(--gold); }
  .toggle-dot::after { content: '✓'; color: var(--dark); font-size: 0.65rem; font-weight: 700; opacity: 0; transition: opacity 0.15s; }
  .toggle-item:has(input:checked) .toggle-dot::after { opacity: 1; }
  .toggle-label { font-size: 0.88rem; color: var(--dark); }
  .form-footer { padding: 24px 32px; background: var(--dark); }
  .btn-predict { width: 100%; padding: 16px; background: var(--gold); border: none; border-radius: 3px; font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 500; color: var(--dark); cursor: pointer; letter-spacing: 0.5px; transition: background 0.2s, transform 0.1s; display: flex; align-items: center; justify-content: center; gap: 10px; }
  .btn-predict:hover { background: var(--gold-light); }
  .btn-predict:active { transform: scale(0.99); }
  .btn-predict.loading { opacity: 0.7; cursor: not-allowed; }
  .result-panel { display: flex; flex-direction: column; gap: 20px; }
  .result-card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; box-shadow: var(--shadow); opacity: 0; transform: translateY(12px); transition: opacity 0.4s ease, transform 0.4s ease; }
  .result-card.visible { opacity: 1; transform: translateY(0); }
  .result-header { background: var(--dark); padding: 18px 24px; font-family: 'Playfair Display', serif; font-size: 0.85rem; color: var(--gold); letter-spacing: 1px; text-transform: uppercase; }
  .result-body { padding: 28px 24px; }
  .price-display { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 700; color: var(--dark); line-height: 1; margin-bottom: 6px; }
  .price-sub { font-size: 0.82rem; color: #8a7a65; margin-bottom: 20px; }
  .price-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 20px; }
  .price-fill { height: 100%; background: linear-gradient(90deg, var(--sage), var(--gold)); border-radius: 3px; width: 0%; transition: width 1s ease; }
  .breakdown-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .breakdown-item { background: var(--cream); border-radius: 3px; padding: 12px; border: 1px solid var(--border); }
  .breakdown-label { font-size: 0.68rem; color: #8a7a65; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
  .breakdown-value { font-size: 0.95rem; font-weight: 500; color: var(--dark); }
  .history-card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; box-shadow: var(--shadow); }
  .history-header { background: var(--dark); padding: 14px 20px; font-size: 0.78rem; color: var(--gold); letter-spacing: 1px; text-transform: uppercase; font-family: 'Playfair Display', serif; }
  .history-list { max-height: 280px; overflow-y: auto; }
  .history-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; border-bottom: 1px solid var(--border); animation: slideIn 0.3s ease; }
  .history-item:last-child { border-bottom: none; }
  @keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: none; } }
  .history-loc { font-size: 0.82rem; color: var(--dark); font-weight: 500; }
  .history-loc span { font-size: 0.72rem; color: #8a7a65; display: block; font-weight: 400; }
  .history-price { font-family: 'Playfair Display', serif; font-size: 0.95rem; color: var(--gold); font-weight: 700; }
  .empty-history { padding: 32px 20px; text-align: center; color: #b0a090; font-size: 0.85rem; }
  .spinner { width: 18px; height: 18px; border: 2px solid rgba(26,20,16,0.3); border-top-color: var(--dark); border-radius: 50%; animation: spin 0.7s linear infinite; display: none; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error-msg { background: #fff0ee; border: 1px solid #e0a090; border-radius: 3px; padding: 12px 16px; font-size: 0.85rem; color: var(--rust); display: none; margin-top: 12px; }
  @media (max-width: 800px) { .main { grid-template-columns: 1fr; padding: 32px 16px; } .hero { padding: 48px 24px 56px; } header { padding: 0 24px; } .stats-bar { flex-wrap: wrap; } .stat { min-width: 50%; } .field-grid.cols3 { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<header>
  <div class="logo">Lanka<span>Home</span> Value</div>
  <div class="badge">AI Powered</div>
</header>
<section class="hero">
  <div class="hero-tag">Sri Lanka Real Estate Intelligence</div>
  <h1>Predict Your Home's <em>True Value</em></h1>
  <p>Powered by Gradient Boosting trained on 20,000 Sri Lankan properties. Get an instant, data-driven price estimate.</p>
</section>
<div class="stats-bar">
  <div class="stat"><span class="stat-num">20,000</span><span class="stat-label">Properties Trained</span></div>
  <div class="stat"><span class="stat-num">96.6%</span><span class="stat-label">Model Accuracy (R²)</span></div>
  <div class="stat"><span class="stat-num">25</span><span class="stat-label">Districts Covered</span></div>
  <div class="stat"><span class="stat-num">LKR 1.6M</span><span class="stat-label">Avg Error (MAE)</span></div>
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
      <div class="field-grid" style="margin-bottom:16px;">
        <div class="field"><label>Water Supply</label><div class="select-wrap"><select id="water_supply"><option value="Pipe-borne">Pipe-borne</option><option value="Well">Well</option><option value="Both">Both</option></select></div></div>
        <div class="field"><label>Electricity</label><div class="select-wrap"><select id="electricity"><option value="Single phase">Single Phase</option><option value="Three phase">Three Phase</option></select></div></div>
      </div>
      <div class="toggle-row">
        <label class="toggle-item"><input type="checkbox" id="has_garden"/><div class="toggle-dot"></div><span class="toggle-label">🌿 Garden</span></label>
        <label class="toggle-item"><input type="checkbox" id="has_ac"/><div class="toggle-dot"></div><span class="toggle-label">❄️ Air Conditioning</span></label>
      </div>
    </div>
    <div class="form-footer">
      <button class="btn-predict" onclick="predict()">
        <span id="btn-text">Estimate Property Value</span>
        <div class="spinner" id="spinner"></div>
      </button>
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
const AREA_MAP = {
  "Ampara":["Ampara Central"],"Anuradhapura":["Madawachchiya","New Town","Nuwaragam Palatha"],
  "Badulla":["Badulla Town","Bandarawela","Hali Ela"],"Batticaloa":["Batticaloa Town","Eravur","Kallady"],
  "Colombo":["Bambalapitiya","Borella","Dehiwala","Kollupitiya","Mount Lavinia","Narahenpita","Nugegoda","Rajagiriya","Wellawatte"],
  "Galle":["Galle Fort","Hikkaduwa","Karapitiya","Unawatuna"],"Gampaha":["Gampaha Town","Ja-Ela","Kadawatha","Negombo","Ragama","Wattala"],
  "Hambantota":["Ambalantota","Hambantota Town","Tangalle"],"Jaffna":["Chunnakam","Jaffna Town","Kokuvil","Nallur"],
  "Kalutara":["Beruwala","Kalutara North","Panadura","Wadduwa"],"Kandy":["Gatambe","Kandy City","Katugastota","Peradeniya","Tennekumbura"],
  "Kegalle":["Kegalle Central"],"Kilinochchi":["Kilinochchi Central"],"Kurunegala":["Kurunegala Town","Melsiripura","Pannala","Polgahawela"],
  "Mannar":["Mannar Central"],"Matale":["Matale Central"],"Matara":["Matara Town","Mirissa","Weligama"],
  "Monaragala":["Monaragala Central"],"Mullaitivu":["Mullaitivu Central"],"Nuwara Eliya":["Nuwara Eliya Central"],
  "Polonnaruwa":["Polonnaruwa Central"],"Puttalam":["Puttalam Central"],"Ratnapura":["Kuruwita","Ratnapura Town"],
  "Trincomalee":["Nilaveli","Trincomalee Town"],"Vavuniya":["Vavuniya Central"]
};
const districtSel = document.getElementById('district');
Object.keys(AREA_MAP).sort().forEach(d => { const o = document.createElement('option'); o.value=d; o.textContent=d; districtSel.appendChild(o); });
function updateAreas() {
  const d = districtSel.value, a = document.getElementById('area');
  a.innerHTML = '<option value="">Select area…</option>';
  if (d && AREA_MAP[d]) { AREA_MAP[d].forEach(x => { const o=document.createElement('option'); o.value=x; o.textContent=x; a.appendChild(o); }); a.value=AREA_MAP[d][0]; }
}
districtSel.value='Colombo'; updateAreas();
const history=[];
async function predict() {
  const btn=document.querySelector('.btn-predict'), spinner=document.getElementById('spinner'), btnText=document.getElementById('btn-text'), errorEl=document.getElementById('error-msg');
  const district=document.getElementById('district').value, area=document.getElementById('area').value;
  if (!district||!area) { showError('Please select a district and area.'); return; }
  const payload={district,area,perch:+document.getElementById('perch').value,bedrooms:+document.getElementById('bedrooms').value,bathrooms:+document.getElementById('bathrooms').value,kitchen_area_sqft:+document.getElementById('kitchen_area_sqft').value,parking_spots:+document.getElementById('parking_spots').value,has_garden:document.getElementById('has_garden').checked,has_ac:document.getElementById('has_ac').checked,water_supply:document.getElementById('water_supply').value,electricity:document.getElementById('electricity').value,floors:+document.getElementById('floors').value,year_built:+document.getElementById('year_built').value};
  btn.classList.add('loading'); spinner.style.display='block'; btnText.textContent='Estimating…'; errorEl.style.display='none';
  try {
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if (!res.ok) throw new Error();
    showResult(await res.json(), payload);
  } catch { showError('Prediction failed. Is the model loaded?'); }
  finally { btn.classList.remove('loading'); spinner.style.display='none'; btnText.textContent='Estimate Property Value'; }
}
function showResult(data, payload) {
  const card=document.getElementById('result-card'), priceEl=document.getElementById('price-display'), subEl=document.getElementById('price-sub'), fillEl=document.getElementById('price-fill'), gridEl=document.getElementById('breakdown-grid');
  const price=data.predicted_price_lkr, millions=(price/1e6).toFixed(2);
  let start=0;
  const step=(ts)=>{ if(!start)start=ts; const p=Math.min((ts-start)/900,1), e=1-Math.pow(1-p,3); priceEl.textContent='LKR '+(price*e/1e6).toFixed(2)+'M'; if(p<1)requestAnimationFrame(step); else priceEl.textContent=`LKR ${millions}M`; };
  requestAnimationFrame(step);
  subEl.textContent=`Range: LKR ${(price*.92/1e6).toFixed(1)}M – ${(price*1.08/1e6).toFixed(1)}M`;
  setTimeout(()=>fillEl.style.width=Math.min(((price-1500000)/(150000000-1500000))*100,100)+'%',100);
  const age=2025-payload.year_built;
  gridEl.innerHTML=`<div class="breakdown-item"><div class="breakdown-label">District</div><div class="breakdown-value">${payload.district}</div></div><div class="breakdown-item"><div class="breakdown-label">Area</div><div class="breakdown-value">${payload.area}</div></div><div class="breakdown-item"><div class="breakdown-label">Land Size</div><div class="breakdown-value">${payload.perch} perch</div></div><div class="breakdown-item"><div class="breakdown-label">House Age</div><div class="breakdown-value">${age} years</div></div><div class="breakdown-item"><div class="breakdown-label">Bedrooms</div><div class="breakdown-value">${payload.bedrooms} bed / ${payload.bathrooms} bath</div></div><div class="breakdown-item"><div class="breakdown-label">Amenities</div><div class="breakdown-value">${[payload.has_garden&&'🌿 Garden',payload.has_ac&&'❄️ AC'].filter(Boolean).join(', ')||'None'}</div></div>`;
  card.classList.add('visible');
  history.unshift({district:payload.district,area:payload.area,price:millions});
  if(history.length>8)history.pop();
  renderHistory();
}
function renderHistory() {
  const el=document.getElementById('history-list');
  el.innerHTML=history.length?history.map(h=>`<div class="history-item"><div class="history-loc">${h.district}<span>${h.area}</span></div><div class="history-price">LKR ${h.price}M</div></div>`).join(''):'<div class="empty-history">Your estimates will appear here</div>';
}
function showError(msg) { const e=document.getElementById('error-msg'); e.textContent=msg; e.style.display='block'; }
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
