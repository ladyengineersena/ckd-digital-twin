from fastapi import FastAPI
from pydantic import BaseModel
import os, joblib, yaml, numpy as np
from typing import Optional

app = FastAPI(title="CKD Digital Twin API")

cfg_path = os.path.join("configs","training_config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

MODEL = None
MODEL_PATH = "outputs/lgbm_model.joblib"

if os.path.exists(MODEL_PATH):
    MODEL = joblib.load(MODEL_PATH)

class SummaryRequest(BaseModel):
    mean_egfr_30d: float
    sd_egfr_30d: float
    last_egfr: float
    slope_egfr: float
    age: float
    diabetes: int
    htn: int
    acei: int

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict_summary")
def predict_summary(req: SummaryRequest):
    global MODEL
    if MODEL is None:
        return {"error":"model not loaded"}
    x = np.array([[req.mean_egfr_30d, req.sd_egfr_30d, req.last_egfr, req.slope_egfr, req.age, req.diabetes, req.htn, req.acei]])
    prob = float(MODEL.predict_proba(x)[:,1][0])
    return {"risk_of_severe_decline": prob}

@app.get("/forecast/{patient_id}")
def forecast(patient_id: str, model_path: Optional[str] = "outputs/lstm_model.pt", horizon: int = 90):
    # lightweight dispatcher: spawn twin forecast script
    # For safety, this is a synchronous call that requires model to exist.
    if not os.path.exists(model_path):
        return {"error":"lstm model not found"}
    # call twin forecast (imported)
    from src.simulator.twin import load_model, forecast_patient
    from src.preprocess.features import load_data, build_sequence_data
    device = "cuda" if False else "cpu"
    meta, labs, meds, events = load_data("data/synthetic")
    X_seq, X_tab, pids = build_sequence_data(meta, labs, seq_len=90)
    model = load_model(model_path, device)
    pids = list(pids)
    if patient_id not in pids:
        return {"error":"patient_id not found in synthetic data"}
    preds = forecast_patient(patient_id, model, X_seq, X_tab, pids, horizon=horizon, device=device)
    return {"patient_id": patient_id, "forecast": preds}

