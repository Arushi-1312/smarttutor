from fastapi import FastAPI
import joblib, pandas as pd
from pathlib import Path
from features import add_features
app = FastAPI(title="SmartTutor API")
MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "model.joblib"
model_bundle = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
@app.get("/")
def root(): return {"status":"ok"}
@app.post("/predict")
def predict(payload: dict):
    if not model_bundle: return {"error":"Model not trained"}
    df = pd.DataFrame([payload])
    df = add_features(df)
    X = df[model_bundle["features"]]
    prob = model_bundle["model"].predict_proba(X)[:,1][0]
    return {"prob_high_performer": float(prob), "pred": int(prob>0.5)}
