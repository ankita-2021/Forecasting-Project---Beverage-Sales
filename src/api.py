# src/api.py — FastAPI REST API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import holidays
import os

app = FastAPI(title="Beverage Sales Forecasting API")

# Models load karo
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

def load_model(state):
    path = os.path.join(MODEL_DIR, f"xgb_{state.replace(' ', '_')}.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model not found for state: {state}")
    return joblib.load(path)

# Request body
class ForecastRequest(BaseModel):
    state: str
    start_date: str        # Format: "2024-01-01"
    lag_1: float
    lag_4: float
    lag_8: float
    rolling_mean_4: float
    rolling_std_4: float
    rolling_mean_8: float

# --- Routes ---

@app.get("/")
def root():
    return {"message": "Beverage Sales Forecasting API is running!"}

@app.get("/states")
def get_states():
    states = [f.replace("xgb_", "").replace(".pkl", "").replace("_", " ")
              for f in os.listdir(MODEL_DIR) if f.startswith("xgb_")]
    return {"states": sorted(states)}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    model = load_model(req.state)
    us_holidays = holidays.US(years=range(2024, 2026))

    start = pd.to_datetime(req.start_date)
    dates = [start + pd.Timedelta(weeks=i) for i in range(8)]

    predictions = []
    lag_1 = req.lag_1
    lag_4 = req.lag_4
    lag_8 = req.lag_8
    rolling_mean_4 = req.rolling_mean_4
    rolling_std_4 = req.rolling_std_4
    rolling_mean_8 = req.rolling_mean_8
    recent_preds = []

    for date in dates:
        row = {
            'week': date.isocalendar()[1],
            'month': date.month,
            'quarter': date.quarter,
            'year': date.year,
            'dayofweek': date.dayofweek,
            'is_holiday': 1 if date.date() in us_holidays else 0,
            'lag_1': lag_1,
            'lag_4': lag_4,
            'lag_8': lag_8,
            'rolling_mean_4': rolling_mean_4,
            'rolling_std_4': rolling_std_4,
            'rolling_mean_8': rolling_mean_8
        }

        X = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X)[0])
        predictions.append({"date": str(date.date()), "predicted_sales": round(pred, 2)})

        # Update lags
        recent_preds.append(pred)
        lag_8 = lag_4
        lag_4 = lag_1
        lag_1 = pred
        if len(recent_preds) >= 4:
            rolling_mean_4 = float(np.mean(recent_preds[-4:]))
            rolling_std_4 = float(np.std(recent_preds[-4:]))
        if len(recent_preds) >= 8:
            rolling_mean_8 = float(np.mean(recent_preds[-8:]))

    return {
        "state": req.state,
        "model": "XGBoost",
        "forecast_weeks": 8,
        "predictions": predictions
    }