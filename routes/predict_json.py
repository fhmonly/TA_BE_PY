from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error

router = APIRouter()

class TimeSeriesData(BaseModel):
    date: List[str]
    value: List[float]

class PredictionRequest(BaseModel):
    data: TimeSeriesData
    model: Literal['optimal', 'custom', 'auto'] = "auto"
    forecast_step: int
    order: Optional[List[int]] = None

def determine_d(series):
    """ Menentukan jumlah differencing (d) berdasarkan uji Augmented Dickey-Fuller """
    d = 0
    while adfuller(series)[1] > 0.05 and d < 2:
        series = series.diff().dropna()
        d += 1
    return d

def determine_p_q(series):
    """ Menentukan p dan q berdasarkan ACF dan PACF """
    acf_vals = acf(series.dropna(), nlags=10)
    pacf_vals = pacf(series.dropna(), nlags=10)
    p = next((i for i, v in enumerate(pacf_vals[1:], start=1) if abs(v) > 0.2), 1)
    q = next((i for i, v in enumerate(acf_vals[1:], start=1) if abs(v) > 0.2), 1)
    return p, q

@router.post("/predict-json")
async def predict_json(request: PredictionRequest):
    if len(request.data.date) != len(request.data.value):
        raise HTTPException(status_code=400, detail="Date and value lists must have the same length.")

    try:
        df = pd.DataFrame({"date": pd.to_datetime(request.data.date), "value": request.data.value})
        df = df.dropna().sort_values(by="date").set_index("date")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")

    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Insufficient data: At least 60 records required.")

    train_size = int(len(df) * 0.7)
    train, test = df[:train_size], df[train_size:]

    if request.model == "auto":
        d = determine_d(train["value"])
        p, q = determine_p_q(train["value"])
    elif request.model == "optimal":
        p, d, q = 2, 1, 2
    elif request.model == "custom":
        if not request.order or len(request.order) != 3:
            raise HTTPException(status_code=400, detail="Custom model requires an array of [p, d, q].")
        p, d, q = request.order
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'auto', 'optimal', or 'custom'.")

    try:
        arima_model = ARIMA(train["value"], order=(p, d, q))
        model_fit = arima_model.fit()
        predictions = model_fit.forecast(steps=len(test)).tolist()
        rmse = np.sqrt(mean_squared_error(test["value"], predictions))
        future_forecast = model_fit.forecast(steps=request.forecast_step).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {str(e)}")

    return {
        "arima_order": [p, d, q],
        "rmse": rmse,
        "forecast": future_forecast
    }
