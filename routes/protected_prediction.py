from fastapi import APIRouter, HTTPException
from schema.prediction import (
    AutoPredictionRequest,
    AutoPredictionResponse,
    ManualPredictionRequest,
    ManualPredictionResponse,
)
from utils.statistic.auto_arima import auto_arima_forecast
from utils.statistic.manual_arima import manual_arima_forecast
from utils.data_preparation import read_csv_string_to_df, df_group_by_interval

router = APIRouter()

@router.post("/predict/private/auto", response_model=AutoPredictionResponse)
def predict_auto(request: AutoPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)

        series = df['amount']

        result = auto_arima_forecast(series, forecast_periods=request.future_steps)

        return AutoPredictionResponse(
            rmse=result["rmse"],
            mape=result["mape"],
            arima_order=tuple(result["arima_order"]),
            prediction=result["prediction"],
            lower=result["lower"],
            upper=result["upper"],
            success=True
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses data: {e}")

@router.post("/predict/private/manual", response_model=ManualPredictionResponse)
def predict_manual(request: ManualPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)
        series = df['amount']

        # Validasi arimaModel
        if len(request.arima_model) != 3:
            raise HTTPException(status_code=400, detail="Parameter arimaModel harus terdiri dari 3 elemen (p, d, q).")

        p, d, q = request.arima_model

        result = manual_arima_forecast(series, p=p, d=d, q=q, forecast_periods=request.future_steps)

        return ManualPredictionResponse(
            arima_order=tuple(result["arima_order"]),
            prediction=result["prediction"],
            lower=result["lower"],
            upper=result["upper"],
            success=True,            
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses data: {e}")
