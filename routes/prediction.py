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

@router.post("/predict/auto", response_model=AutoPredictionResponse)
def predict_auto(request: AutoPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)

        if request.date_column not in df.columns or request.value_column not in df.columns:
            raise HTTPException(status_code=400, detail="Kolom tanggal atau nilai tidak ditemukan di data.")

        freq = "W" if request.prediction_period == "weekly" else "M"

        # Gunakan parameter date_column & value_column dari request
        ts_df = df_group_by_interval(
            df,
            date_col=request.date_column,
            value_col=request.value_column,
            freq=freq if request.date_regroup else None  # hanya grup jika date_regroup True
        )

        series = ts_df[request.value_column]

        result = auto_arima_forecast(series, forecast_periods=3)

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

@router.post("/predict/manual", response_model=ManualPredictionResponse)
def predict_manual(request: ManualPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)

        if any(val is None for val in [p, d, q]):
            raise HTTPException(status_code=400, detail="Semua elemen arimaModel harus memiliki nilai.")

        if request.date_column not in df.columns or request.value_column not in df.columns:
            raise HTTPException(status_code=400, detail="Kolom tanggal atau nilai tidak ditemukan di data.")

        freq = "W" if request.prediction_period == "weekly" else "M"

        # Gunakan freq hanya kalau date_regroup True
        ts_df = df_group_by_interval(
            df,
            date_col=request.date_column,
            value_col=request.value_column,
            freq=freq if request.date_regroup else None
        )

        series = ts_df[request.value_column]

        # Validasi arimaModel
        if len(request.arima_model) != 3:
            raise HTTPException(status_code=400, detail="Parameter arimaModel harus terdiri dari 3 elemen (p, d, q).")

        p, d, q = request.arima_model

        result = manual_arima_forecast(series, p=p, d=d, q=q, forecast_periods=3)

        return ManualPredictionResponse(
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
