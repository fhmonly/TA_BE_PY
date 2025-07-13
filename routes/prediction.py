from fastapi import APIRouter, HTTPException
from schema.prediction import (
    AutoPredictionRequest,
    AutoPredictionResponse,
    ManualPredictionRequest,
    ManualPredictionResponse,
)
from utils.statistic.homemade_arima import MyARIMA
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

        model = MyARIMA(
            dataset=series.tolist()
        )
        model.fit()
        result = model.forecast(1)

        return AutoPredictionResponse(
            rmse=model.best_model['rmse'],
            mape=model.best_model['mape'],
            arima_order=model.best_model["order"],
            prediction=result["predictions"],
            lower=result["lower_bound"],
            upper=result["upper_bound"],
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

        model = MyARIMA(
            dataset=series.tolist(),
            model={'p': p, 'd': d, 'q': q}
        )
        result = model.forecast(1)

        return ManualPredictionResponse(
            arima_order=model.best_model['order'],
            prediction=result["predictions"],
            lower=result["lower_bound"],
            upper=result["upper_bound"],
            success=True
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses data: {e}")
