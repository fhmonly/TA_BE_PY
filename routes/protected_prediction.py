from fastapi import APIRouter, HTTPException
from schema.prediction import (
    AutoPredictionRequest,
    AutoPredictionResponse,
    ManualPredictionRequest,
    ManualPredictionResponse,
)
from utils.statistic.homemade_arima import MyARIMA
from utils.data_preparation import read_csv_string_to_df

router = APIRouter()

@router.post("/predict/private/auto", response_model=AutoPredictionResponse)
def predict_auto(request: AutoPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)

        series = df['amount']

        model = MyARIMA(
            dataset=series.tolist()
        )
        model.fit()
        result = model.forecast(steps=request.future_steps)

        return AutoPredictionResponse(
            rmse=model.best_model["rmse"],
            mape=model.best_model["mape"],
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

@router.post("/predict/private/manual", response_model=ManualPredictionResponse)
def predict_manual(request: ManualPredictionRequest):
    try:
        df = read_csv_string_to_df(request.csv_string)
        series = df['amount']

        # Validasi arimaModel
        if len(request.arima_model) != 3:
            raise HTTPException(status_code=400, detail="Parameter arimaModel harus terdiri dari 3 elemen (p, d, q).")

        p, d, q = request.arima_model

        model = MyARIMA(
            dataset=series.tolist(),
            model={'p': p, 'd': d, 'q': q}
        )
        result = model.forecast(steps=request.future_steps)

        return ManualPredictionResponse(
            arima_order=model.best_model['order'],
            prediction=result["predictions"],
            lower=result["lower_bound"],
            upper=result["upper_bound"],
            success=True,            
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses data: {e}")
