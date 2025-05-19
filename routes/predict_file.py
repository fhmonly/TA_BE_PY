from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Literal
import pandas as pd
import io

from services.forecastService import forecast_arima_per_product

router = APIRouter()

@router.post("/predict-file")
async def predict(
    sheet: UploadFile = File(...),
    # recordPeriod: Literal["daily", "weekly", "monthly"] = Form(...),
    predictionPeriod: Literal["weekly", "monthly"] = Form(...),
    predictionMode: Literal["auto", "optimal", "custom"] = Form(...),
    arimaModel: str = Form("")
):
    try:
        # Parse model
        model_values: List[int] = []
        if predictionMode == "custom":
            if not arimaModel:
                raise HTTPException(status_code=400, detail="arimaModel harus diisi saat predictionMode adalah 'custom'")
            try:
                model_values = list(map(int, arimaModel.split(",")))
                if len(model_values) != 3:
                    raise ValueError
            except ValueError:
                raise HTTPException(status_code=400, detail="Format arimaModel harus 'p,d,q'.")

        # Baca file
        content = await sheet.read()
        df = pd.read_csv(io.BytesIO(content)) if sheet.filename.endswith(".csv") else pd.read_excel(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="File tidak berisi data.")

        # Validasi kolom
        if 'product_code' not in df.columns and 'product_name' not in df.columns:
            raise HTTPException(status_code=400, detail="Data harus memiliki kolom 'product_code' atau 'product_name'.")
        if 'date' not in df.columns or 'sold(qty)' not in df.columns:
            raise HTTPException(status_code=400, detail="Data harus memiliki kolom 'date' dan 'sold(qty)'.")

        product_column = 'product_name' if 'product_name' in df.columns else 'product_code'
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=[product_column, 'date'])

        freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
        horizon = 3

        results = []
        for product, group in df.groupby(product_column):
            try:
                result = forecast_arima_per_product(group, freq_map[predictionPeriod], predictionMode, model_values, horizon)
                forecast = result["forecast"]
                results.append({
                    "predictionPeriod":predictionPeriod,
                    "product": product,
                    "order": ",".join(map(str, result["model_params"])),
                    "phase1": forecast[0] if len(forecast) > 0 else None,
                    "phase2": forecast[1] if len(forecast) > 1 else None,
                    "phase3": forecast[2] if len(forecast) > 2 else None,
                })
            except Exception as model_err:
                results.append({
                    "product": product,
                    "error": str(model_err)
                })

        return {"status": "success", "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses file: {str(e)}")
