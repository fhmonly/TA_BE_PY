from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def forecast_arima_per_product(group: pd.DataFrame, freq: str, mode: str, arima_order: list[int], horizon: int):
    group = group.set_index('date')
    df_resampled = group.resample(freq).sum().dropna()
    series = df_resampled['sold(qty)']

    if adfuller(series)[1] > 0.05:
        series = series.diff().dropna()

    try:
        if mode == "auto":
            model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                d=None,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            forecast = model.predict(n_periods=horizon)
            return {
                "forecast": forecast.tolist(),
                "model_params": model.order
            }

        elif mode == "optimal":
            model_order = (2, 1, 2) 
            model = ARIMA(series, order=model_order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            return {
                "forecast": forecast.tolist(),
                "model_params": model_order
            }
            
        elif mode == "custom":
            if len(arima_order) != 3:
                raise ValueError("Parameter ARIMA harus 3 angka: p,d,q.")
            model = ARIMA(series, order=tuple(arima_order))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            return {
                "forecast": forecast.tolist(),
                "model_params": arima_order
            }

        else:
            raise ValueError("Mode prediksi tidak valid.")

    except Exception as e:
        raise RuntimeError(f"Model ARIMA gagal dibentuk: {str(e)}")
