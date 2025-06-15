from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

def test_stationarity(timeseries):
    try:
        dftest = adfuller(timeseries.dropna(), autolag='AIC')
        return dftest[1]  # p-value
    except Exception as e:
        print(f"Uji stasioneritas gagal: {e}")
        return 1.0

def find_d_parameter(timeseries):
    d = 0
    ts_stationary = timeseries.copy()
    p_value = test_stationarity(ts_stationary.dropna())

    while p_value > 0.05 and d < 3:
        d += 1
        ts_stationary = ts_stationary.diff().dropna()
        if ts_stationary.empty or len(ts_stationary) < 20:
            return timeseries.diff(d-1).dropna() if d > 0 else timeseries, d-1 if d > 0 else 0
        p_value = test_stationarity(ts_stationary)
    return ts_stationary, d

def manual_arima_forecast(series: pd.Series,
                   p=None, d=None, q=None, forecast_periods=1):
    if series is None or series.empty:
        raise ValueError("Data tidak valid atau kosong.")

    if d is None:
        _, d_optimal = find_d_parameter(series)
    else:
        d_optimal = d

    p_optimal = p if p is not None else 1
    q_optimal = q if q is not None else 1

    model = ARIMA(series.astype(float), order=(p_optimal, d_optimal, q_optimal))
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=forecast_periods)
    forecast_values = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int()

    # if isinstance(series.index, pd.DatetimeIndex):
    #     last_date = series.index[-1]
    #     freq = pd.infer_freq(series.index) or 'D'
    #     forecast_index = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]
    # else:
    #     last_index_val = series.index[-1] if not series.index.empty else -1
    #     forecast_index = pd.RangeIndex(start=last_index_val + 1, stop=last_index_val + 1 + forecast_periods)

    # forecast_series = pd.Series(forecast_values.values, index=forecast_index)
    # confidence_intervals.index = forecast_index
    return {
        "arima_order": (p_optimal, d_optimal, q_optimal),
        "prediction": [float(x) for x in forecast_values.values],
        "lower": list(confidence_intervals.iloc[:, 0]),
        "upper": list(confidence_intervals.iloc[:, 1]),
        "success": True
    }
