from sklearn.metrics import mean_squared_error as mse
from utils.math.mape import mean_absolute_percentage_error
import pmdarima as pm
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
def auto_arima_forecast(series: pd.Series, train_ratio=0.7, forecast_periods: int = 1) -> dict:
    if series is None or series.empty:
        raise ValueError("Data tidak valid atau kosong.")

    # Split data
    train_len = int(len(series) * train_ratio)
    train_data = series.iloc[:train_len]
    test_data = series.iloc[train_len:]

    # Training model
    model = pm.auto_arima(
        train_data,
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        d=None,
        test='adf',
        seasonal=False,
        m=1,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Prediksi test set
    n_test = len(test_data)
    preds_test, _ = model.predict(n_periods=n_test, return_conf_int=True)
    rmse = np.sqrt(mse(test_data, preds_test))
    mape = mean_absolute_percentage_error(test_data, preds_test)

    # Refit dengan seluruh data
    model.update(test_data)

    # Forecast n-step forward
    forecast, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)

    # Buat index forecast
    # if isinstance(series.index, pd.DatetimeIndex):
        # last_date = series.index[-1]
        # freq = pd.infer_freq(series.index) or 'D'
        # forecast_index = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]
    # else:
        # last_index = series.index[-1] if not series.index.empty else -1
        # forecast_index = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + forecast_periods)

    forecast_values = list(forecast)
    lower_bounds = list(conf_int[:, 0])
    upper_bounds = list(conf_int[:, 1])

    return {
        "rmse": rmse,
        "mape": mape,
        "arima_order": model.order,
        "prediction": forecast_values,
        "lower": lower_bounds,
        "upper": upper_bounds,
        "success": True
    }