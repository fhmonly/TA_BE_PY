from typing import List, Tuple, Union, Optional, Dict, TypedDict
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

class ARIMACoefficients(TypedDict):
    constant: float
    ar_coefs: List[float]
    ma_coefs: List[float]
    residuals: List[float]

class BestModel(TypedDict, total=False):
    order: Tuple[int, int, int]
    coefs: ARIMACoefficients
    mape: Optional[float]
    rmse: Optional[float]

class ForecastResult(TypedDict):
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]

class MyARIMA:
    def __init__(self, dataset: List[float], train_ratio: float = 0.7, model: Optional[Dict[str, int]] = None):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.model_dict = model

        self.ar_ma_orders: List[Tuple[int, int]] = [(2, 2), (2, 1), (1, 2), (1, 1)]

        self.data_train: Optional[List[float]] = None
        self.data_test: Optional[List[float]] = None

        self.best_model: Optional[BestModel] = None
        self.mape: Optional[float] = None

        self._split_data()
        self._initialize_model()

    def _split_data(self) -> None:
        split_idx = int(len(self.dataset) * self.train_ratio)
        self.data_train = self.dataset[:split_idx]
        self.data_test = self.dataset[split_idx:]

    def _initialize_model(self) -> None:
        if self.model_dict:
            p = self.model_dict.get('p', 1)
            d = self.model_dict.get('d', 0)
            q = self.model_dict.get('q', 1)
            full_order = (p, d, q)

            coefs = self._get_arima_coefficients(self.dataset, full_order)

            self.best_model = {
                'order': full_order,
                'coefs': coefs,
                'mape': None
            }

    @staticmethod
    def _get_arima_coefficients(
        data: List[Union[int, float]],
        order: Tuple[int, int, int]
    ) -> ARIMACoefficients:
        trend = 'c' if order[1] == 0 else 'n'
        model = ARIMA(data, order=order, trend=trend)
        model_fit = model.fit()

        param_dict = dict(zip(model_fit.param_names, model_fit.params))
        return {
            'constant': param_dict.get('const', 0.0),
            'ar_coefs': model_fit.arparams.tolist(),
            'ma_coefs': model_fit.maparams.tolist(),
            'residuals': model_fit.resid.tolist()
        }

    @staticmethod
    def _difference(data: List[float], d: int = 1) -> List[float]:
        if d == 0:
            return data
        result = list(np.diff(data, n=d))
        return result

    @staticmethod
    def _inverse_difference(last_ob_series: List[float], diff_forecast: List[float], d: int) -> List[float]:
        history = last_ob_series[:]
        for forecast_val in diff_forecast:
            for i in range(d):
                if i == 0:
                    inverted_val = history[-1] + forecast_val
                else:
                    diff_history = np.diff(history[-(d-i):]).tolist()
                    inverted_val = diff_history[-1] + forecast_val
                forecast_val = inverted_val
            history.append(inverted_val)
        return history[len(last_ob_series):]

    @staticmethod
    def _mape(y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("Panjang y_true dan y_pred harus sama.")
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        if not np.any(non_zero_mask):
            return 0.0 if np.all(y_pred == 0) else float('inf')
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    @staticmethod
    def _rmse(y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("Panjang y_true dan y_pred harus sama.")
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

    @staticmethod
    def _check_stationarity_adf(series: List[float]) -> dict:
        result = adfuller(series)
        return {'p_value': result[1], 'is_stationary': result[1] < 0.05}

    def _arima_manual_forecast(
        self,
        data: List[float],
        order: Tuple[int,int,int],
        coefs: Dict[str, Union[float, List[float]]],
        steps: int
    ) -> List[float]:
        p, d, q = order
        ar_coefs = coefs['ar_coefs']
        ma_coefs = coefs['ma_coefs']
        c = coefs['constant']
        residuals = coefs['residuals']

        if d > 0:
            history = self._difference(data, d)
        else:
            history = data[:]

        errors = residuals[-q:] if q > 0 else []
        forecasts_diff = []

        for _ in range(steps):
            ar_part = sum(ar_coefs[i] * history[-i - 1] for i in range(p))
            ma_part = sum(ma_coefs[i] * errors[-i - 1] for i in range(len(errors))) # len(errors) aman jika q=0

            yhat = c + ar_part+ ma_part
            forecasts_diff.append(yhat)

            history.append(yhat)
            errors.append(0)
            if q > 0:
                errors.pop(0)

        if d > 0:
            last_obs_for_inverse = data[-d:]
            predicted = self._inverse_difference(last_obs_for_inverse, forecasts_diff, d)
        else:
            predicted = forecasts_diff

        return predicted

    def fit(self) -> None:
        if self.best_model and self.best_model.get('mape') is not None:
            return
        if self.best_model and self.model_dict:
             return

        d = 0 if self._check_stationarity_adf(self.data_train)['is_stationary'] else 1
        best_mape = float('inf')
        best_model_info = None

        for p, q in self.ar_ma_orders:
            full_order = (p, d, q)
            try:
                coefs = self._get_arima_coefficients(self.data_train, full_order)

                preds = self._arima_manual_forecast(
                    data=self.data_train,
                    order=full_order,
                    coefs=coefs,
                    steps=len(self.data_test)
                )

                current_mape = self._mape(self.data_test, preds)
                current_rmse = self._rmse(self.data_test, preds)

                if current_mape < best_mape:
                    best_mape = current_mape
                    best_model_info = {
                        'order': full_order,
                        'coefs': coefs,
                        'mape': current_mape,
                        'rmse': current_rmse
                    }
                    
            except Exception as e:
                continue

        self.best_model = best_model_info
        self.mape = best_mape

    def forecast(self, steps: int) -> ForecastResult:
        if not self.best_model:
            raise RuntimeError("Model belum di-fit. Panggil .fit() terlebih dahulu.")

        best_order = self.best_model['order']
        final_coefs = self._get_arima_coefficients(self.dataset, best_order)

        preds = self._arima_manual_forecast(
            data=self.dataset,
            order=best_order,
            coefs=final_coefs,
            steps=steps
        ) 
        residuals = np.array(final_coefs['residuals'])
        std_err = np.std(residuals, ddof=1)  

        z = 1.96  
        preds = np.array(preds)
        lower_bound = preds - z * std_err
        upper_bound = preds + z * std_err
        return {
            'predictions': preds.tolist(), 
            'lower_bound': lower_bound.tolist(), 
            'upper_bound': upper_bound.tolist()
        }