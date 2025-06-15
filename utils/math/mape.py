import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Hitung MAPE manual karena sklearn belum built-in."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100