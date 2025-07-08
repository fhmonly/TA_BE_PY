from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple


class BasePredictionRequest(BaseModel):
    csv_string: str
    prediction_period: Literal["weekly", "monthly"]
    value_column: str='sold_qty'
    date_column: str='date'
    date_regroup: bool=False
    future_steps: int=1


class AutoPredictionRequest(BasePredictionRequest):
    """Request model for automatic ARIMA prediction."""
    pass


class ManualPredictionRequest(BasePredictionRequest):
    """Request model for manual ARIMA prediction."""
    arima_model: List[Optional[int]] = []


class BasePredictionResponse(BaseModel):
    arima_order: Tuple[int, int, int]
    upper: List[float]
    lower: List[float]
    prediction: List[float]
    success: bool


class AutoPredictionResponse(BasePredictionResponse):
    """Response model for automatic ARIMA prediction with error metrics."""
    rmse: float
    mape: float


class ManualPredictionResponse(BasePredictionResponse):
    """Response model for manual ARIMA prediction."""
    pass
