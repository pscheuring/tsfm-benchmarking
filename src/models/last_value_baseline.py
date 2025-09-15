import numpy as np
import pandas as pd

from .base_model import BaseModel
from src.utils import logger


class LastValueBaseline(BaseModel):
    """Predicts the last observed value repeatedly."""

    def __init__(self, **params):
        self.forecast_horizon = params["forecast_horizon"]
        self.context_length = params["context_length"]

    def _init_model(self) -> None:
        """No model to initialize for the last value baseline."""
        pass

    def prepare_zero_shot(self, series_df: pd.DataFrame) -> None:
        """Stores the last observed value from the series."""
        self.last_value = series_df["value"].iloc[-(self.forecast_horizon + 1)]

    def predict(self) -> np.ndarray:
        """Predicts the last value for the given horizon."""
        logger.debug(f"Predicting {self.forecast_horizon} steps")
        return np.full(self.forecast_horizon, self.last_value)
