import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ttm import TinyTimeMixerForecaster

from .base_model import BaseModel
from src.logging import logger


class TtmSktime(BaseModel):
    """Wrapper for the TinyTimeMixerForecaster from sktime using IBM's foundation model."""

    def __init__(self, **params):
        """
        Args:
            context_length (int): Number of past time steps to use as input.
            forecast_horizon (int): Number of future steps to predict.
            model_path (str, optional): Path to the Hugging Face TinyTimeMixer model.
            fit_strategy (str, optional): Fit strategy ('zero-shot' by default).
            batch_size (int, optional): Batch size for inference (not used in sktime).
            random_seed (int, optional): Random seed for reproducibility.
        """
        self.context_length = params["context_length"]
        self.forecast_horizon = params["forecast_horizon"]
        self.model_path = params["model_path"]
        self.fit_strategy = params["fit_strategy"]
        self.batch_size = params["batch_size"]
        self.random_seed = params["random_seed"]
        self.revision = "main"

        self._init_model()

    def _init_model(self):
        """Initializes the TinyTimeMixer forecaster from sktime with the given configuration."""
        logger.debug(f"Loading model from {self.model_path}")
        self.forecaster = TinyTimeMixerForecaster(
            model_path=self.model_path,
            fit_strategy=self.fit_strategy,
            config={
                "context_length": self.context_length,
                "prediction_length": self.forecast_horizon,
            },
        )
        # set seed via sktime API
        self.forecaster.set_random_state(self.random_seed)

    def prepare_zero_shot(self, series_df: pd.DataFrame) -> None:
        """
        Prepares the forecaster for zero-shot prediction by fitting on the last context window.

        Args:
            series (pd.DataFrame): Univariate time series (e.g. last context window).
        """
        value_series = series_df["value"]
        context_window = value_series.iloc[
            -(self.context_length + self.forecast_horizon) : -self.forecast_horizon
        ]
        self.fh = ForecastingHorizon(
            values=np.arange(1, self.forecast_horizon + 1), is_relative=True
        )
        self.forecaster.fit(context_window, fh=self.fh)

    def predict(self: int) -> np.ndarray:
        """
        Predicts the next time steps using the pre-initialized forecaster.

        Returns:
            np.ndarray: Array of predicted values of shape (horizon,)
        """
        logger.debug(f"Predicting {self.forecast_horizon} steps")

        y_pred = self.forecaster.predict()
        return y_pred.values

    def reset(self):
        """
        Reinitializes the model (stateless zero-shot reset).
        """
        self._init_model()
