from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BaseDataset(ABC):
    """
    Abstract base class for loading and preprocessing time series datasets.
    Handles missing value treatment and normalization consistently across datasets.
    """

    def __init__(
        self,
        data_dir: str,
        missing_value_handling: Optional[str],
        normalization: Optional[str],
        context_length: int,
        forecast_horizon: int,
        sort_dates: bool = True,
    ):
        """
        Initializes the dataset with preprocessing options.

        Args:
            data_dir (str): Path to the dataset directory.
            missing_value_handling (Optional[str]): Strategy for missing values ('dropna', 'mean', 'forward-fill', 'back-fill', or None).
            normalization (Optional[str]): Normalization method ('z-score', 'min-max', or None).
            context_length (int): Number of past time steps used for forecasting.
            forecast_horizon (int): Number of future time steps to predict.
            sort_dates (bool, optional): Whether to sort the dataset by timestamps. Defaults to True.
        """
        self.data_dir = data_dir
        self.missing_value_handling = missing_value_handling
        self.normalization = normalization
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.sort_dates = sort_dates

    @abstractmethod
    def load_preprocessed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads and preprocesses the complete dataset.
        This method must be implemented by subclasses.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping series IDs to preprocessed DataFrames.
        """
        pass

    def impute_data(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the missing value handling strategy to a time series.

        Args:
            series_df (pd.DataFrame): Time series DataFrame with a 'value' column.

        Returns:
            pd.DataFrame: Imputed DataFrame.
        """
        if self.missing_value_handling is None:
            return series_df

        if self.missing_value_handling == "dropna":
            series_df = series_df.dropna(subset=["value"])
        elif self.missing_value_handling == "mean":
            series_df["value"] = series_df["value"].fillna(series_df["value"].mean())
        elif self.missing_value_handling == "forward-fill":
            series_df["value"] = series_df["value"].ffill()
        elif self.missing_value_handling == "back-fill":
            series_df["value"] = series_df["value"].bfill()
        else:
            raise ValueError(
                f"Unsupported missing value handling method: {self.missing_value_handling}"
            )
        return series_df

    def normalize_data(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the time series using a scikit-learn self.normalization, excluding the forecast horizon value.

        Args:
            series_df (pd.DataFrame): DataFrame with columns 'timestamp' and 'value'

        Returns:
            pd.DataFrame: DataFrame with normalized values for both context and forecast horizon
        """

        # If no normalization is required, return the original DataFrame
        if self.normalization is None:
            return series_df

        # Determine which self.normalization to use
        if self.normalization == "z-score":
            scaler = StandardScaler()
        elif self.normalization == "min-max":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization}")

        # Extract context (all data except for the forecast horizon)
        context_data = (
            series_df["value"].iloc[: -self.forecast_horizon].values.reshape(-1, 1)
        )

        # Fit and transform the context data (fit on the context)
        scaler.fit(context_data)
        context_data_normalized = scaler.transform(context_data)

        # Normalize the forecast horizon (last value or set of values)
        forecast_horizon_data = (
            series_df["value"].iloc[-self.forecast_horizon :].values.reshape(-1, 1)
        )
        forecast_horizon_normalized = scaler.transform(forecast_horizon_data)

        # Combine the normalized context data and forecast horizon
        normalized_data = pd.concat(
            [
                pd.Series(context_data_normalized.flatten(), name="value"),
                pd.Series(forecast_horizon_normalized.flatten(), name="value"),
            ],
            ignore_index=True,
        )

        # Create a new DataFrame with the normalized values
        df_normalized = series_df.copy()
        df_normalized["value"] = normalized_data

        return df_normalized

    def train_test_split(
        self, values: pd.Series, test_size: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Splits a time series into training and testing sets.

        Args:
            values (pd.Series): The full time series.
            test_size (int): Number of time steps reserved for testing.

        Returns:
            Tuple (train, test): Train and test sets as slices of the original series.
        """
        return values[:-test_size], values[-test_size:]
