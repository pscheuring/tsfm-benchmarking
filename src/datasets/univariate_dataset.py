import os

import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.logging import logger


class UnivariateDataset(BaseDataset):
    """
    Loads and preprocesses univariate time series from individual CSV files in a directory.

    Each CSV file must contain columns 'date' and 'data'.
    The method converts them into a standard format with 'timestamp' and 'value',
    and applies missing value handling and normalization if configured.
    """

    def load_preprocessed_data(self, filename: str) -> pd.DataFrame:
        """
        Loads, preprocesses, and filters time series data from CSV files.

        Args:
            filename (str): Name of the CSV file to load.

        Returns:
            series_df (pd.Dataframe): dataframe used for benchmarking job
        """
        file_path = os.path.join(self.data_dir, filename)
        raw_df = pd.read_csv(file_path)

        if self.sort_dates:
            raw_df = raw_df.sort_values("date")

        # Keep only relevant columns and rename
        series_df = raw_df[["date", "data"]].copy()
        series_df = series_df.rename(columns={"date": "timestamp", "data": "value"})

        # Convert to datetime
        series_df["timestamp"] = pd.to_datetime(series_df["timestamp"])

        # Apply missing value handling
        series_df = self.impute_data(series_df)

        if len(series_df) < self.forecast_horizon + 10:
            logger.info(
                f"Skipping '{filename}' – only {len(series_df)} rows (needs > {self.forecast_horizon})"
            )
            return None

        # Apply normalization
        series_df = self.normalize_data(series_df)

        return series_df
