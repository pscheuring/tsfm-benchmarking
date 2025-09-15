import logging
from datetime import datetime
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Subdirectories
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Timestamped log directory
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = BASE_DIR / "logs" / RUN_TIMESTAMP

# File paths
LOG_PATH = LOG_DIR / "main.log"
EXPERIMENT_PATH = CONFIG_DIR / "experiment.yaml"
MODEL_MAP_PATH = CONFIG_DIR / "model_config_map.yaml"
DATA_MAP_PATH = CONFIG_DIR / "dataset_config_map.yaml"
SUMMARY_PATH = RESULTS_DIR / "benchmark_summary.csv"


### Benchmark summary ###
SUMMARY_COLUMNS = [
    "dataset_file",
    "model_name",
    "fit_strategy",
    "few_shot_rate",
    "context_length",
    "forecast_horizon",
    "normalization",
    "missing_value_handling",
    "random_seed",
    "timestamp",
    "result_folder",
]

### Mappings ###
RESULTS_PATH_MAPPINGS = {
    "context_length": "ctx",
    "forecast_horizon": "fh",
    "fit_strategy": {
        "zero-shot": "zs",
        "few-shot": "fs",
        "full-shot": "full",
    },
    "normalization": {
        None: "none",
        "z-score": "z",
        "min-max": "mm",
    },
    "missing_value_handling": {
        None: "none",
        "forward-fill": "ff",
        "back-fill": "bf",
        "mean": "mean",
    },
}

### Logging ###
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROJECT_LOGGER_NAME = "ts_foundation_models"
LOGGING_LEVEL = logging.INFO
