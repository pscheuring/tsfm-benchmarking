"""
Utility helpers for generating benchmark jobs, running experiments, saving results,
maintaining a summary index, and loading/aggregating results for analysis.

Data flow (high level):
1) generate_benchmark_jobs  -> creates concrete job dicts from YAML configs
2) create_dataset_class     -> instantiates the dataset loader for a job
3) run_benchmark_job        -> runs a single job (zero-shot, last-window)
4) save_result              -> writes y_true/y_pred + config.json to a structured folder
5) append_summary           -> appends one line of job metadata to benchmark_summary.csv
6) load_benchmarking_results-> reads the summary, filters runs, computes metrics
7) job_already_done         -> checks de-duplication against the benchmark_summary.csv

Naming & reflection helpers:
- create_full_job_name, generate_result_path, get_dataset_files, load_yaml, get_class, camel_to_snake
"""

import glob
import importlib
import itertools
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.constants import (
    DATA_DIR,
    DATA_MAP_PATH,
    RESULTS_DIR,
    RESULTS_PATH_MAPPINGS,
    SUMMARY_COLUMNS,
    SUMMARY_PATH,
)
from src.evaluation import metrics_collection
from src.logging import logger


def generate_benchmark_jobs(
    experiment_path: str, model_map_path: str, data_map_path: str
) -> List[Dict]:
    """
    Generate a list of all possible benchmark jobs based on experiment configurations.

    Combines the cross product of datasets, (context_length, forecast_horizon),
    seeds, models, fit strategies, normalization, and missing-value handling.

    Args:
        experiment_path (str): Path to the YAML file containing the experiment configuration.
        model_map_path (str): Path to the YAML file containing model parameters and mappings.
        data_map_path (str): Path to the YAML file containing dataset mappings.

    Returns:
        List[Dict]: A list of unique job dictionaries, each describing a full setup.
    """
    config = load_yaml(experiment_path)
    model_config_map = load_yaml(model_map_path)
    model_map = {
        model_config["name"]: model_config
        for model_config in model_config_map["models"]
    }
    data_config_map = load_yaml(data_map_path)
    dataset_map = {
        dataset_config["name"]: dataset_config
        for dataset_config in data_config_map["datasets"]
    }
    all_jobs = []

    experiment = config["experiment"]

    def fix_none_strings(x: str) -> str:
        return None if x == "None" else x

    datasets = experiment["datasets"]
    context_and_horizon = experiment["context_and_horizon"]
    seeds = experiment["random_seed"]
    models = experiment["models"]
    fit_strategies = experiment["fit_strategy"]
    normalizations = [
        fix_none_strings(n) for n in experiment["data_preprocessing"]["normalization"]
    ]
    missing_value_handlings = [
        fix_none_strings(m)
        for m in experiment["data_preprocessing"]["missing_value_handling"]
    ]

    for dataset in datasets:
        try:
            dataset_config = dataset_map[dataset]
        except KeyError:
            logger.warning(
                f"Dataset {dataset} not found in data config map. Skipping..."
            )
            continue
        csv_files = get_dataset_files(dataset_config)
        dataset_class = dataset_config["class"]

        for csv_file in csv_files:
            for combination in itertools.product(
                context_and_horizon,
                seeds,
                models,
                fit_strategies,
                normalizations,
                missing_value_handlings,
            ):
                (
                    (ctx_len, fh),
                    seed,
                    model,
                    fit_strategy,
                    normalization,
                    missing_value_handling,
                ) = combination

                try:
                    model_config = model_map[model]
                except KeyError:
                    logger.warning(
                        f"Model {model_config} not found in model config map. Skipping..."
                    )
                    continue

                model_name = model_config["name"]
                model_class = model_config["class"]
                model_params = model_config.get("params", {})

                job = {
                    "data_path": dataset_config["data_path"],
                    "dataset_class": dataset_class,
                    "dataset_file": csv_file,
                    "context_length": ctx_len,
                    "forecast_horizon": fh,
                    "random_seed": seed,
                    "model_name": model_name,
                    "model_class": model_class,
                    "model_params": model_params,
                    "fit_strategy": fit_strategy["fit_strategy_name"],
                    "normalization": normalization,
                    "missing_value_handling": missing_value_handling,
                }

                if "few_shot_rate" in fit_strategy:
                    job["few_shot_rate"] = fit_strategy["few_shot_rate"]
                if "train_datasets" in fit_strategy:
                    job["train_datasets"] = fit_strategy["train_datasets"]

                all_jobs.append(job)

    # Deduplicate jobs by serializing each job dict into a JSON string.
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        job_key = json.dumps(job, sort_keys=True)
        if job_key not in seen:
            seen.add(job_key)
            unique_jobs.append(job)

    # Count and log how many duplicates were removed
    num_removed = len(all_jobs) - len(unique_jobs)
    if num_removed > 0:
        logger.debug(
            f"Removed {num_removed} duplicate job(s) (kept {len(unique_jobs)} unique)."
        )

    # Final logging: how many unique jobs were generated from config
    logger.info(f"Generated {len(unique_jobs)} benchmark jobs from config.")
    return unique_jobs


def create_dataset_class(job: Dict):
    """
    Dynamically load and initialize the dataset class specified by the job.

    Args:
        job (Dict): Job dictionary containing at least:
            - dataset_class (str)
            - data_path (str)
            - context_length (int)
            - forecast_horizon (int)
            - normalization (str|None)
            - missing_value_handling (str|None)

    Returns:
        Any: An instance of the dataset loader (e.g., UnivariateDataset), already configured.
    """
    dataset_class = get_class(
        f"src.datasets.{camel_to_snake(job['dataset_class'])}",
        job["dataset_class"],
    )
    dataset_loader = dataset_class(
        data_dir=Path(DATA_DIR, job["data_path"]),
        context_length=job["context_length"],
        forecast_horizon=job["forecast_horizon"],
        sort_dates=True,
        normalization=job["normalization"],
        missing_value_handling=job["missing_value_handling"],
    )
    return dataset_loader


def run_benchmark_job(
    series_df: pd.DataFrame,
    job: Dict,
) -> Dict:
    """
    Run a single model on a dataset using a simple zero-shot/last-window setup.

    - Dynamically loads and instantiates the model defined in job.
    - Supports only fit_strategy == "zero-shot".
    - Uses the last window: model consumes the final context window and predicts
      the next forecast_horizon points; y_true is sliced from the end.

    Args:
        series_df (pd.DataFrame): Time series DataFrame with a value column.
        job (dict): Benchmark job configuration.

    Returns:
        Dict: {"y_true": List[float], "y_pred": List[float]}
    """
    # Dynamically load model class
    model_class = get_class(
        f"src.models.{camel_to_snake(job['model_class'])}",
        job["model_class"],
    )
    model = model_class(
        context_length=job["context_length"],
        forecast_horizon=job["forecast_horizon"],
        random_seed=job["random_seed"],
        fit_strategy=job["fit_strategy"],
        **job["model_params"],
    )

    if job["fit_strategy"] != "zero-shot":
        raise NotImplementedError(
            f"Only 'zero-shot' fit_strategy is implemented, got '{job['fit_strategy']}'"
        )
    # Zero-shot forecasting
    model.prepare_zero_shot(series_df)
    preds = model.predict()

    # Last-window extraction
    target = series_df["value"].iloc[-job["forecast_horizon"] :].values
    results = {"y_true": target[: len(preds)].tolist(), "y_pred": preds.tolist()}

    return results


def save_result(
    job: Dict,
    result: Dict,
    base_dir: str,
):
    """
    Save one prediction result to a structured result folder.

    The folder path is generated via generate_result_path and contains:
    - y_true.npy, y_pred.npy (arrays)
    - config.json (the full job dictionary)

    Args:
        job (Dict): Full job dict (all parameters).
        result (Dict): Dict with keys y_true, y_pred.
        base_dir (str): Root of the results directory.

    Returns:
        str: Absolute path of the created result folder.
    """
    model_name = job["model_name"]
    filename = job["dataset_file"]
    result_dir = generate_result_path(base_dir, model_name, job, filename)
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "y_true.npy"), np.array(result["y_true"]))
    np.save(os.path.join(result_dir, "y_pred.npy"), np.array(result["y_pred"]))

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(job, f, indent=2)

    return result_dir


def append_summary(job: Dict, result_dir: str) -> None:
    """
    Append a single row to the global benchmark_summary.csv.

    Notes:
        - The function writes without header (append mode).
        - result_folder is stored relative to the results/ root.
        - A completion timestamp is added.

    Args:
        job (Dict): Full job dict (all parameters).
        result_dir (str): Absolute path to the result folder (will be converted to relative).
    """
    row = {col: job.get(col, None) for col in SUMMARY_COLUMNS}
    relative_result_dir = result_dir.split("results/")[1]

    row["result_folder"] = relative_result_dir
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_row = pd.DataFrame([row])
    df_row.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)


def load_benchmarking_results(
    filter_dict: Dict = None,
    eval_metric_names: List = None,
    summary_csv_path: str = SUMMARY_PATH,
):
    """
    Load benchmark results listed in the summary CSV and compute evaluation metrics.

    Filtering:
        - Direct CSV columns can be filtered by exact match or list membership.
        - Special key 'dataset_collection' will map collection names from the
        dataset_config_map.yaml to concrete dataset_file names.

    Metrics:
        - For each row, loads y_true.npy and y_pred.npy from the result folder,
          computes the metrics from metrics_collection.py, and returns a compact DataFrame.

    Args:
        filter_dict (Dict, optional): Column filters and/or 'dataset_collection' mapping.
        eval_metric_names (List, optional): Names of metric functions in metrics_collection.
        summary_csv_path (str): Path to the summary CSV.

    Returns:
        pd.DataFrame: one row per benchmark run (subset), including computed metrics.
    """
    summary_df = pd.read_csv(summary_csv_path)

    # Apply filtering if specified
    if filter_dict:
        if "dataset_collection" in filter_dict:
            data_config_map = load_yaml(DATA_MAP_PATH)
            dataset_map = {}

            # Create Mapping dataset_collection → filename
            for dataset in data_config_map["datasets"]:
                name = dataset["name"]
                path = Path(dataset["data_path"])

                if "files" in dataset:
                    dataset_map[name] = dataset["files"]
                if "file_pattern" in dataset:
                    pattern = str(DATA_DIR / path / dataset["file_pattern"])
                    dataset_map[name] = [
                        Path(f).name for f in glob.glob(pattern, recursive=True)
                    ]

            dataset_filter = filter_dict["dataset_collection"]
            if not isinstance(dataset_filter, list):
                dataset_filter = [dataset_filter]

            all_allowed_files = []
            for dataset_name in dataset_filter:
                allowed_files = dataset_map.get(dataset_name)
                if allowed_files is None:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in dataset_config_map.yaml."
                    )
                all_allowed_files.extend(allowed_files)

            summary_df = summary_df[summary_df["dataset_file"].isin(all_allowed_files)]

        for key, value in filter_dict.items():
            if key == "dataset_collection":
                continue
            if isinstance(value, list):
                summary_df = summary_df[summary_df[key].isin(value)]
            else:
                summary_df = summary_df[summary_df[key] == value]

    # Compute evaluation metrics
    results = []

    for _, row in summary_df.iterrows():
        dataset = row["dataset_file"]
        model = row["model_name"]
        config = row["result_folder"].split("/")[-2]
        results_folder = Path(RESULTS_DIR, row["result_folder"])

        y_true_path = Path(results_folder, "y_true.npy")
        y_pred_path = Path(results_folder, "y_pred.npy")

        if not y_true_path.exists() or not y_pred_path.exists():
            logger.warning(f"Missing result files in {results_folder}")
            continue

        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)

        eval_metrics = {}
        for metric_name in eval_metric_names or []:
            if hasattr(metrics_collection, metric_name):
                metric_func = getattr(metrics_collection, metric_name)
                if callable(metric_func):
                    eval_metrics[metric_name] = metric_func(y_true, y_pred)
                else:
                    logger.warning(f"{metric_name} in metrics_module is not callable.")
            else:
                logger.warning(f"{metric_name} not found in metrics_module.")

        results.append(
            {
                "dataset_file": dataset,
                "model_name": model,
                "config": config,
                **eval_metrics,
            }
        )

    return pd.DataFrame(results)


def job_already_done(job: Dict[str, Any]) -> bool:
    """
    Check if a given benchmark job has already been completed by looking into
    the summary CSV (SUMMARY_PATH).

    A job is considered "done" if there exists any row where all compared columns
    (SUMMARY_COLUMNS except 'timestamp' and 'result_folder') match exactly. None
    in the job and NaN in the CSV are considered equal.

    Args:
        job (Dict[str, Any]): Dictionary describing the benchmark job configuration.

    Returns:
        bool: True if a matching row exists in the summary file, False otherwise.
    """
    df = pd.read_csv(SUMMARY_PATH)

    # Only check columns that are both in SUMMARY_COLUMNS and in the dataframe
    filter_columns = [
        col
        for col in SUMMARY_COLUMNS
        if col in df.columns and col not in ["timestamp", "result_folder"]
    ]

    # Build a boolean mask: initially all True
    match_mask = pd.Series([True] * len(df))

    for col in filter_columns:
        val = job.get(col, None)

        if val is None:
            match_mask &= df[col].isna()
        else:
            match_mask &= df[col] == val

    return match_mask.any()


def create_full_job_name(job: Dict) -> str:
    """
    Create a human-readable name for a benchmark job (for printing/logging/UIs).

    Args:
        job (Dict): Full job dictionary.

    Returns:
        str: A compact, human-friendly representation of the job.
    """
    full_job_name = (
        f"{job['dataset_class']}/{job['dataset_file']} | "
        f"Context: {job['context_length']} | "
        f"Horizon: {job['forecast_horizon']} | "
        f"Seed: {job['random_seed']} | "
        f"Model: {job['model_name']} ({job['model_class']}) | "
        f"Fit: {job['fit_strategy']} | "
        f"Norm: {job['normalization']} | "
        f"Missing: {job['missing_value_handling']}"
    )

    # Add few-shot info (optional)
    if "few_shot_rate" in job:
        full_job_name += f" | FewShotRate: {job['few_shot_rate']}"

    if "train_datasets" in job:
        full_job_name += f" | TrainSets: {','.join(job['train_datasets'])}"

    return full_job_name


def generate_result_path(
    base_dir: str, model_name: str, job: Dict, filename: str
) -> str:
    """
    Generate a unique, short, readable relative path for a result folder
    based on the experiment configuration.

    Path structure combines short tokens for key settings plus a timestamp +
    series-specific folder name.

    Args:
        base_dir (str): Base directory (e.g., RESULTS_DIR).
        model_name (str): Name of the model (e.g., 'ttm_sktime_r2_simple').
        job (Dict): Full job dictionary.
        filename (str): Original CSV filename (used in the last segment).

    Returns:
        str: The relative/absolute path assembled from the parts.
    """
    parts = []

    # Map context length (prefix ctx + value)
    context_length_prefix = RESULTS_PATH_MAPPINGS["context_length"]
    parts.append(f"{context_length_prefix}{job['context_length']}")

    # Map forecast horizon (prefix fh + value)
    forecast_horizon_prefix = RESULTS_PATH_MAPPINGS["forecast_horizon"]
    parts.append(f"{forecast_horizon_prefix}{job['forecast_horizon']}")

    # Random seed
    parts.append(f"seed{job['random_seed']}")

    # Map fit_strategy
    fit_strategy_short = RESULTS_PATH_MAPPINGS["fit_strategy"].get(
        job["fit_strategy"], job["fit_strategy"]
    )
    parts.append(f"fit-{fit_strategy_short}")

    # Few-shot rate if available
    if "few_shot_rate" in job:
        parts.append(f"fsr-{str(job['few_shot_rate']).replace('.', '')}")

    # Train datasets if available
    if "train_datasets" in job:
        parts.append("train-" + "_".join(job["train_datasets"]))

    # Map normalization
    normalization_short = RESULTS_PATH_MAPPINGS["normalization"].get(
        job.get("normalization", None), "none"
    )
    parts.append(f"norm-{normalization_short}")

    # Map missing_value_handling
    missing_value_handling_short = RESULTS_PATH_MAPPINGS["missing_value_handling"].get(
        job.get("missing_value_handling", None), "none"
    )
    parts.append(f"imp-{missing_value_handling_short}")

    # Build setting name
    setting_name = "_".join(parts)

    # Timestamp + series folder
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    series_folder = f"{timestamp}_{filename.replace('.csv', '')}"

    return os.path.join(base_dir, model_name, setting_name, series_folder)


def get_dataset_files(group: dict) -> List[str]:
    """
    Resolve a dataset group's files list from YAML definition.

    Supports:
        - 'file_pattern' (e.g., "*.csv")
        - 'files' (explicit list)
        - both combined (union)

    Args:
        group (dict): Dataset group from YAML (must include 'data_path').

    Returns:
        List[str]: Sorted list of matched file names (not full paths).
    """
    path = Path(DATA_DIR, group["data_path"])
    file_names = set()

    # Load files using pattern (e.g., "*.csv")
    if "file_pattern" in group:
        pattern = group["file_pattern"]
        matched_files = path.glob(pattern)
        file_names.update(f.name for f in matched_files)

    # Load explicitly listed files
    if "files" in group:
        file_names.update(group["files"])

    return sorted(file_names)


def load_yaml(path: str) -> dict:
    """
    Load a YAML file into a Python dictionary.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_class(path: str, name: str):
    """
    Dynamically import a class by name from a module.

    Args:
        path (str): Dotted module path (e.g., 'src.models.ttm_sktime').
        name (str): Class name inside that module (e.g., 'TtmSktime').

    Returns:
        type: The imported class object.
    """
    return getattr(importlib.import_module(path), name)


def camel_to_snake(name: str) -> str:
    """
    Convert a CamelCase class name to snake_case module name.

    Args:
        name (str): The CamelCase name.

    Returns:
        str: snake_case version of the name.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
