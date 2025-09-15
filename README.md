# 📈 Time Series Foundation Model Benchmarking

This repository provides a flexible benchmarking pipeline for time series (foundation) models.  
It allows to systematically run experiments across different **datasets, models, preprocessing methods, and training strategies**, and stores results in a structured format for later analysis.

## Features
- Benchmarking of time series (foundation) models on forecasting tasks
- Modular design to integrate custom models and datasets.  
- Configurable preprocessing (normalization, missing value handling).  
- Automatic tracking, logging, and structured saving of results (`y_true`, `y_pred`, configs).  
- Load and explore results programmatically for analysis and visualization.
- Currently, experiments are based on the **last window** of each time series, from which context and forecast horizon are derived.  
- At the moment only **zero-shot forecasting** is supported;  
  **few-shot** and **full-shot** capabilities will be added in the future.    


## Repository Structure

```text
.
├── main.py                           # Entry point for benchmarking
├── src/
│   ├── utils.py                      # Utility functions (job generation, runner, etc.)
│   ├── constants.py                  # Paths and global constants
│   ├── logging.py                    # Logging setup
│   ├── models/                       # Time series foundation models
│   ├── datasets/                     # Data handling
│   └── evaluation/                   # Evaluation metrics
├── config/
│   ├── experiment.yaml               # Experiment configuration
│   ├── dataset_config_map.yaml       # Dataset definitions
│   └── model_config_map.yaml         # Model definitions
├── results/                          # Results folder
├── notebooks/                        # Jupyter notebooks
├── data/                             # Raw input data
│   └── univariate_raw/
├── logs/                             # Training / experiment logs
├── pyproject.toml                    # uv setup
├── uv.lock
├── README.md
```

## Configuration

The benchmarking pipeline is fully driven by YAML config files.

### 1. Experiment Config (`experiment.yaml`)

This is the main YAML defining the *grid of benchmarking experiments* that will be run.

Example config:
```yaml
experiment:
  datasets:
    - example_all

  context_and_horizon:
    - [512, 96]
    - [1024, 96]

  random_seed:
    - 2
    - 3

  data_preprocessing:
    normalization:
      - "z-score"
    missing_value_handling:
      - "dropna"

  models:
    - example_model_epoch_100_batch_64
    - example_model_epoch_500_batch_64

  fit_strategy:
    - fit_strategy_name: "zero-shot"
```
Each experiment is expanded into all possible combinations of these parameters.

In this example:
- 1 dataset_collection (assuming **10** datasets in `examples_all`)
- 2 context/horizon settings
- 2 random seed settings
- 1 normalization setting
- 1 missing value handling setting
- 2 model setting
- 1 fit strategy

This results in **10 × 2 × 2 × 1 × 1 × 2 × 1 = 80 benchmark jobs** being executed.

For each experiment, a separate folder inside the `results/` directory will be created.
Each folder contains the predictions (`y_true.npy`, `y_pred.npy`) and the corresponding `config.json`.

To keep the `experiment.yaml` clean, datasets and models are only referenced by name. The actual dataset and model definitions are in `dataset_config_map.yaml` and `model_config_map.yaml`. During job generation, the names in `experiment.yaml` are resolved against these config files. Accordingly each dataset or model config in the `experiment.yaml` must exist in the `dataset_config_map.yaml` or `model_config_map.yaml` respectively.

### 2. Dataset Config (`dataset_config_map.yaml`)
Defines dataset collections that can applied in the `experiment.yaml`. Collections can be defined either by explicit file lists or by file patterns. Once a dataset collection is defined, its name can be referenced in the `experiment.yaml`
```yaml
datasets:
  - name: example_1_10
    class: UnivariateDataset
    data_path: univariate_raw
    files:
      - example_1.csv
      - example_10.csv

  - name: example_all
    class: UnivariateDataset
    data_path: univariate_raw
    file_pattern: "example*.csv"

  - ... 
---
```
### 3. Model Config (`model_config_map.yaml`)
Defines available models and their parameters. Each model may define its own set of parameters and model-specific options. Once a model is defined, its name can be referenced in the `experiment.yaml`
```yaml
models
  - name: example_model_epoch_100_batch_64
    class: "ExampleModel"
    params:
      epochs: 100
      batch_size: 64

  - ...

---
```

## Track experiment runs through `benchmark_summary.csv`

All experiment runs are tracked in a global benchmark_summary file: **`benchmark_summary.csv`**.  
This file is automatically created in the project root on the first run.  

### What does it contain?
- One row per executed benchmark job  
- Metadata from the benchmark job config
- Path to the corresponding result folder (`result_folder`)  
- A timestamp of when the run finished  

### Avoiding duplicate runs
Before starting a new job, the pipeline checks whether the job has **already been completed**

A job is considered *done* if all relevant configuration columns in the job dictionary match **exactly** with an entry in `benchmark_summary.csv`.   

If a match is found, the job is **skipped** to avoid duplicate work.

### Appending new results
When a job finishes, the pipeline:
1. Stores the results in the `results/` folder under a corresponding path.  
2. Appends a new row with the job’s metadata to the `benchmark_summary.csv` file.  

Over time, `benchmark_summary.csv` serves as a **central experiment log**, making it easy to filter, load and reproduce results.  

## Results

Each experiment run automaitcally stores:
- y_true.npy – ground truth values  
- y_pred.npy – predicted values  
- config.json – full job configuration  

in a folder following a structured save path:
```text
results/
└── <model_name>/
    └── <configs_short_name>/
        └── <timestep_datasetname>/
            ├── y_true.npy
            ├── y_pred.npy
            └── config.json

```

## Adding a new model class
To integrate a new time series (foundation) model into the benchmarking framework, follow these steps:

**1. Create a new file in src/models/**
- The filename must be the snake_case version of the class name.
- Example: `ttm_sktime.py` → `class TtmSktime`.

**2. Inherit from BaseModel**
- Every model must implement the following abstract methods:
  - `_init_model(self)` – initialize the underlying forecasting model.
  - `prepare_zero_shot(self, series_df: pd.DataFrame)` – prepare the model for zero-shot forecasting.
  - `predict(self)` – return predictions for the specified horizon.
  - few-shot and full-shot are currently not implemented, but it is planned to integrate them later

**3. Define the model’s constructor `(__init__)`**
- Accept all necessary parameters via **params.
- Store them as attributes and call `_init_model()` inside `__init__`.

The easiest way to implement a new model is to copy the structure of an existing one (e.g. `TtmSktime`) and adapt it to the new library/model.

## Adding a new dataset class
To integrate a new dataset class into the benchmarking framework, follow these steps:

**1. Create a new file in src/models/**
- The filename must be the snake_case version of the class name.
- Example: `univariate_dataset.py` → `class UnivariateDataset`.

**2. Inherit from BaseDataset**
Every dataset class must implement:
- `load_preprocessed_data(self, ..) -> pd.DataFrame`
- This method is responsible for loading raw data, applying missing value handling, normalization, and returning DataFrames ready for benchmarking.
- Column naming: after load_preprocessed_data() the dataframe must provide two columns named timestamp (time information) and value (observations).

## Load and explore benchmarking results
Use the `load_benchmarking_results` function from `src/utils.py` to load benchmarking results.

You can filter by dataset, model, forecast horizon, normalization, etc.
An example workflow is shown in `notebooks/01_explore_benchmarking_results.ipynb`, where results are loaded, filtered, metrics are computed, and models are compared visually.

## Start & Installation  

**Requirements:**  
- Python >=3.11
- [uv](https://docs.astral.sh/uv/) for dependency management  

**Install dependencies:**  
```bash
uv sync
```
**Running benchmarks**

Run the pipeline from the project root:

```bash
uv run main.py
```

The pipeline will:

1. Generate all benchmark jobs as defined in the configs (`experiment.yaml`, `dataset_config_map.yaml`, `model_config_map.yaml`).
2. For each job:
   - Load dataset.
   - Apply preprocessing.
   - Configure and run the model.
   - Save predictions, ground truth and configs.
   - Update `benchmark_summary.csv`

Progress is displayed using tqdm.

## 📖 License

MIT License – feel free to use and extend this framework.
