import os

import pandas as pd
from tqdm import tqdm

from src.constants import (
    DATA_MAP_PATH,
    EXPERIMENT_PATH,
    MODEL_MAP_PATH,
    SUMMARY_COLUMNS,
    SUMMARY_PATH,
    RESULTS_DIR,
)
from src.utils import (
    append_summary,
    create_dataset_class,
    create_full_job_name,
    generate_benchmark_jobs,
    job_already_done,
    run_benchmark_job,
    save_result,
)
from src.logging import logger


def main():
    logger.info("Benchmarking started.")
    benchmark_jobs = generate_benchmark_jobs(
        EXPERIMENT_PATH, MODEL_MAP_PATH, DATA_MAP_PATH
    )

    # Initialize or load summary file
    if os.path.exists(SUMMARY_PATH):
        summary_df = pd.read_csv(SUMMARY_PATH)
    else:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        summary_df.to_csv(SUMMARY_PATH, index=False)

    # Run all benchmark_jobs
    with tqdm(
        total=len(benchmark_jobs), desc="Running benchmark_jobs", ncols=100
    ) as pbar:
        for job in benchmark_jobs:
            short_job_name = f"{job['dataset_file']} | {job['model_name']}"
            full_job_name = create_full_job_name(job)
            tqdm.write(f"Now running: {short_job_name}")

            if job_already_done(job):
                logger.debug(f"Skipping job (already exists): {full_job_name}")
                pbar.update(1)
                continue

            logger.debug(f"Started job: {full_job_name}")

            # Load dataset
            dataset_loader = create_dataset_class(job)
            series_df = dataset_loader.load_preprocessed_data(
                filename=job["dataset_file"]
            )
            if series_df is None:
                logger.debug(f"Skipping job (no data): {full_job_name}")
                pbar.update(1)
                continue

            # Run experiment
            results = run_benchmark_job(
                series_df=series_df,
                job=job,
            )

            # Save results
            logger.debug("Saving results")
            result_dir = save_result(job, results, base_dir=RESULTS_DIR)

            # Append job to summary
            logger.debug("Appending summary")
            append_summary(job, result_dir)

            logger.debug(f"Finished job: {short_job_name}")

            pbar.update(1)

    logger.info(f"Finished benchmarking {len(benchmark_jobs)} benchmark_jobs.")


if __name__ == "__main__":
    main()
