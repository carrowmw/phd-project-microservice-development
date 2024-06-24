"""
This module runs the pipeline according to parameters specified in configs/
"""

import logging
from phd_project.src.pipeline import (
    read_or_download_raw_data,
    preprocess_raw_data,
    apply_feature_engineering,
    load_data,
    train_model,
    evaluate_model,
)

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    filename="logs/pipeline_output.txt",
    filemode="w",
)


def run_pipeline():
    """
    Run all the steps defined in pipeline.py

    Returns:
        None
    """
    logging.info("Starting pipeline...")

    logging.info("Reading or downloading raw data...")
    raw_dfs = read_or_download_raw_data()

    logging.info("Preprocessing raw data...")
    preprocessed_dfs = preprocess_raw_data(raw_dfs)

    logging.info("Applying feature engineering...")
    engineered_dfs = apply_feature_engineering(preprocessed_dfs)

    logging.info("Loading data...")
    data_loaders_list = load_data(engineered_dfs)

    logging.info("Training models...")
    trained_models_and_metrics_list = train_model(data_loaders_list)

    logging.info("Evaluating models...")
    evaluation_metrics_list = evaluate_model(trained_models_and_metrics_list)

    logging.info("Pipeline completed.")

    return evaluation_metrics_list


if __name__ == "__main__":
    run_pipeline()
