"""
This module runs the pipeline according to parameters specifed in configs/
"""

from src.pipeline import (
    download_raw_data,
    preprocess_raw_data,
    apply_feature_engineering,
    load_data,
    train_model,
    evaluate_model,
)


def run_pipeline():
    """
    Run all the steps defined in pipeline.py
    """
    raw_dfs = download_raw_data()
    preprocessed_dfs = preprocess_raw_data(raw_dfs)
    engineered_dfs = apply_feature_engineering(preprocessed_dfs)
    data_loaders_list = load_data(engineered_dfs)
    trained_models_and_metrics_list = train_model(data_loaders_list)
    evaluate_model(trained_models_and_metrics_list)
    # evaluation_metrics_list =


if __name__ == "__main__":
    run_pipeline()
