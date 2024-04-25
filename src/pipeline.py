import numpy as np
import torch

from src.utils.general_utils import load_config
from src.utils.pipeline_utils import process_data


def preprocess_raw_data(raw_dfs) -> list:
    """ """
    preprocessing_config_path = "configs/preprocessing_config.json"
    print(f"\n\n {len(raw_dfs)} DataFrames found\n")

    preprocessed_dfs = []
    preprocessing_config = load_config("configs/preprocessing_config.json")
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]
    print(f"Completeness: {completeness_threshold*100}% per day")

    for i, df in enumerate(raw_dfs):
        print(f"\n\nProcessing {i+1}/{len(raw_dfs)}\n")
        processed_df = process_data(df[1], preprocessing_config_path)
        sensor_name = df[0]
        preprocessed_dfs.append((sensor_name, processed_df))

    print("\n\nFinished preprocessing\n")
    return preprocessed_dfs


def apply_feature_engineering(preprocessed_dfs):
    """ """

    empty_df_count = 0

    # Filter out empty DataFrames and count them
    non_empty_preprocessed_dfs = []
    for sensor_name, df in preprocessed_dfs[:]:
        if df.empty:
            empty_df_count += 1
        else:
            non_empty_preprocessed_dfs.append((sensor_name, df))

    feature_engineering_config_path = "configs/feature_engineering_config.json"
    print(f"\n\n Dropped {empty_df_count} empty DataFrames\n")
    print(
        f"\n\n Engineering features for {len(non_empty_preprocessed_dfs)} DataFrames\n"
    )

    engineered_dfs = []
    for i, df in enumerate(non_empty_preprocessed_dfs):
        print(f"\n\nProcessing {i+1}/{len(non_empty_preprocessed_dfs)}\n")
        if i == 0:
            print(f"Pre-engineered: {df}")
        engineered_df = process_data(df[1], feature_engineering_config_path)
        sensor_name = df[0]
        engineered_dfs.append((sensor_name, engineered_df))
    print("\n\nFinished engineering\n")
    return engineered_dfs


def load_data(engineered_dfs):

    dataloader_config = "configs/dataloader_config.json"
    print(f"\n\nLoading {len(engineered_dfs)} engineered DataFrames\n")

    empty_df_count = 0

    # Filter out empty DataFrames with short sequences.
    non_empty_engineered_dfs = []
    config = load_config(dataloader_config)
    window_size = config["dataloader_steps"][0]["kwargs"]["window_size"]
    print(f"Window size: {window_size}")
    for sensor_name, df in engineered_dfs[:]:
        if len(df) <= window_size * 10:
            empty_df_count += 1
        else:
            non_empty_engineered_dfs.append((sensor_name, df))
    print(f"\n\n Dropped {empty_df_count} empty DataFrames\n")
    print(f"\n\n Loading in {len(non_empty_engineered_dfs)} tensors\n")

    list_of_dataloaders = []
    for i, df in enumerate(non_empty_engineered_dfs):
        print(f"\n\nProcessing {i+1}/{len(non_empty_engineered_dfs)}\n")
        print(f"Length of NaN values: {df[1].isna().sum().sum()}")
        sensor_name = df[0]
        model, train_loader, val_loader, test_loader = process_data(
            df[1], dataloader_config
        )
        list_of_dataloaders.append(
            (sensor_name, model, train_loader, val_loader, test_loader)
        )
    print("\n\nFinished loading data\n")
    return list_of_dataloaders


def train_model(dataloaders_list):
    """ """
    training_config_path = "configs/training_config.json"
    print(f"\n\nTraining {len(dataloaders_list)} models...")

    list_of_trained_models_and_metrics = []
    for i, (sensor_name, model, train_loader, val_loader, _) in enumerate(
        dataloaders_list
    ):
        print(f"\n\nTraining {i+1}/{len(dataloaders_list)}\n")
        model, train_metrics, val_metrics = process_data(
            (model, train_loader, val_loader), training_config_path
        )
        list_of_trained_models_and_metrics.append(
            (sensor_name, model, train_metrics, val_metrics)
        )
    print("\n\nFinished training\n")
    return list_of_trained_models_and_metrics


def model_evaluation():
    """"""
