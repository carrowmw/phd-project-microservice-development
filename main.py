import os
import torch
from src.pipeline import (
    download_raw_data,
    preprocess_raw_data,
    apply_feature_engineering,
    load_data,
    train_model,
)
from src.utils.fstore_utils import (
    create_raw_file_path_from_config,
    create_dataloaders_file_path_from_config,
    load_raw_data,
    load_preprocessed_data,
    load_engineered_data,
    load_dataloaders,
    load_trained_models,
    save_raw_data,
    save_preprocessed_data,
    save_engineered_data,
    save_dataloaders,
    save_trained_models,
)


def run_pipeline():

    if torch.backends.mps.is_available():
        print("MPS backend available")
    else:
        print("Warning, cannot move to MPS backend")

    # Checkpoint 1: Check if raw data exists
    raw_data_path = create_raw_file_path_from_config()
    if os.path.exists(raw_data_path):
        print("Preprocessed data found. Skipping preprocessing step.")
        raw_dfs = load_raw_data()
    else:
        print("Preprocessed data not found. Running preprocessing step.")
        raw_dfs = download_raw_data()
        save_raw_data(raw_dfs, raw_data_path)

    # Checkpoint 2: Check if preprocessed data exists
    preprocessed_data_path = create_raw_file_path_from_config().replace(
        "raw", "preprocessed"
    )
    if os.path.exists(preprocessed_data_path):
        print("Preprocessed data found. Skipping preprocessing step.")
        preprocessed_dfs = load_preprocessed_data()
    else:
        print("Preprocessed data not found. Running preprocessing step.")
        preprocessed_dfs = preprocess_raw_data(raw_dfs)
        save_preprocessed_data(preprocessed_dfs, preprocessed_data_path)

    # Checkpoint 3: Check if engineered features exist
    engineered_data_path = create_raw_file_path_from_config().replace(
        "raw", "engineered"
    )
    if os.path.exists(engineered_data_path):
        print("Engineered features found. Skipping feature engineering step.")
        engineered_dfs = load_engineered_data()
    else:
        print("Engineered features not found. Running feature engineering step.")
        engineered_dfs = apply_feature_engineering(preprocessed_dfs)
        save_engineered_data(engineered_dfs, engineered_data_path)

    # Checkpoint 4: Check if data loaders exist
    data_loaders_path = create_dataloaders_file_path_from_config()
    if os.path.exists(data_loaders_path):
        print("Data loaders found. Skipping data loading step.")
        data_loaders_list = load_dataloaders()
    else:
        print("Data loaders not found. Running data loading step.")
        data_loaders_list = load_data(engineered_dfs)
        save_dataloaders(data_loaders_list, data_loaders_path)

    # Checkpoint 5: Check if trained models exist
    trained_models_path = create_dataloaders_file_path_from_config().replace(
        "dataloaders", "trained_models"
    )
    if os.path.exists(trained_models_path):
        print("Trained models found. Skipping training step.")
        trained_models_and_metrics_list = load_trained_models()
    else:
        print("Trained models not found. Running training step.")
        trained_models_and_metrics_list = train_model(data_loaders_list)
        save_trained_models(trained_models_and_metrics_list, trained_models_path)

    # # Checkpoint 6: Run evaluation
    # print("Running evaluation step.")
    # evaluation_metrics_list = evaluate_model(
    #     trained_models_and_metrics_list, data_loaders_list
    # )
    # print(evaluation_metrics_list)


if __name__ == "__main__":
    run_pipeline()
