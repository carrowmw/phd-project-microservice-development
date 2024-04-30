from typing import Tuple
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader

from src.training.training_datasets import TimeSeriesDataset
from src.models import model_definitions
from src.utils.general_utils import load_config


def sliding_windows(df: pd.DataFrame, **kwargs) -> Tuple[torch.tensor, torch.tensor]:
    """
    Generate sliding windows from the provided time-series data for sequence learning.

    Parameters:
    - df (pd.DataFrame): The time-series data from which windows will be generated.
    - window_size (int): Specifies the size of each sliding window.
    - input_feature_indices (list of ints | None): The indices of features to be considered as input.
    - target_feature_index (int): Index of the feature that needs to be predicted.
    - horizon (int): How many steps ahead the prediction should be.
    - stride (int, optional): Steps between the start of each window. Defaults to 1.
    - shapes (bool, optional): If set to True, it prints shapes of input and target for the first window. Defaults to False.

    Returns:
    - tuple: Contains inputs and targets as torch tensors.
    """
    dataloader_config = load_config("configs/dataloader_config.json")
    input_feature_indices = dataloader_config["kwargs"]["input_feature_indices"]
    target_feature_index = dataloader_config["kwargs"]["target_feature_index"]
    window_size = kwargs.get("window_size")
    horizon = kwargs.get("horizon")
    stride = kwargs.get("stride")

    if "sequence" not in df.columns:
        raise ValueError("'sequence' column not found in the DataFrame.")

    # Drop the "sequence" column from the DataFrame
    # df_without_sequence = df.drop(columns="sequence")

    # If input_feature_indices is None, generate indices from 0 to number of features in the DataFrame (excluding "sequence")
    if input_feature_indices is None and target_feature_index == 0:
        input_feature_indices = list(range(0, df.shape[1]))
    elif target_feature_index != 0:
        print("Target feature not in the first column of DataFrame - check configs")

    inputs = []
    targets = []

    unique_sequences = df["sequence"].unique()

    for sequence in unique_sequences:
        sequence_data = df[df["sequence"] == sequence]
        # sequence_input_data = sequence_data.drop(columns="sequence")

        assert (
            len(sequence_data) > window_size
        ), f"Number of records for sequence {sequence} is less than the window size."

        for i in range(0, len(sequence_data) - window_size - horizon + 1, stride):
            input_data = sequence_data.iloc[
                i : i + window_size, input_feature_indices
            ].values
            target_data = sequence_data.iloc[
                i + window_size + horizon - 1, target_feature_index
            ]

            inputs.append(input_data)
            targets.append(target_data)

    print(f"Generated {len(inputs)} sliding windows")
    inputs = np.array(inputs)
    targets = np.array(targets)
    feature_dim = len(input_feature_indices)

    pipeline = (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        feature_dim,
    )
    return pipeline


def create_dataloaders(
    pipeline: Tuple[torch.Tensor, torch.Tensor, int], **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Prepares training, validation, and test dataloaders using the inputs and targets generated by the sliding windows function.

    Parameters:
    - inputs (torch.Tensor): Inputs generated by the sliding windows function.
    - targets (torch.Tensor): Targets generated by the sliding windows function.
    - batch_size (int): Number of samples per batch to load.
    - shuffle (bool, optional): Whether to shuffle the data samples. Defaults to False.
    - num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
    - train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.7.
    - val_ratio (float, optional): Ratio of data to use for validation. Defaults to 0.15.

    Returns:
    - Tuple[DataLoader, DataLoader, DataLoader, int]: Contains train DataLoader, validation DataLoader, test DataLoader, and feature dimension.
    """
    inputs, targets = pipeline[0], pipeline[1]
    feature_dim = pipeline[2]
    batch_size = kwargs.get("batch_size")
    num_workers = kwargs.get("num_workers")
    shuffle = kwargs.get("shuffle")
    train_ratio = kwargs.get("train_ratio", 0.7)
    val_ratio = kwargs.get("val_ratio", 0.15)

    # Calculate train/val/test split indices
    total_size = len(inputs)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split data into train, validation, and test sets
    train_inputs, val_inputs, test_inputs = torch.split(
        inputs, [train_size, val_size, test_size]
    )
    train_targets, val_targets, test_targets = torch.split(
        targets, [train_size, val_size, test_size]
    )

    # Create custom PyTorch Dataset instances
    train_dataset = TimeSeriesDataset(train_inputs, train_targets)
    val_dataset = TimeSeriesDataset(val_inputs, val_targets)
    test_dataset = TimeSeriesDataset(test_inputs, test_targets)

    # Create DataLoader instances for train, validation, and test sets
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    pipeline = (train_dataloader, val_dataloader, test_dataloader, feature_dim)
    return pipeline


def add_model_to_dataloaders(
    pipeline: Tuple[DataLoader, DataLoader, DataLoader, int], **kwargs
) -> Tuple[torch.nn.Module, DataLoader, DataLoader, DataLoader]:
    dataloader_config = load_config("configs/dataloader_config.json")
    feature_dim = pipeline[3]

    model_type = dataloader_config["kwargs"]["model_type"]
    if model_type.lower() == "lstm":
        model_type = "LSTM" + "Model"
    elif model_type.lower() == "lstmautoencoder":
        model_type = "LSTMAutoencoder" + "Model"
    elif model_type.lower() == "randomforest":
        model_type = "RandomForest" + "Model"
    else:
        model_type = model_type.capitalize() + "Model"

    print(f"\n\nAttempting to load: {model_type}\n\n")

    if model_type not in dir(model_definitions):
        raise ValueError(f"Model type {model_type} not found in model_definitions.py")

    ModelClass = getattr(model_definitions, model_type)
    model = ModelClass(feature_dim)
    print("Model loaded successfully.")

    return (model, pipeline[0], pipeline[1], pipeline[2])
