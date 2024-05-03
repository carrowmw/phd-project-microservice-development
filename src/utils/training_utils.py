import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_tensor_to_numpy(tensor):
    """
    Safely converts a tensor to a numpy array.

    Parameters:
    - tensor: A PyTorch tensor.

    Returns:
    - A numpy array converted from the input tensor.
    """
    try:
        return tensor.detach().cpu().numpy()
    except AttributeError as e:
        raise ValueError("Input is not a tensor.") from e


def validate_dataloader(dataloader):
    """Ensures the dataloader is initialized and not empty."""
    if not isinstance(dataloader, DataLoader):
        raise TypeError(
            "The dataloader must be an instance of torch.utils.data.DataLoader."
        )
    if len(dataloader) == 0:
        raise ValueError(
            "The dataloader is empty. Please provide a dataloader with data."
        )


def get_config(kwargs, key, default_value):
    """
    Safely retrieves a configuration value from kwargs with a default.
    Validates type or value range if necessary.
    """
    value = kwargs.get(key, default_value)
    # Add specific validations if necessary, e.g., check type or value range
    return value


def validate_model_and_criterion(model, criterion):
    """Validates that model and criterion are initialized."""
    if model is None:
        raise ValueError("Model is not initialized.")
    if criterion is None:
        raise ValueError("Criterion (loss function) is not initialized.")


def check_mps_availability(**kwargs):
    """
    Checks if MPS (Metal Performance Shaders) is available and returns the device.

    Parameters:
    - **kwargs: Keyword arguments, expecting 'device' to specify the desired device type.

    Returns:
    - torch.device: The device to use, either 'mps' if available or 'cpu' as fallback.
    """
    device_type = kwargs.get("device")
    if device_type is None:
        device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    elif device_type == "mps" and not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
            )
        device_type = "cpu"  # Fallback to CPU if MPS is not available
    return torch.device(device_type)


def map_model_to_mps(model: nn.Module, **kwargs):
    """
    Moves the given model to MPS if available.

    Parameters:
    - model: The PyTorch model to be moved.
    - **kwargs: Keyword arguments for device selection and additional configurations.
    """
    device = check_mps_availability(**kwargs)
    model.to(device)
    print(f"{type(model).__name__} successfully mapped to {device}.")


def map_tensor_to_mps(tensor: torch.Tensor, **kwargs):
    """
    Moves the given tensor to MPS if available.

    Parameters:
    - tensor: The PyTorch tensor to be moved.
    - **kwargs: Keyword arguments for device selection and additional configurations.

    Returns:
    - torch.Tensor: The tensor moved to the specified device.
    """
    device = check_mps_availability(**kwargs)
    return tensor.to(device)


def create_optimiser(model_params, **kwargs):
    """
    Creates and returns a PyTorch optimiser based on the provided arguments.

    This function allows for dynamic creation of an optimiser for a model,
    using the specified learning rate, momentum, and other relevant parameters.

    Parameters:
    - model_params (iterable): The parameters of the model to optimise.
    - **kwargs: Arbitrary keyword arguments including:
        - name (str): The name of the optimiser to create (default: 'adam').
        - lr (float): The learning rate (default: 0.01).
        - momentum (float): The momentum used with some optimisers (default: 0.9).

    Returns:
    - optimiser (torch.optim.Optimizer): The created PyTorch optimiser.

    Raises:
    - ValueError: If an unsupported optimiser name is provided.

    Example Usage:
    ```python
    model_params = model.parameters()
    optimiser_kwargs = {'name': 'adam', 'lr': 0.001}
    optimiser = create_optimiser(model_params, **optimiser_kwargs)
    ```
    """
    optimiser_name = kwargs.get("name", "adam").lower()
    lr = kwargs.get("lr", 0.01)  # Default learning rate
    momentum = kwargs.get("momentum", 0.9)  # Default momentum

    if optimiser_name == "adam":
        optimiser = torch.optim.Adam(model_params, lr=lr)
    elif optimiser_name == "sgd":
        optimiser = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimiser: {optimiser_name}")

    return optimiser


def create_criterion(**kwargs):
    """
    Creates and returns a PyTorch loss function (criterion) based on the provided arguments.

    This function facilitates the dynamic selection of a loss function for model training,
    according to the specified criterion name.

    Parameters:
    - **kwargs: Arbitrary keyword arguments including:
        - name (str): The name of the criterion to create (default: 'mse').

    Returns:
    - criterion (torch.nn.modules.loss._Loss): The PyTorch loss function.

    Raises:
    - ValueError: If an unsupported criterion name is provided.

    Example Usage:
    ```python
    criterion_kwargs = {'name': 'crossentropy'}
    criterion = create_criterion(**criterion_kwargs)
    ```
    """
    criterion_name = kwargs.get("name", "mse").lower()

    if criterion_name == "mse":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    return criterion


def create_performance_metrics_df(metric_dict: dict) -> pd.DataFrame:
    return None


###### NEEDS UPDATING #######
def save_model_state(
    model,
    model_name,
    folder_name,
):
    """
    Save the state of a model to disk.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name for the saved model file (without extension).
        path (str): Directory where the model should be saved.
    """
    path = "C:\\#code\\#python\\#current\\mres-project\\analysis_files\\"
    path = os.path.join(path, folder_name)
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(path, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)


def compute_performance_metrics(predictions, targets):
    """
    Compute common performance metrics for regression.

    Parameters:
        - predictions (torch.Tensor): The predicted values.
        - targets (torch.Tensor): The true target values.

    Returns:
        dict: A dictionary containing the following metrics:
            - MAE (float): Mean Absolute Error.
            - MSE (float): Mean Squared Error.
            - rmse (float): Root Mean Squared Error.
            - R^2 (float): R squared or Coefficient of Determination.

    Note:
        The function assumes the predictions and targets are torch tensors.
        They are then flattened and detached before computation.
    """
    predictions_flat_np = predictions.flatten()
    targets_flat_np = targets.flatten()

    return {
        "MAE": mean_absolute_error(targets_flat_np, predictions_flat_np),
        "MSE": mean_squared_error(targets_flat_np, predictions_flat_np),
        "rmse": np.sqrt(mean_absolute_error(targets_flat_np, predictions_flat_np)),
        "R^2": r2_score(targets_flat_np, predictions_flat_np),
    }
