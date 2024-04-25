from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.utils.training_utils import (
    create_criterion,
    create_optimiser,
    map_model_to_mps,
    map_tensor_to_mps,
    safe_tensor_to_numpy,
    validate_dataloader,
    validate_model_and_criterion,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, dataloader, criterion, **kwargs):
    """
    Evaluate the model performance on a given dataloader.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): The DataLoader for evaluation data.
        criterion: The loss function used for evaluation.

    Returns:
        tuple: Returns a tuple containing average loss, MAE, RMSE, and R2 score.
    """
    model.eval()  # Set model to evaluation mode
    total_loss, total_mae, total_rmse, total_r2_score, total_count = 0, 0, 0, 0, 0

    with torch.no_grad():  # No need to track gradients
        for X, y in dataloader:
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(
                y, **kwargs
            )  # Map tensors to Metal Performance Shaders (MPS) if available

            predictions = model(X)
            loss = criterion(predictions, y)

            # Detach predictions and labels from the graph and move to CPU for metric calculation
            predictions_np = safe_tensor_to_numpy(predictions)
            labels_np = safe_tensor_to_numpy(y)

            # Accumulate metrics
            total_loss += loss.item() * X.size(0)
            total_mae += mean_absolute_error(labels_np, predictions_np) * X.size(0)
            total_rmse += np.sqrt(
                mean_squared_error(labels_np, predictions_np)
            ) * X.size(0)
            total_r2_score += r2_score(labels_np, predictions_np) * X.size(0)
            total_count += X.size(0)

    # Calculate average metrics
    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    avg_rmse = total_rmse / total_count
    avg_r2_score = total_r2_score / total_count

    return avg_loss, avg_mae, avg_rmse, avg_r2_score


def train(
    pipeline: Tuple[nn.Module, DataLoader, DataLoader, DataLoader],
    **kwargs,
) -> nn.Module:
    """
    Train and evaluate the neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the val data.
        **kwargs: Keyword arguments for configurations like epochs, criterion_config, and
        optimiser_config.

    Prints:
        Loss and metric information for each epoch.
    """
    (model, train_dataloader, val_dataloader) = pipeline
    # Config setup and error handling
    validate_dataloader(train_dataloader)
    validate_dataloader(val_dataloader)
    criterion = create_criterion(**kwargs.get("criterion_config", {}))
    validate_model_and_criterion(model, criterion)
    optimiser = create_optimiser(
        model.parameters(), **kwargs.get("optimiser_config", {})
    )
    scheduler = lr_scheduler.StepLR(
        optimiser,
        step_size=kwargs.get("scheduler_step_size", 1),
        gamma=kwargs.get("scheduler_gamma", 0.1),
    )
    epochs = kwargs.get("epochs")
    map_model_to_mps(model, **kwargs)  # Map model to MPS if available

    train_performance_metric_list = []
    val_performance_metric_list = []

    # Model training
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        for X, y in train_dataloader:
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(y, **kwargs)
            optimiser.zero_grad()

            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item() * X.size(0)

        scheduler.step()

        # Evaluate model performance on both training and val datasets
        train_loss, train_mae, train_rmse, _ = evaluate_model(
            model, train_dataloader, criterion
        )
        val_loss, val_mae, val_rmse, val_r2 = evaluate_model(
            model, val_dataloader, criterion
        )

        train_metrics_dict = {
            "Epoch": int(epoch + 1),
            "Train loss": np.round(train_loss, 3),
            "Train MAE": np.round(train_mae, 3),
            "Train RMSE": np.round(train_rmse, 3),
        }
        if epoch == 0:
            print(
                f"Train Length: {len(train_dataloader)} | Val Length: {len(val_dataloader)}"
            )
        # Print key value pairs
        print(
            " | ".join([f"{key}: {value}" for key, value in train_metrics_dict.items()])
        )
        train_performance_metric_list.append(train_metrics_dict)

        val_metrics_dict = {
            "Epoch": int(epoch + 1),
            "Val loss": np.round(val_loss, 3),
            "Val MAE": np.round(val_mae, 3),
            "Val RMSE": np.round(val_rmse, 3),
            "Val R2": np.round(val_r2, 3),
        }
        # Print key value pairs
        print(
            " | ".join([f"{key}: {value}" for key, value in val_metrics_dict.items()])
        )
        val_performance_metric_list.append(val_metrics_dict)

    return model, train_performance_metric_list, val_performance_metric_list
