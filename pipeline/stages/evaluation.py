"""
Module for evaluating trained models on test datasets and calculating performance metrics.
"""

from typing import List
import numpy as np

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_helper import TrainedModelItem, EvaluationItem
from pipeline.utils.training_helper import (
    validate_dataloader,
    create_criterion,
    map_tensor_to_mps,
    safe_tensor_to_numpy,
)


def evaluate_model(model, dataloader, criterion, **kwargs):
    """
    Evaluate the model performance on a given dataloader.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): The DataLoader for evaluation data.
        criterion: The loss function used for evaluation.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the predicted values, true labels, average loss,
               MAE, RMSE, and R2 score.
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

    return predictions_np, labels_np, avg_loss, avg_mae, avg_rmse, avg_r2_score


def test_model(pipeline: List[TrainedModelItem], **kwargs) -> EvaluationItem:
    """
    Evaluate the trained model on the test dataset and calculate performance metrics.

    Args:
        pipeline (List[TrainedModelItem]): A list containing the trained model and test dataloader.

    Returns:
        EvaluationItem: A tuple containing the test predictions, test labels, and test metrics.
    """
    model, test_dataloader = pipeline
    criterion = create_criterion()
    validate_dataloader(test_dataloader)

    test_predictions, test_labels, test_loss, test_mae, test_rmse, test_r2 = (
        evaluate_model(model, test_dataloader, criterion, **kwargs)
    )
    test_metrics = {
        "Val loss": np.round(test_loss, 3),
        "Val MAE": np.round(test_mae, 3),
        "Val RMSE": np.round(test_rmse, 3),
        "Val R2": np.round(test_r2, 3),
    }
    # Print key value pairs
    print(" | ".join([f"{key}: {value}" for key, value in test_metrics.items()]))

    return test_predictions, test_labels, test_metrics
