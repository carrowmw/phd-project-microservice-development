# phd_package/pipeline/stages/test_model.py

"""
Module for evaluating trained models on test datasets and calculating performance metrics.
"""

from typing import List
import numpy as np

import torch
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ...utils.data_helper import TrainedModelItem, TestItem
from ..utils.training_helper import (
    validate_dataloader,
    create_criterion,
    map_tensor_to_mps,
    safe_tensor_to_numpy,
)


def evaluate_model(model, dataloader, criterion, stage, **kwargs):
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

    predictions_list = []
    labels_list = []

    with torch.no_grad():  # No need to track gradients
        for _, (X, y) in enumerate(dataloader):
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(
                y, **kwargs
            )  # Map tensors to Metal Performance Shaders (MPS) if available

            predictions = model(X)
            loss = criterion(predictions, y)

            # Detach predictions and labels from the graph and move to CPU for metric calculation
            predictions_np = safe_tensor_to_numpy(predictions)
            labels_np = safe_tensor_to_numpy(y)

            predictions_list.extend(predictions_np)
            labels_list.extend(labels_np)

            # Accumulate metrics
            total_loss += loss.item() * X.size(0)
            total_mae += mean_absolute_error(labels_np, predictions_np) * X.size(0)
            total_rmse += np.sqrt(
                mean_squared_error(labels_np, predictions_np)
            ) * X.size(0)
            if X.size(0) > 1:  # Avoid error when batch size is 1
                total_r2_score += r2_score(labels_np, predictions_np) * X.size(0)
            total_count += X.size(0)

    # Calculate average metrics
    avg_loss = total_loss / total_count
    avg_mae = total_mae / total_count
    avg_rmse = total_rmse / total_count
    avg_r2_score = total_r2_score / total_count

    mlflow.log_metric(f"avg_{stage}_loss", f"{avg_loss:2f}")
    mlflow.log_metric(f"avg_{stage}_mae", f"{avg_mae:2f}")
    mlflow.log_metric(f"avg_{stage}_rmse", f"{avg_rmse:2f}")
    mlflow.log_metric(f"avg_{stage}_r2_score", f"{avg_r2_score:2}")

    predictions_array = np.array(predictions_list)
    labels_array = np.array(labels_list)

    return predictions_array, labels_array, avg_loss, avg_mae, avg_rmse, avg_r2_score


def test_model(pipeline: List[TrainedModelItem], **kwargs) -> TestItem:
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
        evaluate_model(model, test_dataloader, criterion, "test", **kwargs)
    )
    test_metrics = {
        "Test loss": np.round(test_loss, 3),
        "Test MAE": np.round(test_mae, 3),
        "Test RMSE": np.round(test_rmse, 3),
        "Test R2": np.round(test_r2, 3),
    }
    # Print key value pairs
    print(" | ".join([f"{key}: {value}" for key, value in test_metrics.items()]))

    print("Prediction Length: ", len(test_predictions))
    print("Labels Length: ", len(test_labels))

    return test_predictions, test_labels, test_metrics
