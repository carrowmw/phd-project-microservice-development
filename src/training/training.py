"""
Module for training neural network models and tracking performance metrics.
"""

import numpy as np

from torch.optim import lr_scheduler

from src.utils.training_utils import (
    create_criterion,
    create_optimiser,
    map_model_to_mps,
    map_tensor_to_mps,
    validate_dataloader,
    validate_model_and_criterion,
)
from src.utils.pipeline_types import DataLoaderItem, TrainedModelItem

from src.training.evaluation import evaluate_model


def train(
    pipeline: DataLoaderItem,
    **kwargs,
) -> TrainedModelItem:
    """
    Train and evaluate the neural network model.

    Args:
        pipeline (DataLoaderItem): A tuple containing the model, train dataloader, and validation dataloader.
        **kwargs: Keyword arguments for configurations like epochs, criterion_config, and optimiser_config.

    Returns:
        TrainedModelItem: A tuple containing the trained model, train performance metrics, and validation performance metrics.
    """
    model, train_dataloader, val_dataloader = pipeline
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

    # Model training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        # Training step
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

        # Evaluate model performance on both training and validation datasets
        _, _, train_loss, train_mae, train_rmse, _ = evaluate_model(
            model, train_dataloader, criterion
        )
        _, _, val_loss, val_mae, val_rmse, val_r2 = evaluate_model(
            model, val_dataloader, criterion
        )

        # Collect train metrics for the current epoch
        train_metrics_per_epoch = {
            "Epoch": int(epoch + 1),
            "Train loss": np.round(train_loss, 3),
            "Train MAE": np.round(train_mae, 3),
            "Train RMSE": np.round(train_rmse, 3),
        }
        if epoch == 0:
            print(
                f"Train Length: {len(train_dataloader)} | Val Length: {len(val_dataloader)}"
            )
        # Print key-value pairs for train metrics
        print(
            " | ".join(
                [f"{key}: {value}" for key, value in train_metrics_per_epoch.items()]
            )
        )
        train_performance_metric_list.append(train_metrics_per_epoch)

        # Collect validation metrics for the current epoch
        val_metrics_per_epoch = {
            "Epoch": int(epoch + 1),
            "Val loss": np.round(val_loss, 3),
            "Val MAE": np.round(val_mae, 3),
            "Val RMSE": np.round(val_rmse, 3),
            "Val R2": np.round(val_r2, 3),
        }
        # Print key-value pairs for validation metrics
        print(
            " | ".join(
                [f"{key}: {value}" for key, value in val_metrics_per_epoch.items()]
            )
        )
        val_performance_metric_list.append(val_metrics_per_epoch)

    return model, train_performance_metric_list, val_performance_metric_list
