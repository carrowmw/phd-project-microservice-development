# # phd_package/pipeline/stages/train_model.py

# """
# Module for training neural network models and tracking performance metrics.
# """

# import numpy as np

# from torch.optim import lr_scheduler

# import mlflow
# import mlflow.pytorch

# import sklearn.metrics

# from ..utils.training_helper import (
#     create_criterion,
#     create_optimiser,
#     map_model_to_mps,
#     map_tensor_to_mps,
#     validate_dataloader,
#     validate_model_and_criterion,
#     safe_tensor_to_numpy,
#     evaluate_model,
# )
# from ...utils.data_helper import DataLoaderItem, TrainedModelItem
# from ...utils.config_helper import (
#     get_epochs,
# )


# def train_model(
#     pipeline: DataLoaderItem,
#     **kwargs,
# ) -> TrainedModelItem:
#     """
#     Train and evaluate the neural network model.

#     Args:
#         pipeline (DataLoaderItem): A tuple containing the model, train dataloader, and validation dataloader.
#         **kwargs: Keyword arguments for configurations like epochs, criterion_config, and optimiser_config.

#     Returns:
#         TrainedModelItem: A tuple containing the trained model, train performance metrics, and validation performance metrics.
#     """
#     model, train_dataloader, val_dataloader = pipeline

#     epochs = get_epochs()

#     # Config setup and error handling
#     validate_dataloader(train_dataloader)
#     validate_dataloader(val_dataloader)
#     criterion = create_criterion()
#     validate_model_and_criterion(model, criterion)
#     optimiser = create_optimiser(model.parameters())
#     scheduler = lr_scheduler.StepLR(
#         optimiser,
#         step_size=kwargs.get("scheduler_step_size", 1),
#         gamma=kwargs.get("scheduler_gamma", 0.1),
#     )

#     map_model_to_mps(model, **kwargs)  # Map model to MPS if available

#     train_performance_metric_list = []
#     val_performance_metric_list = []

#     # Model training loop
#     for epoch in range(epochs):
#         model.train()  # Set model to training mode
#         total_train_loss = 0

#         # Training step
#         for batch, (X, y) in enumerate(train_dataloader):
#             X, y = X.float(), y.unsqueeze(-1).float()
#             X, y = map_tensor_to_mps(X, **kwargs), map_tensor_to_mps(y, **kwargs)
#             optimiser.zero_grad()

#             predictions = model(X)
#             loss = criterion(predictions, y)
#             loss.backward()
#             optimiser.step()

#             predictions_np = safe_tensor_to_numpy(predictions)
#             labels_np = safe_tensor_to_numpy(y)

#             mape_score = sklearn.metrics.mean_absolute_percentage_error(
#                 predictions_np, labels_np
#             )

#             if batch % 10 == 0:
#                 loss_score, current = loss.item(), batch
#                 step = batch // 100 * (epoch + 1)
#                 mlflow.log_metric("batch", f"{current}", step=step)
#                 mlflow.log_metric("batch_train_loss", f"{loss_score:2f}", step=step)
#                 mlflow.log_metric(
#                     "batch_train_mape_score", f"{mape_score:2f}", step=step
#                 )

#             total_train_loss += loss.item() * X.size(0)

#         scheduler.step()

#         # Evaluate model performance on both training and validation datasets
#         _, _, train_loss, train_mape, train_rmse, _ = evaluate_model(
#             model, train_dataloader, criterion, "train"
#         )
#         _, _, val_loss, val_mape, val_rmse, val_r2 = evaluate_model(
#             model, val_dataloader, criterion, "val"
#         )

#         # Collect train metrics for the current epoch
#         train_metrics_per_epoch = {
#             "Epoch": int(epoch + 1),
#             "Train loss": np.round(train_loss, 3),
#             "Train MAPE": np.round(train_mape, 3),
#             "Train RMSE": np.round(train_rmse, 3),
#         }
#         if epoch == 0:
#             print(
#                 f"Train Length: {len(train_dataloader)} | Val Length: {len(val_dataloader)}"
#             )
#         # Print key-value pairs for train metrics
#         print(
#             " | ".join(
#                 [f"{key}: {value}" for key, value in train_metrics_per_epoch.items()]
#             )
#         )
#         train_performance_metric_list.append(train_metrics_per_epoch)

#         # Collect validation metrics for the current epoch
#         val_metrics_per_epoch = {
#             "Epoch": int(epoch + 1),
#             "Val loss": np.round(val_loss, 3),
#             "Val MAPE": np.round(val_mape, 3),
#             "Val RMSE": np.round(val_rmse, 3),
#             "Val R2": np.round(val_r2, 3),
#         }
#         # Print key-value pairs for validation metrics
#         print(
#             " | ".join(
#                 [f"{key}: {value}" for key, value in val_metrics_per_epoch.items()]
#             )
#         )
#         val_performance_metric_list.append(val_metrics_per_epoch)

#     return model, train_performance_metric_list, val_performance_metric_list

import numpy as np
import torch
from torch.optim import lr_scheduler
import mlflow
import mlflow.pytorch
import sklearn.metrics
from typing import Tuple, List, Dict

from ..utils.training_helper import (
    create_criterion,
    create_optimiser,
    map_model_to_mps,
    map_tensor_to_mps,
    validate_dataloader,
    validate_model_and_criterion,
    safe_tensor_to_numpy,
    evaluate_model,
)
from ...utils.data_helper import DataLoaderItem, TrainedModelItem
from ...utils.config_helper import (
    get_epochs,
    get_early_stopping_patience,
    get_scheduler_step_size,
    get_scheduler_gamma,
)


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, **kwargs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = get_epochs()
        self.scheduler_step_size = get_scheduler_step_size()
        self.scheduler_gamma = get_scheduler_gamma()
        self.kwargs = kwargs

        self.criterion = create_criterion()
        validate_model_and_criterion(self.model, self.criterion)
        self.optimiser = create_optimiser(self.model.parameters())
        self.scheduler = lr_scheduler.StepLR(
            self.optimiser,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

        map_model_to_mps(self.model, **kwargs)

        self.train_performance_metric_list = []
        self.val_performance_metric_list = []

        self.best_val_loss = float("inf")
        self.patience = get_early_stopping_patience()
        self.patience_counter = 0

    def train_epoch(self) -> float:
        self.model.train()
        total_train_loss = 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.float(), y.unsqueeze(-1).float()
            X, y = map_tensor_to_mps(X, **self.kwargs), map_tensor_to_mps(
                y, **self.kwargs
            )
            self.optimiser.zero_grad()
            predictions = self.model(X)
            loss = self.criterion(predictions, y)
            loss.backward()
            self.optimiser.step()

            self.log_batch_metrics(batch, loss, predictions, y)

            total_train_loss += loss.item() * X.size(0)

        return total_train_loss / len(self.train_dataloader.dataset)

    def log_batch_metrics(
        self, batch: int, loss: torch.Tensor, predictions: torch.Tensor, y: torch.Tensor
    ):
        if batch % 10 == 0:
            predictions_np = safe_tensor_to_numpy(predictions)
            labels_np = safe_tensor_to_numpy(y)
            mape_score = sklearn.metrics.mean_absolute_percentage_error(
                predictions_np, labels_np
            )

            step = batch // 100 * (self.current_epoch + 1)
            mlflow.log_metric("batch", f"{batch}", step=step)
            mlflow.log_metric("batch_train_loss", f"{loss.item():2f}", step=step)
            mlflow.log_metric("batch_train_mape_score", f"{mape_score:2f}", step=step)

    def evaluate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        _, _, train_loss, train_mape, train_rmse, _ = evaluate_model(
            self.model, self.train_dataloader, self.criterion, "train"
        )
        _, _, val_loss, val_mape, val_rmse, val_r2 = evaluate_model(
            self.model, self.val_dataloader, self.criterion, "val"
        )

        train_metrics = {
            "Epoch": int(self.current_epoch + 1),
            "Train loss": np.round(train_loss, 3),
            "Train MAPE": np.round(train_mape, 3),
            "Train RMSE": np.round(train_rmse, 3),
        }

        val_metrics = {
            "Epoch": int(self.current_epoch + 1),
            "Val loss": np.round(val_loss, 3),
            "Val MAPE": np.round(val_mape, 3),
            "Val RMSE": np.round(val_rmse, 3),
            "Val R2": np.round(val_r2, 3),
        }

        return train_metrics, val_metrics

    def print_metrics(self, metrics: Dict[str, float]):
        print(" | ".join([f"{key}: {value}" for key, value in metrics.items()]))

    def check_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"Early stopping triggered after {self.current_epoch + 1} epochs")
            return True

        return False

    def train(
        self,
    ) -> Tuple[torch.nn.Module, List[Dict[str, float]], List[Dict[str, float]]]:
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.scheduler.step()

            train_metrics, val_metrics = self.evaluate()

            if epoch == 0:
                print(
                    f"Train Length: {len(self.train_dataloader)} | Val Length: {len(self.val_dataloader)}"
                )

            self.print_metrics(train_metrics)
            self.print_metrics(val_metrics)

            self.train_performance_metric_list.append(train_metrics)
            self.val_performance_metric_list.append(val_metrics)

            if self.check_early_stopping(val_metrics["Val loss"]):
                break

        return (
            self.model,
            self.train_performance_metric_list,
            self.val_performance_metric_list,
        )


def train_model(pipeline: DataLoaderItem, **kwargs) -> TrainedModelItem:
    model, train_dataloader, val_dataloader = pipeline

    validate_dataloader(train_dataloader)
    validate_dataloader(val_dataloader)

    trainer = Trainer(model, train_dataloader, val_dataloader, **kwargs)
    return trainer.train()
