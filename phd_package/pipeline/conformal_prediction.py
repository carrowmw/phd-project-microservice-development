# phd_package/pipeline/conformal_prediction.py

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ConformalPredictor:
    def __init__(self, significance_level: float = 0.1):
        """
        Initialize the conformal predictor.

        Args:
            significance_level: The significance level (alpha) for the prediction intervals.
                              Default is 0.1 for 90% prediction intervals.
        """
        self.significance_level = significance_level
        self.calibration_scores = None

    def compute_nonconformity(self, model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
        """
        Compute nonconformity scores for calibration.

        Args:
            model: The trained PyTorch model
            dataloader: DataLoader containing calibration data
            device: Device to run computations on ('cpu' or 'mps')

        Returns:
            Array of nonconformity scores
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                # Using absolute error as nonconformity score
                score = torch.abs(pred - y).cpu().numpy()
                scores.extend(score)

        return np.array(scores)

    def calibrate(self, model: nn.Module, cal_dataloader: DataLoader, device: str):
        """
        Calibrate the conformal predictor using a held-out calibration set.
        """
        nonconformity_scores = self.compute_nonconformity(model, cal_dataloader, device)
        # Store calibration scores sorted in ascending order
        self.calibration_scores = np.sort(nonconformity_scores)

    def predict(self, model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if self.calibration_scores is None:
            raise ValueError("Predictor must be calibrated before making predictions")

        # Calculate the quantile cutoff
        n = len(self.calibration_scores)
        q = np.ceil((n + 1) * (1 - self.significance_level)) / n
        prediction_radius = np.quantile(self.calibration_scores, q)

        model.eval()
        predictions = []
        lower_bounds = []
        upper_bounds = []

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                pred = model(X).cpu().numpy()
                predictions.extend(pred)
                lower_bounds.extend(pred - prediction_radius)
                upper_bounds.extend(pred + prediction_radius)

        return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)

def prepare_calibration_data(train_dataloader: DataLoader, cal_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Split training data to create a calibration set.

    Args:
        train_dataloader: Original training DataLoader
        cal_ratio: Proportion of training data to use for calibration

    Returns:
        Tuple of (new_train_dataloader, calibration_dataloader)
    """
    # Extract data from DataLoader
    X_train = []
    y_train = []
    for X, y in train_dataloader:
        X_train.append(X)
        y_train.append(y)

    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    # Split data
    X_train_new, X_cal, y_train_new, y_cal = train_test_split(
        X_train, y_train, test_size=cal_ratio, random_state=42
    )

    # Create new DataLoaders
    batch_size = train_dataloader.batch_size
    new_train_loader = DataLoader(
        list(zip(X_train_new, y_train_new)),
        batch_size=batch_size,
        shuffle=True
    )
    cal_loader = DataLoader(
        list(zip(X_cal, y_cal)),
        batch_size=batch_size,
        shuffle=False
    )

    return new_train_loader, cal_loader