# phd_package/pipeline/stages/test_model.py

"""
Module for evaluating trained models on test datasets and calculating performance metrics.
"""

from typing import List
import numpy as np

from ...utils.data_helper import TrainedModelItem, TestItem
from ..utils.training_helper import (
    validate_dataloader,
    create_criterion,
    evaluate_model,
)


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

    test_predictions, test_labels, test_loss, test_mape, test_rmse, test_r2 = (
        evaluate_model(model, test_dataloader, criterion, "test", **kwargs)
    )
    test_metrics = {
        "Test loss": np.round(test_loss, 3),
        "Test MAPE": np.round(test_mape, 3),
        "Test RMSE": np.round(test_rmse, 3),
        "Test R2": np.round(test_r2, 3),
    }
    # Print key value pairs
    print(" | ".join([f"{key}: {value}" for key, value in test_metrics.items()]))

    print("Prediction Length: ", len(test_predictions))
    print("Labels Length: ", len(test_labels))

    return test_predictions, test_labels, test_metrics
