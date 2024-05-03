from typing import TypeAlias, Tuple, List, get_args
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def check_type(item, expected_type):
    assert isinstance(item, tuple), f"Expected item to be a tuple, but got {type(item)}"
    assert len(item) == len(
        get_args(expected_type)
    ), f"Expected item to have {len(get_args(expected_type))} elements, but got {len(item)}"
    for i, (elem, expected_elem_type) in enumerate(zip(item, get_args(expected_type))):
        assert isinstance(
            elem, expected_elem_type
        ), f"Expected element {i} to be of type {expected_elem_type}, but got {type(elem)}"


# Type alias for the raw data stage
RawDataItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the preprocessing stage
PreprocessedItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the feature engineering stage
EngineeredItem: TypeAlias = list[Tuple[str, pd.DataFrame]]

# Type alias for the data loading stage
DataLoaderItem: TypeAlias = list[
    Tuple[str, nn.Module, DataLoader, DataLoader, DataLoader]
]

# Type alias for the model training stage
TrainedModelItem: TypeAlias = list[Tuple[str, nn.Module, DataLoader, List, List]]

# Type alias for the evaluation stage
EvaluationItem: TypeAlias = list[Tuple[str, np.ndarray, np.ndarray, dict]]
