from typing import TypeAlias
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# Type alias for the raw data stage
RawDataItem: TypeAlias = list[tuple[str, pd.DataFrame]]

# Type alias for the preprocessing stage
PreprocessedItem: TypeAlias = list[tuple[str, pd.DataFrame]]

# Type alias for the feature engineering stage
EngineeredItem: TypeAlias = list[tuple[str, pd.DataFrame]]

# Type alias for the data loading stage
DataLoaderItem: TypeAlias = list[
    tuple[str, nn.Module, DataLoader, DataLoader, DataLoader]
]

# Type alias for the model training stage
TrainedModelItem: TypeAlias = list[tuple[str, nn.Module, DataLoader, dict, dict]]

# Type alias for the evaluation stage
EvaluationItem: TypeAlias = list[tuple[str, dict, dict, dict]]
