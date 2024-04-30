from utils.pipeline_types import TrainedModelItem, EvaluationItem
from typing import List
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

def evaluate_model(pipeline: List[TrainedModelItem]) -> EvaluationItem:

