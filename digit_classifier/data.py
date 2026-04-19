from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    x_test: torch.Tensor
    y_test: torch.Tensor
