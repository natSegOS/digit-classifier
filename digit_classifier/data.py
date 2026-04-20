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


def load_digit_data(
    test_size: float,
    batch_size: int,
    normalize_divisor: float,
    random_seed: int
) -> DatasetBundle:
    """
    Load the sklearn digits dataset, normalize it, split into train/test sets,
    and wrap tensors in DataLoaders
    """
    digits = load_digits()

    images = digits.images.astype(np.float32)
    labels = digits.target.astype(np.int64)

    images = images / normalize_divisor

    features = images.reshape(len(images), -1)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        x_test=x_test_tensor,
        y_test=y_test_tensor
    )
