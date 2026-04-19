import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
