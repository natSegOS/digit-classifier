from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


def load_custom_image(image_path: Path) -> torch.Tensor:
    """
    Load a custom digit image and convert it to the same format
    expected by the model:
    - grayscale
    - 8x8
    - flattened to 64 values
    - normalized to [0, 1]
    """
    image = Image.open(image_path).convert("L")
    image = image.resize((8, 8))

    image_array = np.asarray(image, dtype=np.float32)

    image_array = 255.0 - image_array
    image_array = image_array / 255.0

    return torch.tensor(image_array.reshape(1, -1), dtype=torch.float32)


def predict_single_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> int:
    """
    Predict the clas of a single image tensor
    """
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        prediction = logits.argmax(dim=1).item()

    return prediction
