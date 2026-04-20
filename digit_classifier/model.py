import torch
from torch import nn


class DigitClassifier(nn.Module):
    """
    A simple multilayer perceptron (MLP) for classifying
    8x8 handwritten digit images

    Architecture:
        64 input features
        -> 128 hidden units
        -> ReLU activation
        -> 10 output logits
    """

    def __init__(self, input_features: int, hidden_features: int, num_classes: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


