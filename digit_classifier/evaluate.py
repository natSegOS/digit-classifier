from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    average_loss: float
    accuracy: float
    all_inputs: torch.Tensor
    all_labels: torch.Tensor
    all_predictions: torch.Tensor


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> EvaluationResult:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    all_inputs = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * inputs.size(0)
            correct_predictions += (predictions == labels).sum().item()
            total_examples += labels.size(0)

            all_inputs.append(inputs.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return EvaluationResult(
        average_loss=average_loss,
        accuracy=accuracy,
        all_inputs=torch.cat(all_inputs, dim=0),
        all_labels=torch.cat(all_labels, dim=0),
        all_predictions=torch.cat(all_predictions, dim=0)
    )


def evaluate_under_noise(
    model: nn.Module,
    inputs: torch.Tensor,
    noise_std: float,
    device: torch.device
) -> torch.Tensor:
    """
    Apply Gaussian noise to inputs and return predictions
    """
    model.eval()

    noisy_inputs = inputs + torch.randn_like(inputs) * noise_std
    noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)

    with torch.no_grad():
        logits = model(noisy_inputs.to(device))
        predictions = logits.argmax(dim=1).cpu()

    return predictions
