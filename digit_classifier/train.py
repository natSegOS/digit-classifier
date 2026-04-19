from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    train_losses: list[float]
    train_accuracies: list[float]


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        predicted_classes = logits.argmax(dim=1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples
    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int
) -> TrainingHistory:
    history = TrainingHistory(train_losses=[], train_accuracies=[])

    for epoch_index in range(num_epochs):
        average_loss, accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        history.train_losses.append(average_loss)
        history.train_accuracies.append(accuracy)

        print(
            f"Epoch {epoch_index + 1:02d}/{num_epochs:02d} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Train Accuracy: {accuracy:.4f}"
        )

    return history
