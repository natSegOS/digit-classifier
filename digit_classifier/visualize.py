from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    path: Path
) -> None:
    matrix = confusion_matrix(labels.numpy(), predictions.numpy())
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_sample_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    path: Path,
    max_examples: int = 12
) -> None:
    num_examples = min(max_examples, len(inputs))

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for index in range(num_examples):
        image = inputs[index].reshape(8, 8).numpy()
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(
            f"Pred: {predictions[index].item()} | True: {labels[index].item()}"
        )

        axes[index].axis("off")

    for index in range(num_examples, len(axes)):
        axes[index].axis("off")

    fig.suptitle("Sample Predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_incorrect_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    path: Path,
    max_examples: int = 12
) -> None:
    incorrect_indices = (labels != predictions).nonzero(as_tuple=True)[0]

    if len(incorrect_indices) == 0:
        print("No incorrect predictions to visualize")

    selected_indices = incorrect_indices[:max_examples]

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for plot_index, sample_index in enumerate(selected_indices):
        image = inputs[sample_index].reshape(8, 8).numpy()
        axes[plot_index].imshow(image, cmap="gray")
        axes[plot_index].set_title(
            f"Pred: {predictions[sample_index].item()} | True: {labels[sample_index].item()}"
        )

        axes[plot_index].axis("off")

    for plot_index in range(len(selected_indices), len(axes)):
        axes[plot_index].axis("off")

    fig.suptitle("Incorrect Predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_noise_comparison(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    noisy_predictions: torch.Tensor,
    path: Path,
    max_examples: int = 12
) -> None:
    num_examples = min(max_examples, len(inputs))

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for index in range(num_examples):
        image = inputs[index].reshape(8, 8).numpy()
        axes[index].imshow(image, cmap="gray")
        axes[index].set_title(
            f"T:{labels[index].item()} C:{clean_predictions[index].item()} N:{noisy_predictions[index].item()}"
        )

        axes[index].axis("off")

    for index in range(num_examples, len(axes)):
        axes[index].axis("off")

    fig.suptitle("Noise Comparison (True / Clean / Noisy)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

