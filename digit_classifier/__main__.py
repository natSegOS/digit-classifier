import torch
from torch import nn
from torch.optim import Adam

from config import Config
from digit_classifier.data import load_digit_data
from digit_classifier.evaluate import evaluate_model, evaluate_under_noise
from digit_classifier.model import DigitClassifier
from digit_classifier.train import train_model
from digit_classifier.utils import ensure_directory, set_random_seed
from digit_classifier.visualize import (
    save_confusion_matrix,
    save_incorrect_predictions,
    save_noise_comparison,
    save_sample_predictions
)


def main() -> None:
    config = Config()

    set_random_seed(config.random_seed)
    ensure_directory(config.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_digit_data(
        test_size=config.test_size,
        batch_size=config.batch_size,
        normalize_divisor=config.normalize_divisor,
        random_seed=config.random_seed
    )

    model = DigitClassifier(
        input_features=config.input_features,
        hidden_features=config.hidden_features,
        num_classes=config.num_classes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print("Starting training...")
    train_model(
        model=model,
        train_loader=dataset.train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs
    )

    print("Evaluating model...")
    evaluation = evaluate_model(
        model=model,
        data_loader=dataset.test_loader,
        loss_fn=loss_fn,
        device=device
    )

    print(
        f"Test Loss: {evaluation.average_loss:.4f} | "
        f"Test Accuracy: {evaluation.accuracy:.4%}"
    )

    torch.save(model.state_dict(), config.model_path)
    print(f"Saved model to: {config.model_path}")

    save_confusion_matrix(
        labels=evaluation.all_labels,
        predictions=evaluation.all_predictions,
        path=config.confusion_matrix_path
    )

    save_sample_predictions(
        inputs=evaluation.all_inputs,
        labels=evaluation.all_labels,
        predictions=evaluation.all_predictions,
        path=config.sample_predictions_path
    )

    save_incorrect_predictions(
        inputs=evaluation.all_inputs,
        labels=evaluation.all_labels,
        predictions=evaluation.all_predictions,
        path=config.incorrect_predictions_path
    )

    noisy_predictions = evaluate_under_noise(
        model=model,
        inputs=evaluation.all_inputs,
        noise_std=config.noise_std,
        device=device
    )

    save_noise_comparison(
        inputs=evaluation.all_inputs,
        labels=evaluation.all_labels,
        clean_predictions=evaluation.all_predictions,
        noisy_predictions=noisy_predictions,
        path=config.noise_comparison_path
    )

    print(
        "Saved visualizations:\n"
        f" - {config.confusion_matrix_path}\n"
        f" - {config.sample_predictions_path}\n"
        f" - {config.incorrect_predictions_path}\n"
        f" - {config.noise_comparison_path}"
    )


if __name__ == "__main__":
    main()
