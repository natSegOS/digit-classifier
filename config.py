from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the project
    """

    # Reproductibility
    random_seed: int = 42

    # Data
    test_size: float = 0.2
    normalize_divisor: float = 16.0 # sklearn digits pixels range from 0 to 16
    num_classes: int = 10
    input_height: int = 8
    input_width: int = 8
    input_features: int = input_height * input_width

    # Model
    hidden_features: int = 128

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10

    # Output paths
    output_dir: Path = Path("results")
    model_path: Path = output_dir / "digit_classifier_model.pt"
    confusion_matrix_path: Path = output_dir / "confusion_matrix.png"
    sample_predictions_path: Path = output_dir / "sample_predictions.png"
    incorrect_predictions_path: Path = output_dir / "incorrect_predictions.png"
    noise_comparison_path: Path = output_dir / "noise_comparison.png"

