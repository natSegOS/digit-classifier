from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import Config
from digit_classifier.model import DigitClassifier
from digit_classifier.visualize import save_processed_input_preview


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
    image_array = image_array / 255.0

    return torch.tensor(image_array.reshape(1, -1), dtype=torch.float32)


def build_model(config: Config, device: torch.device) -> DigitClassifier:
    model = DigitClassifier(
        input_features=config.input_features,
        hidden_features=config.hidden_features,
        num_classes=config.num_classes
    ).to(device)

    return model


def load_model_weights(
    model: DigitClassifier,
    model_path: Path,
    device: torch.device
) -> None:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


def predict_single_image(
    model: DigitClassifier,
    image_tensor: torch.Tensor,
    device: torch.device
) -> tuple[int, torch.Tensor]:
    """
    Predict the clas of a single image tensor
    """
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().squeeze(0)
        prediction = int(probabilities.argmax().item())

    return prediction, probabilities


def parse_args() -> argparse.Namespace:
    config = Config()

    parser = argparse.ArgumentParser(
        description="Predict a handwriten digit from an image using a trained PyTorch model"
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image file"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=config.model_path,
        help=f"Path to the trained model weights (default: {config.model_path})"
    )

    return parser.parse_args()


def main():
    """
    CLI entry-point for single-image prediction
    """
    args = parse_args()
    config = Config()

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Train the model first with python -m digit_classifier"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config=config, device=device)
    load_model_weights(model=model, model_path=args.model, device=device)

    image_tensor = load_custom_image(args.image)
    prediction, probabilities = predict_single_image(
        model=model,
        image_tensor=image_tensor,
        device=device
    )

    preview_path = config.output_dir / "processed_input_preview.png"
    save_processed_input_preview(image_tensor, preview_path)
    print(f"Saved processed preview to: {preview_path}")

    print(f"Input image: {args.image}")
    print(f"Predicted digit: {prediction}")
    print("Class probabilities:")
    for class_index, probability in enumerate(probabilities.tolist()):
        print(f" {class_index}: {probability:.4f}")


if __name__ == "__main__":
    main()
