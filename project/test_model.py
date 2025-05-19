import torch
from torchvision import transforms
from PIL import Image
import os
import glob
import re
import argparse

from model import get_model
from dataset import get_class_names


def test_model(model_path, data_dir):
    """
    Test model on dataset images and print filename with prediction

    Args:
        model_path (str): Path to the trained model weights
        data_dir (str): Directory containing the test images
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model(num_classes=10, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get class names
    class_names = get_class_names()

    # Define preprocessing transform
    transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Process all images in the data directory
    for folder_idx in range(10):  # 10 main folders (00 to 09)
        folder = f"{folder_idx:02d}"
        folder_path = os.path.join(data_dir, folder)

        if os.path.exists(folder_path):
            # Find all PNG files in this folder and its subdirectories
            frames = glob.glob(os.path.join(folder_path, "**/*.png"), recursive=True)

            for frame_path in frames:
                try:
                    # Load and preprocess image
                    image = Image.open(frame_path).convert("L")
                    tensor = transform(image).unsqueeze(0).to(device)

                    # Get prediction
                    with torch.no_grad():
                        outputs = model(tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)

                    pred_class = predicted.item()
                    conf_score = confidence.item()

                    # Get relative path for cleaner output
                    rel_path = os.path.relpath(frame_path, data_dir)

                    # Print prediction
                    print(f"File: {rel_path}")
                    print(
                        f"Prediction: {class_names[pred_class]} (confidence: {conf_score:.2f})"
                    )
                    print("-" * 80)

                except Exception as e:
                    print(f"Error processing {frame_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Test Model on Dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data", type=str, default="data", help="Path to data directory"
    )
    args = parser.parse_args()

    test_model(args.model, args.data)


if __name__ == "__main__":
    main()
