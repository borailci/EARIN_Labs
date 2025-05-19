import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
from dataset import get_class_names


def load_model(model_path, device):
    """Load the trained model"""
    model = get_model(
        num_classes=10, hidden_size=128
    )  # Use same hidden_size as saved model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame):
    """Preprocess frame for model input"""
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to PIL Image
    frame = Image.fromarray(frame)

    # Apply transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transform(frame).unsqueeze(0)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "results/best_model.pth"  # Update with your model path
    model = load_model(model_path, device)

    # Get class names
    class_names = get_class_names()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Create window
    cv2.namedWindow("Real-time Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Gesture Recognition", 800, 600)

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Preprocess frame
        input_tensor = preprocess_frame(frame)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor.to(device))
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_label].item()

        # Get predicted class
        pred_class = class_names[f"{pred_label+1:02d}"]

        # Add text to frame
        cv2.putText(
            frame,
            f"Prediction: {pred_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Show frame
        cv2.imshow("Real-time Gesture Recognition", frame)

        # Check for exit
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
