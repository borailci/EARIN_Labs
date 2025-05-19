import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
from datetime import datetime
import mediapipe as mp

from model import get_model
from dataset import get_class_names


class RealtimeGestureRecognizer:
    def __init__(self, model_path, device=None):
        """
        Initialize the real-time gesture recognizer

        Args:
            model_path (str): Path to the trained model weights
            device (torch.device): Device to run inference on
        """
        # Set device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Load model
        self.model = get_model(num_classes=10, hidden_size=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Get class names
        self.class_names = get_class_names()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Define preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (64, 64), interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Initialize webcam
        self.cap = None

        # Create directories for saving images
        self.capture_dir = "captured_data"
        os.makedirs(self.capture_dir, exist_ok=True)

        # Counter for saved images
        self.image_counter = 0

        # Frame processing parameters
        self.roi_padding = 50  # Padding around the hand bounding box
        self.min_brightness = 30  # Minimum average brightness for valid frame
        self.max_brightness = 225  # Maximum average brightness for valid frame

    def get_hand_roi(self, frame):
        """
        Extract hand region using MediaPipe Hands and process it to match training data format

        Args:
            frame (numpy.ndarray): BGR frame from webcam

        Returns:
            tuple: (ROI image or None, status message)
        """
        if frame is None:
            return None, "No frame captured"

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, "No hand detected"

        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get bounding box of hand
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Calculate center and size of hand
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        size = max(x_max - x_min, y_max - y_min)

        # Add padding to make it square and consistent with dataset
        size = int(size * 1.3)  # Add 30% padding

        # Calculate new boundaries ensuring square aspect ratio
        x_min = max(0, center_x - size // 2)
        x_max = min(w, center_x + size // 2)
        y_min = max(0, center_y - size // 2)
        y_max = min(h, center_y + size // 2)

        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None, "Invalid ROI size"

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Ensure consistent size with dataset (64x64)
        gray_roi = cv2.resize(gray_roi, (64, 64), interpolation=cv2.INTER_LANCZOS4)

        # Enhance contrast to match dataset characteristics
        gray_roi = cv2.equalizeHist(gray_roi)

        # Check brightness
        avg_brightness = np.mean(gray_roi)
        if avg_brightness < self.min_brightness:
            return None, "Image too dark"
        if avg_brightness > self.max_brightness:
            return None, "Image too bright"

        return gray_roi, "OK"

    def preprocess_frame(self, frame):
        """
        Preprocess webcam frame to match training data format exactly

        Args:
            frame (numpy.ndarray): BGR frame from webcam

        Returns:
            tuple: (preprocessed tensor, processing steps images, status message)
        """
        # Get hand ROI
        roi, status = self.get_hand_roi(frame)
        if roi is None:
            return None, None, status

        # Convert to PIL Image
        pil_image = Image.fromarray(roi)

        # Store processing steps
        steps = {
            "original": frame,
            "grayscale": cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR),
            "roi": cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR),
        }

        # Apply preprocessing transform (now simpler since we handle most preprocessing in get_hand_roi)
        try:
            # Convert to tensor and normalize to [-1, 1]
            tensor = torch.from_numpy(roi).float()
            tensor = (
                tensor.unsqueeze(0) / 255.0
            )  # Add channel dimension and normalize to [0, 1]
            tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]

            # Convert normalized tensor back to image for visualization
            vis_tensor = tensor.clone()
            vis_tensor = (vis_tensor * 0.5 + 0.5) * 255
            vis_tensor = vis_tensor.byte().squeeze().cpu().numpy()
            steps["processed"] = cv2.cvtColor(vis_tensor, cv2.COLOR_GRAY2BGR)

            return tensor.unsqueeze(0).to(self.device), steps, "OK"
        except Exception as e:
            return None, None, f"Preprocessing error: {str(e)}"

    def predict(self, frame):
        """
        Make prediction on a single frame

        Args:
            frame (numpy.ndarray): BGR frame from webcam

        Returns:
            tuple: (predicted class name, confidence score, processing steps, status message)
        """
        # Preprocess frame
        tensor, steps, status = self.preprocess_frame(frame)
        if tensor is None:
            return None, 0.0, steps, status

        # Get prediction
        try:
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_class = predicted.item()
            conf_score = confidence.item()

            return self.class_names[pred_class], conf_score, steps, "OK"
        except Exception as e:
            return None, 0.0, steps, f"Prediction error: {str(e)}"

    def create_visualization(self, frame, steps, prediction, confidence, status):
        """Create a visualization of all processing steps"""
        if steps is None:
            # Create error display
            height, width = 480, 640
            display = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                display,
                f"Status: {status}",
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return display

        # Define target size for each image
        target_height = 240
        target_width = 320

        processed_images = []

        # Process each step
        for name, img in steps.items():
            # Resize image
            resized = cv2.resize(img, (target_width, target_height))

            # Create title bar
            title_bar = np.zeros((30, target_width, 3), dtype=np.uint8)
            cv2.putText(
                title_bar,
                name,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Combine title and image
            combined = np.vstack([title_bar, resized])
            processed_images.append(combined)

        # Create 2x2 grid
        top_row = np.hstack(processed_images[:2])
        bottom_row = np.hstack(processed_images[2:])
        grid = np.vstack([top_row, bottom_row])

        # Add prediction text
        if prediction:
            text = f"Prediction: {prediction} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        else:
            text = f"Status: {status}"
            color = (0, 0, 255)

        cv2.putText(
            grid,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        # Add instructions
        cv2.putText(
            grid,
            "Show hand in frame for detection",
            (10, grid.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Add save instruction
        cv2.putText(
            grid,
            "Press 's' to save, 'q' to quit",
            (10, grid.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        return grid

    def save_capture(self, steps, prediction, confidence):
        """
        Save the current frame and its processed versions

        Args:
            steps (dict): Dictionary containing processing step images
            prediction (str): Predicted class name
            confidence (float): Confidence score
        """
        if steps is None:
            print("Cannot save invalid frame")
            return

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory for this capture
        capture_name = f"capture_{self.image_counter:03d}_{timestamp}"
        capture_path = os.path.join(self.capture_dir, capture_name)
        os.makedirs(capture_path, exist_ok=True)

        # Save each processing step
        for step_name, image in steps.items():
            file_path = os.path.join(capture_path, f"{step_name}.png")
            cv2.imwrite(file_path, image)

        # Save metadata
        with open(os.path.join(capture_path, "metadata.txt"), "w") as f:
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write(f"Timestamp: {timestamp}\n")

        print(f"Saved capture {self.image_counter:03d} to {capture_path}")
        self.image_counter += 1

    def run(self):
        """Run real-time gesture recognition"""
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")

            # Window setup
            cv2.namedWindow("Gesture Recognition Pipeline", cv2.WINDOW_NORMAL)

            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Process frame with MediaPipe Hands
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                # Get prediction and processing steps
                pred_class, confidence, steps, status = self.predict(frame)

                # Create visualization
                pipeline_viz = self.create_visualization(
                    frame, steps, pred_class, confidence, status
                )

                # Show visualization
                cv2.imshow("Gesture Recognition Pipeline", pipeline_viz)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and steps is not None:
                    self.save_capture(steps, pred_class, confidence)

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


def main():
    """Main function to run the gesture recognizer"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Gesture Recognition")
    parser.add_argument(
        "--model",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model weights",
    )
    args = parser.parse_args()

    recognizer = RealtimeGestureRecognizer(args.model)
    recognizer.run()


if __name__ == "__main__":
    main()
