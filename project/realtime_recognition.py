import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
from datetime import datetime
import mediapipe as mp
import sys
import matplotlib.pyplot as plt
from collections import deque

from model import get_model
from dataset import get_class_names


class RealtimeGestureRecognizer:
    def __init__(self, model_path, confidence_threshold=0.7):
        """
        Initialize the real-time gesture recognizer

        Args:
            model_path (str): Path to the trained model weights
            confidence_threshold (float): Confidence threshold for predictions
        """
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = get_model(num_classes=10, hidden_size=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize transform
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Get class names
        self.class_names = get_class_names()

        # Set confidence threshold
        self.confidence_threshold = confidence_threshold

        # Initialize variables for smoothing
        self.prediction_history = []
        self.history_size = 5
        self.current_gesture = None
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.5  # seconds

    def get_hand_roi(self, frame):
        """Extract hand region from frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, None, None

        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get bounding box
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None, None, None

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        return gray_roi, (x_min, y_min, x_max, y_max), results.multi_hand_landmarks[0]

    def process_frame(self, frame):
        """Process a single frame and return the predicted gesture"""
        # Get hand ROI
        roi, bbox, landmarks = self.get_hand_roi(frame)
        if roi is None:
            return None, 0.0, None, None, None

        # Convert frame to tensor
        frame_tensor = self.transform(roi).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(frame_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Get prediction details
        pred_idx = predicted.item()
        conf = confidence.item()
        gesture_name = self.class_names[f"{pred_idx+1:02d}"]

        return gesture_name, conf, roi, bbox, landmarks

    def get_most_common_gesture(self):
        """Get the most common gesture from prediction history"""
        if not self.prediction_history:
            return None, 0

        # Count occurrences of each gesture
        gesture_counts = {}
        for gesture, conf in self.prediction_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
            gesture_counts[gesture] += 1

        # Find most common gesture
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        return most_common[0], most_common[1] / len(self.prediction_history)

    def run(self):
        """Run realtime gesture recognition"""
        print("Starting realtime gesture recognition...")
        print("Press 'q' to quit")

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            gesture, confidence, roi, bbox, landmarks = self.process_frame(frame)

            # Draw hand landmarks on camera feed
            if landmarks:
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )

            if gesture and confidence >= self.confidence_threshold:
                # Add to prediction history
                self.prediction_history.append((gesture, confidence))
                if len(self.prediction_history) > self.history_size:
                    self.prediction_history.pop(0)

                # Get smoothed prediction
                current_time = time.time()
                if current_time - self.last_prediction_time >= self.prediction_cooldown:
                    smoothed_gesture, gesture_confidence = (
                        self.get_most_common_gesture()
                    )

                    if (
                        smoothed_gesture and gesture_confidence >= 0.6
                    ):  # At least 60% of predictions agree
                        if smoothed_gesture != self.current_gesture:
                            self.current_gesture = smoothed_gesture
                            print(
                                f"\nDetected gesture: {smoothed_gesture} (Confidence: {confidence:.2%})"
                            )
                        self.last_prediction_time = current_time

                # Draw bounding box and gesture name on camera feed
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{gesture} ({confidence:.2%})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                # Display the grayscale hand image
                if roi is not None:
                    # Resize to match dataset format (64x64)
                    roi = cv2.resize(roi, (64, 64))

                    # Create display window
                    display = np.zeros((64, 64), dtype=np.uint8)
                    display = roi  # Use the grayscale ROI directly

                    # Show the display
                    cv2.imshow("Processed Hand Image", display)

            # Show camera feed
            cv2.imshow("Camera Feed", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the gesture recognizer"""
    import argparse

    parser = argparse.ArgumentParser(description="Realtime hand gesture recognition")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for predictions",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    # Initialize and run recognizer
    recognizer = RealtimeGestureRecognizer(
        args.model_path, confidence_threshold=args.confidence_threshold
    )
    recognizer.run()


if __name__ == "__main__":
    main()
