import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
from datetime import datetime
import queue
from threading import Thread

from model import get_model
from dataset import get_class_names


class FrameProcessor(Thread):
    """Thread class for processing frames in background"""

    def __init__(self, frame_queue, result_queue, model, transform, device):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.transform = transform
        self.device = device
        self.running = True

    def run(self):
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue

                frame, timestamp = frame_data

                # Convert numpy array to PIL Image (frame is already grayscale)
                pil_image = Image.fromarray(frame)
                tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                # Get prediction
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                self.result_queue.put((predicted.item(), confidence.item(), timestamp))

            except queue.Empty:
                continue

    def stop(self):
        self.running = False


class RealtimeGestureRecognizer:
    def __init__(self, model_path, device=None, save_frames=True):
        """
        Initialize the real-time gesture recognizer

        Args:
            model_path (str): Path to the trained model weights
            device (torch.device): Device to run inference on
            save_frames (bool): Whether to save processed frames
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

        # Frame saving settings
        self.save_frames = save_frames
        self.capture_dir = "captured_frames"
        if save_frames:
            os.makedirs(self.capture_dir, exist_ok=True)

        # Processing queues
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size
        self.result_queue = queue.Queue()

        # Frame processor thread
        self.frame_processor = None

        # Frame processing parameters
        self.roi_size = 600  # Increased ROI size from 400 to 600
        self.min_brightness = 30  # Minimum average brightness for valid frame
        self.max_brightness = 225  # Maximum average brightness for valid frame

        # Frame buffer for gesture sequence
        self.frame_buffer = []
        self.buffer_size = 30  # Number of frames to keep in buffer
        self.min_frames_for_prediction = 10  # Minimum frames needed for a prediction

    def get_roi(self, frame):
        """Extract and validate the region of interest from the frame"""
        if frame is None:
            return None, "No frame captured"

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get frame center
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2

        # Calculate ROI boundaries
        half_size = self.roi_size // 2
        top = max(0, center_y - half_size)
        bottom = min(h, center_y + half_size)
        left = max(0, center_x - half_size)
        right = min(w, center_x + half_size)

        # Extract ROI
        roi = gray[top:bottom, left:right]

        # Validate ROI
        if roi.size == 0:
            return None, "Invalid ROI size"

        # Check brightness
        avg_brightness = np.mean(roi)
        if avg_brightness < self.min_brightness:
            return None, "Image too dark"
        if avg_brightness > self.max_brightness:
            return None, "Image too bright"

        return roi, "OK"

    def save_frame(self, frame, prediction, confidence, timestamp):
        """Save a frame with its metadata"""
        if not self.save_frames:
            return

        # Create timestamp-based directory
        frame_dir = os.path.join(
            self.capture_dir, f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        )
        os.makedirs(frame_dir, exist_ok=True)

        # Save original frame
        cv2.imwrite(os.path.join(frame_dir, "original.png"), frame)

        # Save ROI
        roi, _ = self.get_roi(frame)
        if roi is not None:
            cv2.imwrite(os.path.join(frame_dir, "roi.png"), roi)

        # Save metadata
        with open(os.path.join(frame_dir, "metadata.txt"), "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Confidence: {confidence:.4f}\n")

    def process_frame_buffer(self):
        """Process the frame buffer to make a prediction"""
        if len(self.frame_buffer) < self.min_frames_for_prediction:
            return None, 0.0

        # Get most common prediction and average confidence
        predictions = [pred for _, pred, _ in self.frame_buffer]
        confidences = [conf for _, _, conf in self.frame_buffer]

        unique_preds, counts = np.unique(predictions, return_counts=True)
        most_common_pred = unique_preds[np.argmax(counts)]
        avg_confidence = np.mean(
            [
                conf
                for pred, conf in zip(predictions, confidences)
                if pred == most_common_pred
            ]
        )

        return most_common_pred, avg_confidence

    def run(self):
        """Run real-time gesture recognition"""
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")

            # Start frame processor thread
            self.frame_processor = FrameProcessor(
                self.frame_queue,
                self.result_queue,
                self.model,
                self.transform,
                self.device,
            )
            self.frame_processor.start()

            # Window setup
            cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

            last_prediction = None
            last_confidence = 0.0
            prediction_start_time = None
            stable_frames = 0
            required_stable_frames = 10

            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert frame to grayscale for display
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert back to BGR for drawing colored elements
                display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

                # Get ROI
                roi, status = self.get_roi(frame)

                if roi is not None:
                    # Add frame to processing queue
                    try:
                        timestamp = datetime.now()
                        self.frame_queue.put((roi, timestamp), block=False)
                    except queue.Full:
                        pass  # Skip frame if queue is full

                # Check for processed results
                try:
                    while not self.result_queue.empty():
                        pred_class, confidence, timestamp = (
                            self.result_queue.get_nowait()
                        )

                        # Update frame buffer
                        self.frame_buffer.append((timestamp, pred_class, confidence))
                        if len(self.frame_buffer) > self.buffer_size:
                            self.frame_buffer.pop(0)

                        # Process buffer for stable prediction
                        current_pred, avg_confidence = self.process_frame_buffer()

                        if current_pred is not None:
                            if current_pred == last_prediction:
                                stable_frames += 1
                                if stable_frames >= required_stable_frames:
                                    if prediction_start_time is None:
                                        prediction_start_time = time.time()
                                        # Save frame when prediction becomes stable
                                        self.save_frame(
                                            frame,
                                            self.class_names[current_pred],
                                            avg_confidence,
                                            timestamp,
                                        )
                            else:
                                stable_frames = 0
                                prediction_start_time = None

                            last_prediction = current_pred
                            last_confidence = avg_confidence

                except queue.Empty:
                    pass

                # Draw ROI guide
                h, w = display_frame.shape[:2]
                center_y, center_x = h // 2, w // 2
                half_size = self.roi_size // 2
                cv2.rectangle(
                    display_frame,
                    (center_x - half_size, center_y - half_size),
                    (center_x + half_size, center_y + half_size),
                    (0, 255, 0),
                    2,
                )

                # Display prediction
                if last_prediction is not None and prediction_start_time is not None:
                    text = (
                        f"{self.class_names[last_prediction]} ({last_confidence:.2f})"
                    )
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Display hold time
                    hold_time = time.time() - prediction_start_time
                    cv2.putText(
                        display_frame,
                        f"Hold time: {hold_time:.1f}s",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        "Waiting for stable gesture...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                    )

                # Show frame
                cv2.imshow("Gesture Recognition", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and roi is not None:
                    # Force save current frame
                    self.save_frame(
                        frame,
                        (
                            self.class_names[last_prediction]
                            if last_prediction is not None
                            else "unknown"
                        ),
                        last_confidence,
                        datetime.now(),
                    )

        finally:
            # Cleanup
            if self.frame_processor is not None:
                self.frame_processor.stop()
                self.frame_processor.join()

            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


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
    parser.add_argument("--no-save", action="store_true", help="Disable frame saving")
    args = parser.parse_args()

    recognizer = RealtimeGestureRecognizer(args.model, save_frames=not args.no_save)
    recognizer.run()


if __name__ == "__main__":
    main()
