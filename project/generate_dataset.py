import cv2
import mediapipe as mp
import numpy as np
import os
import time
import argparse
from datetime import datetime
import json


class DatasetGenerator:
    def __init__(self, output_dir, gestures, samples_per_gesture=100, countdown=3):
        self.output_dir = output_dir
        self.gestures = gestures
        self.samples_per_gesture = samples_per_gesture
        self.countdown = countdown

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # Create output directory structure
        self.create_directories()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")

        # Metadata for the dataset
        self.metadata = {
            "dataset_info": {
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_gestures": len(gestures),
                "samples_per_gesture": samples_per_gesture,
                "gestures": gestures,
            },
            "samples": {},
        }

    def create_directories(self):
        """Create the directory structure for the dataset"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for gesture in self.gestures:
            gesture_dir = os.path.join(self.output_dir, gesture)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)

        # Create directory for hand landmarks
        landmarks_dir = os.path.join(self.output_dir, "landmarks")
        if not os.path.exists(landmarks_dir):
            os.makedirs(landmarks_dir)

    def countdown_timer(self):
        """Display countdown timer before capturing samples"""
        for i in range(self.countdown, 0, -1):
            ret, frame = self.cap.read()
            if not ret:
                return False

            # Add countdown text
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                str(i),
                (w // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 255, 255),
                6,
            )
            cv2.imshow("Capture", frame)

            # Wait for 1 second
            time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def capture_gesture(self, gesture, sample_idx):
        """Capture a single gesture sample"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Save frame
        frame_filename = f"{gesture}_{sample_idx}.jpg"
        frame_path = os.path.join(self.output_dir, gesture, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Save landmarks if detected
        landmarks = None
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

            landmarks_filename = f"{gesture}_{sample_idx}.json"
            landmarks_path = os.path.join(
                self.output_dir, "landmarks", landmarks_filename
            )
            with open(landmarks_path, "w") as f:
                json.dump(landmarks, f)

        return frame, landmarks

    def generate_dataset(self):
        """Generate the complete dataset"""
        try:
            for gesture in self.gestures:
                print(f"\nPreparing to capture {gesture}")
                print(f"Position your hand and press SPACE when ready")

                sample_count = 0
                while sample_count < self.samples_per_gesture:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Show current progress
                    h, w = frame.shape[:2]
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Samples: {sample_count}/{self.samples_per_gesture}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    # Draw hand landmarks if detected
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )

                    cv2.imshow("Capture", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    elif key == ord(" "):  # Space to start capturing
                        if self.countdown_timer():
                            # Capture multiple samples quickly
                            for _ in range(10):  # Capture 10 samples at once
                                if sample_count >= self.samples_per_gesture:
                                    break

                                frame, landmarks = self.capture_gesture(
                                    gesture, sample_count
                                )
                                if frame is not None:
                                    # Update metadata
                                    self.metadata["samples"][
                                        f"{gesture}_{sample_count}"
                                    ] = {
                                        "gesture": gesture,
                                        "has_landmarks": landmarks is not None,
                                    }
                                    sample_count += 1

                                # Brief pause between captures
                                time.sleep(0.1)

                print(f"Completed capturing {gesture}")

            # Save metadata
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)

        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    parser = argparse.ArgumentParser(description="Generate hand gesture dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/custom",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples per gesture"
    )
    parser.add_argument(
        "--countdown", type=int, default=3, help="Countdown timer duration"
    )
    args = parser.parse_args()

    # Define the gestures to capture
    gestures = [
        "palm",
        "l_shape",
        "fist",
        "fist_moved",
        "thumb",
        "index_finger",
        "ok_sign",
        "palm_moved",
        "c_shape",
        "down_sign",
    ]

    # Create and run the dataset generator
    generator = DatasetGenerator(
        output_dir=args.output_dir,
        gestures=gestures,
        samples_per_gesture=args.samples,
        countdown=args.countdown,
    )

    print("\nDataset Generation Instructions:")
    print("1. Position your hand in front of the camera")
    print("2. Press SPACE when ready to capture a gesture")
    print("3. Hold the gesture steady during the countdown")
    print("4. Press 'q' to quit at any time")
    print("\nStarting dataset generation...")

    generator.generate_dataset()


if __name__ == "__main__":
    main()
