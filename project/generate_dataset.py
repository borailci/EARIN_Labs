"""
Dataset Generation Module for Hand Gesture Recognition

This module provides functionality for capturing and organizing hand gesture data
using MediaPipe hand detection and OpenCV camera interface. The system supports
interactive dataset generation with real-time hand landmark detection and
proper gesture labeling for training deep learning models.

Key Features:
- Interactive gesture data capture using webcam
- MediaPipe integration for robust hand detection
- Automatic image preprocessing and normalization
- Organized dataset structure compatible with training pipeline
- Support for multiple gesture classes with customizable sample counts
- Real-time feedback during data collection process

Classes:
    DatasetGenerator: Main class for interactive dataset collection

Functions:
    create_user_dataset: Create personalized gesture dataset for specific users

Usage:
    python generate_dataset.py --output_dir ./custom_data --user_name user1

Author: Course Project Team
Date: Academic Year 2024
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import argparse
from datetime import datetime
import json


class DatasetGenerator:
    def __init__(self, output_dir, gestures, samples_per_gesture=100, countdown=0):
        self.output_dir = output_dir
        self.gestures = gestures
        self.samples_per_gesture = samples_per_gesture
        self.countdown = countdown

        # Gesture mapping to match existing dataset structure
        self.gesture_mapping = {
            "palm": "01_palm",
            "l_shape": "02_l",
            "fist": "03_fist",
            "fist_moved": "04_fist_moved",
            "thumb": "05_thumb",
            "index_finger": "06_index",
            "ok_sign": "07_ok",
            "palm_moved": "08_palm_moved",
            "c_shape": "09_c",
            "down_sign": "10_down",
        }

        # Find next available user ID
        self.user_id = self.find_next_user_id()
        print(f"Using user ID: {self.user_id:02d}")

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
                "user_id": self.user_id,
                "total_gestures": len(gestures),
                "samples_per_gesture": samples_per_gesture,
                "gestures": gestures,
            },
            "samples": {},
        }

    def find_next_user_id(self):
        """Find the next available user ID to avoid conflicts"""
        if not os.path.exists(self.output_dir):
            return 0

        existing_ids = []
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 2:
                existing_ids.append(int(item))

        if not existing_ids:
            return 0
        return max(existing_ids) + 1

    def create_directories(self):
        """Create the directory structure for the dataset"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create user directory
        user_dir = os.path.join(self.output_dir, f"{self.user_id:02d}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Create gesture subdirectories
        for gesture in self.gestures:
            if gesture in self.gesture_mapping:
                gesture_dir = os.path.join(user_dir, self.gesture_mapping[gesture])
                if not os.path.exists(gesture_dir):
                    os.makedirs(gesture_dir)

    def countdown_timer(self):
        """Display countdown timer before capturing samples (now skipped)"""
        # Skip countdown - capture immediately
        return True

    def crop_hand_region(self, frame, hand_landmarks):
        """Crop and process hand region to 64x64 grayscale"""
        h, w = frame.shape[:2]

        # Get hand bounding box from landmarks
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding around hand
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            return None

        # Convert to grayscale
        if len(hand_crop.shape) == 3:
            hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        else:
            hand_gray = hand_crop

        # Resize to 64x64
        hand_resized = cv2.resize(hand_gray, (64, 64))

        return hand_resized

    def capture_gesture(self, gesture, sample_idx):
        """Capture a single gesture sample"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Get gesture mapping and create filename
        if gesture not in self.gesture_mapping:
            print(f"Warning: Gesture '{gesture}' not found in mapping")
            return None, None

        gesture_code = self.gesture_mapping[gesture]
        gesture_num = gesture_code.split("_")[
            0
        ]  # Extract number part (e.g., "01" from "01_palm")

        # Create filename in format: frame_{user_id}_{gesture_id}_{frame_number}.png
        frame_filename = f"frame_{self.user_id:02d}_{gesture_num}_{sample_idx:04d}.png"

        # Save to appropriate directory structure
        user_dir = os.path.join(self.output_dir, f"{self.user_id:02d}")
        gesture_dir = os.path.join(user_dir, gesture_code)
        frame_path = os.path.join(gesture_dir, frame_filename)

        # Process hand region if detected
        hand_processed = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_processed = self.crop_hand_region(frame, hand_landmarks)
                break  # Use first detected hand

        # Save the processed hand image (64x64 grayscale) instead of full frame
        if hand_processed is not None:
            cv2.imwrite(frame_path, hand_processed)
        else:
            # Fallback: save resized grayscale of center crop if no hand detected
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray_frame.shape
            center_crop = gray_frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            fallback_processed = cv2.resize(center_crop, (64, 64))
            cv2.imwrite(frame_path, fallback_processed)
            hand_processed = fallback_processed

        # Extract landmarks if detected
        landmarks = None
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

        return hand_processed, landmarks

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

                                hand_processed, landmarks = self.capture_gesture(
                                    gesture, sample_count
                                )
                                if hand_processed is not None:
                                    # Show preview of captured processed image
                                    preview = cv2.resize(
                                        hand_processed, (128, 128)
                                    )  # Scale up for visibility
                                    cv2.imshow(
                                        "Last Captured (64x64 Grayscale)", preview
                                    )

                                    # Update metadata
                                    gesture_code = self.gesture_mapping[gesture]
                                    gesture_num = gesture_code.split("_")[0]
                                    frame_id = f"frame_{self.user_id:02d}_{gesture_num}_{sample_count:04d}"

                                    self.metadata["samples"][frame_id] = {
                                        "gesture": gesture,
                                        "gesture_code": gesture_code,
                                        "has_landmarks": landmarks is not None,
                                        "user_id": self.user_id,
                                    }
                                    sample_count += 1

                                # Brief pause between captures
                                time.sleep(0.1)

                print(f"Completed capturing {gesture}")

            # Save metadata
            user_dir = os.path.join(self.output_dir, f"{self.user_id:02d}")
            metadata_path = os.path.join(user_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)

            print(f"\nDataset generation completed!")
            print(f"Files saved in: {user_dir}")
            print(f"Total samples captured: {len(self.metadata['samples'])}")

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
        default="custom_data",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="Number of samples per gesture"
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=0,
        help="Countdown timer duration (0 = disabled)",
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
    print("3. Images will be captured immediately (no countdown)")
    print("4. Preview window shows processed 64x64 grayscale hand crop")
    print("5. Press 'q' to quit at any time")
    print(f"\nFiles will be saved to: custom_data/{generator.user_id:02d}/")
    print("This will NOT override existing data!")
    print("\nStarting dataset generation...")

    generator.generate_dataset()


if __name__ == "__main__":
    main()
