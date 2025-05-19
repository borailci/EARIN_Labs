import cv2
import os
import numpy as np
import mediapipe as mp
import time
from datetime import datetime


class DatasetCapture:
    def __init__(self, output_dir="custom_data"):
        """
        Initialize the dataset capture tool

        Args:
            output_dir (str): Directory to save captured images
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Set output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize variables
        self.current_gesture = None
        self.capture_count = 0
        self.capture_delay = 0.1  # seconds between captures (faster for auto-capture)
        self.last_capture_time = time.time()
        self.frames_per_sequence = 100
        self.sequences_per_gesture = 10
        self.current_sequence_count = 0
        self.current_sequence = 0
        self.is_capturing = False

        # Create gesture directories with exact same structure
        self.gestures = {
            1: "palm",
            2: "l",
            3: "fist",
            4: "fist_moved",
            5: "thumb",
            6: "index",
            7: "ok",
            8: "palm_moved",
            9: "c",
            10: "down",
        }

        # Load reference images
        self.reference_images = {}
        reference_dir = "reference_images"
        if not os.path.exists(reference_dir):
            os.makedirs(reference_dir)
            print(f"\nPlease add reference images in the '{reference_dir}' directory:")
            for gesture_id, gesture_name in self.gestures.items():
                print(f"  {gesture_id:02d}_{gesture_name}.jpg")
        else:
            for gesture_id, gesture_name in self.gestures.items():
                img_path = os.path.join(
                    reference_dir, f"{gesture_id:02d}_{gesture_name}.jpg"
                )
                if os.path.exists(img_path):
                    self.reference_images[gesture_id] = cv2.imread(img_path)
                else:
                    print(
                        f"Warning: Reference image missing for {gesture_name}: {img_path}"
                    )

        # Create main gesture folders (00-09)
        for i in range(10):
            main_dir = os.path.join(output_dir, f"{i:02d}")
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)

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

        return gray_roi, (x_min, y_min, x_max, y_max), hand_landmarks

    def save_image(self, roi, main_id, gesture_id):
        """Save the captured image"""
        if roi is None:
            return False

        # Resize to 64x64
        roi = cv2.resize(roi, (64, 64))

        # Create gesture folder
        gesture_dir = os.path.join(
            self.output_dir,
            f"{main_id:02d}",
            f"{gesture_id:02d}_{self.gestures[gesture_id]}",
        )
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)

        # Generate filename in the exact format: frame_XX_YY_ZZZZ.png
        # XX: main folder number (00-09)
        # YY: gesture number (01-10)
        # ZZZZ: frame number (0001-0100)
        filename = f"frame_{main_id:02d}_{gesture_id:02d}_{self.current_sequence_count + 1:04d}.png"
        filepath = os.path.join(gesture_dir, filename)
        cv2.imwrite(filepath, roi)

        return True

    def run(self):
        """Run the dataset capture tool"""
        print("Starting dataset capture...")
        print("Press number keys 1-0 to select gesture:")
        for key, gesture in self.gestures.items():
            print(f"  {key}: {gesture}")
        print("Press 'q' to quit")
        print("Press 'c' to start capturing 100 frames")
        print(
            f"Each gesture needs {self.sequences_per_gesture} sequences of {self.frames_per_sequence} frames"
        )

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

            # Get hand ROI
            roi, bbox, landmarks = self.get_hand_roi(frame)

            # Draw hand landmarks and bounding box
            if landmarks:
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Show current gesture and reference image
            if self.current_gesture is not None:
                gesture_name = self.gestures[self.current_gesture]
                cv2.putText(
                    frame,
                    f"Current: {gesture_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Show sequence progress
                progress = f"Sequence: {self.current_sequence + 1}/{self.sequences_per_gesture}"
                cv2.putText(
                    frame,
                    progress,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Show frame progress
                if self.is_capturing:
                    frame_progress = f"Frames: {self.current_sequence_count}/{self.frames_per_sequence}"
                    cv2.putText(
                        frame,
                        frame_progress,
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                # Show reference image if available
                if self.current_gesture in self.reference_images:
                    ref_img = self.reference_images[self.current_gesture]
                    ref_img = cv2.resize(ref_img, (200, 200))
                    h, w = ref_img.shape[:2]
                    frame[
                        10 : 10 + h, frame.shape[1] - w - 10 : frame.shape[1] - 10
                    ] = ref_img

            # Show capture count
            cv2.putText(
                frame,
                f"Total Captured: {self.capture_count}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show camera feed
            cv2.imshow("Camera Feed", frame)

            # Show processed image
            if roi is not None:
                roi = cv2.resize(roi, (64, 64))
                cv2.imshow("Processed Hand Image", roi)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in [ord(str(i)) for i in range(10)]:
                self.current_gesture = int(chr(key))
                if self.current_gesture == 0:
                    self.current_gesture = 10  # Map 0 to 10 for "down" gesture
                self.current_sequence = 0  # Reset sequence counter
                self.current_sequence_count = 0
                self.is_capturing = False
                print(f"\nSelected gesture: {self.gestures[self.current_gesture]}")
                print(
                    f"Starting new gesture (0/{self.sequences_per_gesture} sequences)"
                )
            elif key == ord("c"):
                if self.current_gesture is None:
                    print("\nPlease select a gesture first (1-0)")
                elif roi is None:
                    print("\nNo hand detected")
                elif self.current_sequence >= self.sequences_per_gesture:
                    print(f"\nGesture complete! Select a new gesture to start another.")
                elif self.is_capturing:
                    print("\nAlready capturing frames...")
                else:
                    self.is_capturing = True
                    print(
                        f"\nStarting sequence {self.current_sequence + 1}/{self.sequences_per_gesture}"
                    )

            # Auto-capture frames
            if (
                self.is_capturing
                and self.current_gesture is not None
                and roi is not None
            ):
                current_time = time.time()
                if current_time - self.last_capture_time >= self.capture_delay:
                    # Save in all main folders (00-09)
                    for main_id in range(10):
                        if self.save_image(roi, main_id, self.current_gesture):
                            self.capture_count += 1
                            self.current_sequence_count += 1
                            print(
                                f"\rCapturing frame {self.current_sequence_count}/{self.frames_per_sequence} for sequence {self.current_sequence + 1}/{self.sequences_per_gesture}",
                                end="",
                            )
                    self.last_capture_time = current_time

                    if self.current_sequence_count >= self.frames_per_sequence:
                        self.current_sequence += 1
                        self.current_sequence_count = 0
                        self.is_capturing = False
                        print(
                            f"\nSequence {self.current_sequence}/{self.sequences_per_gesture} complete!"
                        )

                        if self.current_sequence >= self.sequences_per_gesture:
                            print(
                                f"\nGesture {self.gestures[self.current_gesture]} complete! Select a new gesture to start another."
                            )
                        else:
                            print(
                                f"\nPress 'c' to start sequence {self.current_sequence + 1}/{self.sequences_per_gesture}"
                            )

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCaptured {self.capture_count} images")
        print(f"Images saved in: {os.path.abspath(self.output_dir)}")


def main():
    """Main function to run the dataset capture tool"""
    import argparse

    parser = argparse.ArgumentParser(description="Capture hand gesture dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom_data",
        help="Directory to save captured images",
    )

    args = parser.parse_args()

    # Initialize and run capture tool
    capture_tool = DatasetCapture(output_dir=args.output_dir)
    capture_tool.run()


if __name__ == "__main__":
    main()
