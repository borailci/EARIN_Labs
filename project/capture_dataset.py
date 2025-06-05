import cv2
import os
import numpy as np
import mediapipe as mp
import time
from datetime import datetime


class ImprovedDatasetCapture:
    def __init__(self, output_dir="custom_data"):
        """
        Initialize the improved dataset capture tool

        Args:
            output_dir (str): Directory to save captured images
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Set output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Configuration
        self.frames_per_session = 50  # Capture 50 frames per button press
        self.target_frames_per_gesture = 500  # Target 500 frames per gesture
        self.sessions_needed = (
            self.target_frames_per_gesture // self.frames_per_session
        )  # 10 sessions
        self.capture_delay = 0.1  # seconds between captures

        # State variables
        self.current_gesture = None
        self.is_capturing = False
        self.session_frame_count = 0
        self.last_capture_time = time.time()

        # Gesture definitions
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
            0: "down",  # Map 0 key to "down" gesture (gesture_id 10)
        }

        # Main folders (00-09)
        self.main_folders = list(range(10))

        # Create directory structure
        self.create_directories()

        # Load current statistics
        self.gesture_stats = self.load_current_stats()

    def create_directories(self):
        """Create the directory structure for all gestures"""
        for main_id in self.main_folders:
            main_dir = os.path.join(self.output_dir, f"{main_id:02d}")
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)

            for gesture_key, gesture_name in self.gestures.items():
                gesture_id = 10 if gesture_key == 0 else gesture_key  # Map 0->10
                gesture_dir = os.path.join(main_dir, f"{gesture_id:02d}_{gesture_name}")
                if not os.path.exists(gesture_dir):
                    os.makedirs(gesture_dir)

    def load_current_stats(self):
        """Load current statistics for each gesture"""
        stats = {}
        for gesture_key, gesture_name in self.gestures.items():
            gesture_id = 10 if gesture_key == 0 else gesture_key
            total_frames = 0

            # Count existing frames across all main folders
            for main_id in self.main_folders:
                gesture_dir = os.path.join(
                    self.output_dir,
                    f"{main_id:02d}",
                    f"{gesture_id:02d}_{gesture_name}",
                )
                if os.path.exists(gesture_dir):
                    existing_files = [
                        f
                        for f in os.listdir(gesture_dir)
                        if f.startswith("frame_") and f.endswith(".png")
                    ]
                    total_frames += len(existing_files)

            sessions_completed = total_frames // (
                self.frames_per_session * len(self.main_folders)
            )

            stats[gesture_key] = {
                "name": gesture_name,
                "total_frames": total_frames,
                "sessions_completed": sessions_completed,
                "sessions_remaining": max(0, self.sessions_needed - sessions_completed),
            }

        return stats

    def get_next_frame_number(self, main_id, gesture_id):
        """Get the next available frame number for a specific gesture"""
        gesture_name = next(
            name
            for key, name in self.gestures.items()
            if (key if key != 0 else 10) == gesture_id
        )

        gesture_dir = os.path.join(
            self.output_dir, f"{main_id:02d}", f"{gesture_id:02d}_{gesture_name}"
        )

        if not os.path.exists(gesture_dir):
            return 1

        existing_files = [
            f
            for f in os.listdir(gesture_dir)
            if f.startswith(f"frame_{main_id:02d}_{gesture_id:02d}_")
        ]

        if not existing_files:
            return 1

        # Extract frame numbers and find the maximum
        frame_numbers = []
        for filename in existing_files:
            try:
                parts = filename.split("_")
                if len(parts) >= 4:
                    frame_num = int(parts[3].split(".")[0])
                    frame_numbers.append(frame_num)
            except (ValueError, IndexError):
                continue

        return max(frame_numbers) + 1 if frame_numbers else 1

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
        padding = 30
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

    def save_image(self, roi, gesture_key):
        """Save the captured image to all main folders"""
        if roi is None:
            return False

        # Resize to 64x64
        roi = cv2.resize(roi, (64, 64))

        gesture_id = 10 if gesture_key == 0 else gesture_key
        gesture_name = self.gestures[gesture_key]

        saved_count = 0

        # Save to all main folders
        for main_id in self.main_folders:
            # Get next frame number
            frame_num = self.get_next_frame_number(main_id, gesture_id)

            # Create filepath
            gesture_dir = os.path.join(
                self.output_dir, f"{main_id:02d}", f"{gesture_id:02d}_{gesture_name}"
            )

            filename = f"frame_{main_id:02d}_{gesture_id:02d}_{frame_num:04d}.png"
            filepath = os.path.join(gesture_dir, filename)

            # Save image
            cv2.imwrite(filepath, roi)
            saved_count += 1

        return saved_count == len(self.main_folders)

    def draw_ui(self, frame):
        """Draw the user interface on the frame"""
        # Background for UI
        overlay = frame.copy()

        # Draw main UI background
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame,
            "Hand Gesture Data Capture",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Instructions
        y_pos = 60
        cv2.putText(
            frame,
            "Press 1-9,0 to select gesture, SPACE to capture 50 frames",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Current gesture info
        if self.current_gesture is not None:
            gesture_name = self.gestures[self.current_gesture]
            stats = self.gesture_stats[self.current_gesture]

            y_pos += 25
            cv2.putText(
                frame,
                f"Current Gesture: {gesture_name.upper()}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            y_pos += 20
            cv2.putText(
                frame,
                f"Sessions: {stats['sessions_completed']}/{self.sessions_needed}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            y_pos += 15
            cv2.putText(
                frame,
                f"Total frames: {stats['total_frames']}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            # Capture progress
            if self.is_capturing:
                y_pos += 20
                progress = (
                    f"Capturing: {self.session_frame_count}/{self.frames_per_session}"
                )
                cv2.putText(
                    frame,
                    progress,
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Progress bar
                bar_width = 300
                bar_height = 10
                progress_ratio = self.session_frame_count / self.frames_per_session
                cv2.rectangle(
                    frame,
                    (20, y_pos + 10),
                    (20 + bar_width, y_pos + 10 + bar_height),
                    (100, 100, 100),
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (20, y_pos + 10),
                    (20 + int(bar_width * progress_ratio), y_pos + 10 + bar_height),
                    (0, 255, 0),
                    -1,
                )

        # Gesture list
        x_start = 520
        y_start = 30
        cv2.putText(
            frame,
            "Gestures:",
            (x_start, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        for i, (key, name) in enumerate(self.gestures.items()):
            y_pos = y_start + 25 + (i * 20)
            stats = self.gesture_stats[key]

            # Color coding based on completion
            if stats["sessions_completed"] >= self.sessions_needed:
                color = (0, 255, 0)  # Green - complete
                status = "COMPLETE"
            elif stats["sessions_completed"] > 0:
                color = (0, 255, 255)  # Yellow - in progress
                status = f"{stats['sessions_completed']}/{self.sessions_needed}"
            else:
                color = (100, 100, 100)  # Gray - not started
                status = "NOT STARTED"

            text = f"{key}: {name} ({status})"
            cv2.putText(
                frame, text, (x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    def run(self):
        """Run the improved dataset capture tool"""
        print("🚀 Starting Improved Hand Gesture Data Capture")
        print("=" * 60)
        print("📋 Configuration:")
        print(f"  • Frames per session: {self.frames_per_session}")
        print(f"  • Target frames per gesture: {self.target_frames_per_gesture}")
        print(f"  • Sessions needed per gesture: {self.sessions_needed}")
        print(f"  • Main folders: {len(self.main_folders)} (00-09)")
        print("\n📖 Instructions:")
        print("  • Press 1-9,0 to select gesture")
        print("  • Press SPACE to start capturing 50 frames")
        print("  • Press 'q' to quit")
        print("  • Keep your hand visible for better detection")
        print("=" * 60)

        # Print current statistics
        print("\n📊 Current Progress:")
        for key, stats in self.gesture_stats.items():
            name = stats["name"]
            completed = stats["sessions_completed"]
            remaining = stats["sessions_remaining"]
            print(
                f"  {key}: {name:<12} - {completed}/{self.sessions_needed} sessions "
                f"({stats['total_frames']} frames)"
            )
        print()

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        print("✅ Webcam initialized. Starting capture interface...")

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
                    # Different colors based on capture state
                    color = (0, 255, 255) if self.is_capturing else (0, 255, 0)
                    thickness = 3 if self.is_capturing else 2
                    cv2.rectangle(
                        frame, (x_min, y_min), (x_max, y_max), color, thickness
                    )

            # Draw UI
            self.draw_ui(frame)

            # Show main camera feed
            cv2.imshow("Hand Gesture Capture", frame)

            # Show processed hand image
            if roi is not None:
                display_roi = cv2.resize(roi, (128, 128))
                cv2.imshow("Hand ROI", display_roi)

            # Auto-capture frames
            if (
                self.is_capturing
                and self.current_gesture is not None
                and roi is not None
            ):
                current_time = time.time()
                if current_time - self.last_capture_time >= self.capture_delay:
                    if self.save_image(roi, self.current_gesture):
                        self.session_frame_count += 1
                        print(
                            f"\rCapturing: {self.session_frame_count}/{self.frames_per_session}",
                            end="",
                            flush=True,
                        )

                    self.last_capture_time = current_time

                    # Check if session is complete
                    if self.session_frame_count >= self.frames_per_session:
                        self.is_capturing = False
                        self.session_frame_count = 0

                        # Update statistics
                        self.gesture_stats = self.load_current_stats()
                        stats = self.gesture_stats[self.current_gesture]

                        print(
                            f"\n✅ Session complete! "
                            f"Progress: {stats['sessions_completed']}/{self.sessions_needed} sessions"
                        )

                        if stats["sessions_completed"] >= self.sessions_needed:
                            print(f"🎉 Gesture '{stats['name']}' is COMPLETE!")

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in [ord(str(i)) for i in range(10)]:
                # Select gesture
                gesture_key = int(chr(key))
                self.current_gesture = gesture_key
                self.is_capturing = False
                self.session_frame_count = 0

                gesture_name = self.gestures[gesture_key]
                stats = self.gesture_stats[gesture_key]
                print(
                    f"\n📌 Selected: {gesture_name} "
                    f"({stats['sessions_completed']}/{self.sessions_needed} sessions)"
                )

            elif key == ord(" "):  # Space bar
                if self.current_gesture is None:
                    print("\n⚠️  Please select a gesture first (1-9,0)")
                elif roi is None:
                    print("\n⚠️  No hand detected. Please show your hand to the camera.")
                elif self.is_capturing:
                    print("\n⚠️  Already capturing frames...")
                else:
                    stats = self.gesture_stats[self.current_gesture]
                    if stats["sessions_completed"] >= self.sessions_needed:
                        print(f"\n✅ Gesture '{stats['name']}' is already complete!")
                    else:
                        self.is_capturing = True
                        self.session_frame_count = 0
                        print(
                            f"\n🎬 Starting capture session "
                            f"{stats['sessions_completed'] + 1}/{self.sessions_needed}..."
                        )

        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 60)
        print("📊 Final Statistics:")
        total_captured = sum(
            stats["total_frames"] for stats in self.gesture_stats.values()
        )
        print(f"Total frames captured: {total_captured}")

        for key, stats in self.gesture_stats.items():
            status = (
                "✅ COMPLETE"
                if stats["sessions_completed"] >= self.sessions_needed
                else "🔄 IN PROGRESS"
            )
            print(
                f"  {stats['name']:<12}: {stats['total_frames']} frames, "
                f"{stats['sessions_completed']}/{self.sessions_needed} sessions {status}"
            )

        print(f"\n💾 Data saved in: {os.path.abspath(self.output_dir)}")
        print("🎉 Capture session ended!")


def main():
    """Main function"""
    capture = ImprovedDatasetCapture()
    capture.run()


if __name__ == "__main__":
    main()
