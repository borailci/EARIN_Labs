#!/usr/bin/env python3
"""
Hand Gesture Recognition Model Testing Application
A simple matplotlib-based application for testing trained models without GUI.
Shows 10 images at a time with predictions and true classes.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import random
from datetime import datetime

# Import project modules
try:
    from model import get_model
    from dataset import get_data_loaders
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    sys.exit(1)


class ModelTester:
    """Simple model tester that shows 10 images at a time"""

    def __init__(self, model_path, data_dir="data", custom_data_dir="custom_data"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.custom_data_dir = custom_data_dir
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Class names
        self.class_names = [
            "palm",
            "l",
            "fist",
            "fist_moved",
            "thumb",
            "index",
            "ok",
            "palm_moved",
            "c",
            "down",
        ]

        # Load model and data
        self.model = None
        self.test_data = None
        self.current_batch_idx = 0
        self.all_images = []
        self.all_labels = []

        # Initialize
        self.load_model()
        self.load_test_data()

        # Set up the plot
        self.fig = None
        self.axes = None
        self.setup_plot()

    def load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = get_model(hidden_size=128)

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            self.model.to(self.device)
            print(f"✅ Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def load_test_data(self):
        """Load test dataset from both main data and custom data directories"""
        try:
            print(f"Loading test data from: {self.data_dir}")
            if self.custom_data_dir and os.path.exists(self.custom_data_dir):
                print(f"Also loading custom test data from: {self.custom_data_dir}")

            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

            data_loaders = get_data_loaders(
                data_dir=self.data_dir,
                custom_data_dir=(
                    self.custom_data_dir
                    if os.path.exists(self.custom_data_dir)
                    else None
                ),
                batch_size=1,  # Load one by one to have full control
                num_workers=0,
            )
            test_loader = data_loaders["test"]

            # Load all test data into memory
            print("Loading all test images...")
            with torch.no_grad():
                for images, labels in test_loader:
                    # Convert tensor to numpy for matplotlib display
                    img = (
                        images[0].squeeze().cpu().numpy()
                    )  # Remove batch and channel dims
                    label = labels[0].item()

                    self.all_images.append(img)
                    self.all_labels.append(label)

            print(f"✅ Loaded {len(self.all_images)} test images")

            # Display data source information
            main_data_count = 0
            custom_data_count = 0

            # This is a simple estimation based on typical dataset splits
            if self.custom_data_dir and os.path.exists(self.custom_data_dir):
                # Rough estimation: if we have both datasets, roughly half might be from each
                print(
                    f"📊 Test data includes images from both main dataset and custom dataset"
                )
            else:
                print(f"📊 Test data from main dataset only")

        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            sys.exit(1)

    def predict_batch(self, start_idx):
        """Get predictions for 10 images starting from start_idx"""
        end_idx = min(start_idx + 10, len(self.all_images))

        predictions = []
        confidences = []

        with torch.no_grad():
            for i in range(start_idx, end_idx):
                # Prepare image for model
                img = torch.tensor(self.all_images[i]).unsqueeze(0).unsqueeze(0).float()
                img = img.to(self.device)

                # Get prediction
                output = self.model(img)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predictions.append(predicted.item())
                confidences.append(confidence.item())

        return predictions, confidences, end_idx - start_idx

    def setup_plot(self):
        """Set up the matplotlib figure with subplots"""
        plt.style.use("dark_background")
        self.fig, self.axes = plt.subplots(2, 5, figsize=(15, 8))
        self.fig.suptitle(
            "Hand Gesture Recognition - Model Testing",
            fontsize=16,
            fontweight="bold",
            color="white",
        )

        # Flatten axes for easier indexing
        self.axes = self.axes.flatten()

        # Remove axis ticks and labels for all subplots
        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add buttons
        self.setup_buttons()

        # Show initial batch
        self.show_current_batch()

    def setup_buttons(self):
        """Add navigation and control buttons"""
        # Create button axes
        ax_next = plt.axes([0.45, 0.02, 0.1, 0.04])
        ax_prev = plt.axes([0.34, 0.02, 0.1, 0.04])
        ax_random = plt.axes([0.56, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.67, 0.02, 0.1, 0.04])

        # Create buttons
        self.btn_next = Button(
            ax_next, "Next 10", color="orange", hovercolor="darkorange"
        )
        self.btn_prev = Button(
            ax_prev, "Previous 10", color="orange", hovercolor="darkorange"
        )
        self.btn_random = Button(
            ax_random, "Random 10", color="lightblue", hovercolor="blue"
        )
        self.btn_quit = Button(ax_quit, "Quit", color="red", hovercolor="darkred")

        # Connect button events
        self.btn_next.on_clicked(self.next_batch)
        self.btn_prev.on_clicked(self.prev_batch)
        self.btn_random.on_clicked(self.random_batch)
        self.btn_quit.on_clicked(self.quit_app)

    def show_current_batch(self):
        """Display current batch of 10 images with predictions"""
        start_idx = self.current_batch_idx * 10

        if start_idx >= len(self.all_images):
            self.current_batch_idx = 0
            start_idx = 0

        # Get predictions for current batch
        predictions, confidences, actual_count = self.predict_batch(start_idx)

        # Clear all axes first
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Display images
        for i in range(10):
            ax = self.axes[i]

            if i < actual_count:
                img_idx = start_idx + i

                # Display image
                ax.imshow(self.all_images[img_idx], cmap="gray", vmin=-1, vmax=1)

                # Get true and predicted labels
                true_label = self.all_labels[img_idx]
                pred_label = predictions[i]
                confidence = confidences[i]

                true_class = self.class_names[true_label]
                pred_class = self.class_names[pred_label]

                # Determine color based on correctness
                color = "lightgreen" if true_label == pred_label else "lightcoral"

                # Set title with prediction info
                title = (
                    f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}"
                )
                ax.set_title(title, fontsize=10, color=color, fontweight="bold")

                # Add border color based on correctness
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)

            else:
                # Hide unused subplots
                ax.set_visible(False)

        # Update figure title with batch info
        total_batches = (len(self.all_images) + 9) // 10  # Ceiling division
        batch_info = f"Hand Gesture Recognition - Batch {self.current_batch_idx + 1}/{total_batches}"
        self.fig.suptitle(batch_info, fontsize=16, fontweight="bold", color="white")

        # Calculate and display accuracy for current batch
        correct = sum(
            1
            for i in range(actual_count)
            if self.all_labels[start_idx + i] == predictions[i]
        )
        accuracy = correct / actual_count if actual_count > 0 else 0

        # Add accuracy text
        self.fig.text(
            0.02,
            0.95,
            f"Batch Accuracy: {accuracy:.1%} ({correct}/{actual_count})",
            fontsize=12,
            color="yellow",
            fontweight="bold",
        )

        # Add device info
        self.fig.text(
            0.02, 0.92, f"Device: {self.device}", fontsize=10, color="lightgray"
        )

        plt.draw()

    def next_batch(self, event):
        """Show next batch of images"""
        max_batch = (len(self.all_images) + 9) // 10 - 1
        if self.current_batch_idx < max_batch:
            self.current_batch_idx += 1
        else:
            self.current_batch_idx = 0  # Wrap around
        self.show_current_batch()

    def prev_batch(self, event):
        """Show previous batch of images"""
        max_batch = (len(self.all_images) + 9) // 10 - 1
        if self.current_batch_idx > 0:
            self.current_batch_idx -= 1
        else:
            self.current_batch_idx = max_batch  # Wrap around
        self.show_current_batch()

    def random_batch(self, event):
        """Show a random batch of images"""
        max_batch = (len(self.all_images) + 9) // 10 - 1
        self.current_batch_idx = random.randint(0, max_batch)
        self.show_current_batch()

    def quit_app(self, event):
        """Quit the application"""
        print("👋 Goodbye!")
        plt.close("all")
        sys.exit(0)

    def run(self):
        """Run the application"""
        print("\n" + "=" * 60)
        print("🚀 Hand Gesture Recognition Model Tester")
        print("=" * 60)
        print(f"📁 Data Directory: {self.data_dir}")
        if self.custom_data_dir and os.path.exists(self.custom_data_dir):
            print(f"📁 Custom Data Directory: {self.custom_data_dir}")
        print(f"🤖 Model: {os.path.basename(self.model_path)}")
        print(f"📊 Total Test Images: {len(self.all_images)}")
        print(f"💻 Device: {self.device}")
        print(f"🎯 Classes: {', '.join(self.class_names)}")
        print("\n📖 Instructions:")
        print("  • Click 'Next 10' to see next batch")
        print("  • Click 'Previous 10' to see previous batch")
        print("  • Click 'Random 10' to see random batch")
        print("  • Click 'Quit' to exit")
        print("  • Green border = Correct prediction")
        print("  • Red border = Incorrect prediction")
        print("=" * 60)

        # Show the plot
        plt.subplots_adjust(
            bottom=0.1, top=0.9, left=0.05, right=0.95, hspace=0.4, wspace=0.3
        )
        plt.show()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test hand gesture recognition model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to test data directory"
    )
    parser.add_argument(
        "--custom_data_dir",
        type=str,
        default="custom_data",
        help="Path to custom test data directory",
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Available model files in results/:")
        if os.path.exists("results/"):
            model_files = [f for f in os.listdir("results/") if f.endswith(".pth")]
            for f in model_files:
                print(f"  - {f}")
        sys.exit(1)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Check custom data directory (optional)
    if args.custom_data_dir and not os.path.exists(args.custom_data_dir):
        print(f"⚠️  Custom data directory not found: {args.custom_data_dir}")
        print("Will proceed with main dataset only.")
        args.custom_data_dir = None

    try:
        # Create and run tester
        tester = ModelTester(args.model_path, args.data_dir, args.custom_data_dir)
        tester.run()

    except KeyboardInterrupt:
        print("\n⚠️  Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Application failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
