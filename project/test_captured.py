import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict
import math

from model import get_model
from dataset import get_class_names


class CapturedDataTester:
    def __init__(self, model_path, captured_data_dir):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = get_model(num_classes=10, hidden_size=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Get class names
        self.class_names = get_class_names()

        # Map captured gesture names to model class names
        self.gesture_mapping = {
            "thumb_up": "thumb",
            "wave": "palm_moved",
            "ok_sign": "ok",
            "fist": "fist",
            "palm": "palm",
            "l_sign": "l",
            "c_sign": "c",
            "down": "down",
            "index": "index",
            "fist_moved": "fist_moved",
        }

        # Set data directory
        self.data_dir = captured_data_dir

        # Define preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def load_metadata(self, capture_dir):
        """Load metadata from a capture directory"""
        metadata_path = os.path.join(capture_dir, "metadata.txt")
        metadata = {}
        with open(metadata_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                metadata[key] = value
        return metadata

    def process_image(self, image_path):
        """Process a single image for model input"""
        # Load and convert to grayscale
        image = Image.open(image_path).convert("L")

        # Apply transforms
        tensor = self.transform(image)

        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_tensor):
        """Make prediction on a single image"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return self.class_names[predicted.item()], confidence.item()

    def analyze_results(self):
        """Analyze all captured images and generate report"""
        results = []
        true_predictions = defaultdict(list)
        false_predictions = defaultdict(list)

        # Process each capture
        capture_dirs = sorted(
            [d for d in os.listdir(self.data_dir) if d.startswith("capture_")]
        )

        print(f"\nAnalyzing {len(capture_dirs)} captures...")

        for capture_dir in capture_dirs:
            full_dir = os.path.join(self.data_dir, capture_dir)

            # Load metadata (original prediction)
            metadata = self.load_metadata(full_dir)
            original_gesture = metadata["Prediction"]
            # Map the original gesture to our model's class names
            original_pred = self.gesture_mapping.get(original_gesture, original_gesture)
            original_conf = float(metadata["Confidence"])

            # Load and process ROI image
            roi_path = os.path.join(full_dir, "roi.png")
            image_tensor = self.process_image(roi_path)

            # Get new prediction
            new_pred, new_conf = self.predict(image_tensor)

            # Store results
            result = {
                "capture": capture_dir,
                "original_pred": original_pred,
                "original_conf": original_conf,
                "new_pred": new_pred,
                "new_conf": new_conf,
                "matches": original_pred == new_pred,
                "image_path": roi_path,
            }
            results.append(result)

            # Store for visualization
            if result["matches"]:
                true_predictions[original_pred].append(roi_path)
            else:
                false_predictions[original_pred].append(
                    {"path": roi_path, "predicted": new_pred}
                )

        return results, true_predictions, false_predictions

    def create_results_visualization(self, results):
        """Create a grid visualization of all results"""
        # Calculate grid dimensions
        n_images = len(results)
        grid_size = math.ceil(math.sqrt(n_images))

        # Define image display size
        display_width = 250  # Increased width for more text
        display_height = 250  # Increased height for more text

        # Create blank canvas
        canvas_width = grid_size * display_width
        canvas_height = grid_size * display_height
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place images in grid
        for idx, result in enumerate(results):
            # Calculate grid position
            row = idx // grid_size
            col = idx % grid_size

            # Load and resize image
            img = cv2.imread(result["image_path"])
            img = cv2.resize(
                img, (display_width - 20, display_height - 60)
            )  # More space for text

            # Calculate position in canvas
            y_start = row * display_height + 30  # More space at top
            x_start = col * display_width + 10
            y_end = y_start + img.shape[0]
            x_end = x_start + img.shape[1]

            # Place image
            canvas[y_start:y_end, x_start:x_end] = img

            # Add frame number
            frame_text = f"Frame: {result['capture']}"
            cv2.putText(
                canvas,
                frame_text,
                (x_start, y_start - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            # Add true label
            true_text = f"True: {result['original_pred']}"
            cv2.putText(
                canvas,
                true_text,
                (x_start, y_end + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Add prediction and confidence
            pred_text = f"Pred: {result['new_pred']} ({result['new_conf']:.2f})"
            color = (
                (0, 255, 0) if result["matches"] else (0, 0, 255)
            )  # Green if correct, Red if wrong
            cv2.putText(
                canvas,
                pred_text,
                (x_start, y_end + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return canvas

    def generate_report(self):
        """Generate and display analysis report"""
        results, true_predictions, false_predictions = self.analyze_results()

        # Calculate statistics
        total = len(results)
        correct = sum(1 for r in results if r["matches"])
        accuracy = correct / total

        # Print summary with more detail
        print("\n" + "=" * 50)
        print("GESTURE RECOGNITION ANALYSIS REPORT")
        print("=" * 50)
        print(f"\nTotal samples analyzed: {total}")
        print(f"Correct predictions:    {correct}")
        print(f"Incorrect predictions:  {total - correct}")
        print(f"Overall accuracy:       {accuracy:.2%}")

        # Print per-class statistics with more detail
        print("\nPER-CLASS PERFORMANCE:")
        print("-" * 50)
        print(
            f"{'Gesture Type':<15} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}"
        )
        print("-" * 50)

        for gesture in self.class_names.values():
            class_samples = [r for r in results if r["original_pred"] == gesture]
            if class_samples:
                class_correct = sum(1 for r in class_samples if r["matches"])
                class_acc = class_correct / len(class_samples)
                avg_conf = np.mean([r["new_conf"] for r in class_samples])
                print(
                    f"{gesture:<15} {len(class_samples):>8d} {class_correct:>8d} "
                    f"{class_acc:>9.2%} {avg_conf:>9.2f}"
                )

        # Print misclassifications with more detail
        print("\nDETAILED MISCLASSIFICATIONS:")
        print("-" * 80)
        print(
            f"{'Frame':<20} {'True Label':<12} {'Predicted':<12} {'Confidence':>8} {'Status'}"
        )
        print("-" * 80)

        for r in results:
            if not r["matches"]:
                status = "INCORRECT"
                print(
                    f"{r['capture']:<20} {r['original_pred']:<12} {r['new_pred']:<12} "
                    f"{r['new_conf']:>8.2f}  {status}"
                )

        # Create confusion matrix
        y_true = [r["original_pred"] for r in results]
        y_pred = [r["new_pred"] for r in results]
        labels = list(self.class_names.values())

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Plot confusion matrix with improved styling
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Number of Predictions"},
        )
        plt.title("Confusion Matrix of Gesture Recognition", pad=20)
        plt.xlabel("Predicted Gesture")
        plt.ylabel("True Gesture")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, "confusion_matrix.png"))
        plt.close()

        # Save detailed results
        with open(os.path.join(self.data_dir, "analysis_report.txt"), "w") as f:
            f.write("=" * 50 + "\n")
            f.write("GESTURE RECOGNITION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("SUMMARY:\n")
            f.write(f"Total samples analyzed: {total}\n")
            f.write(f"Correct predictions:    {correct}\n")
            f.write(f"Incorrect predictions:  {total - correct}\n")
            f.write(f"Overall accuracy:       {accuracy:.2%}\n\n")

            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"{'Gesture Type':<15} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}\n"
            )
            f.write("-" * 50 + "\n")

            for gesture in self.class_names.values():
                class_samples = [r for r in results if r["original_pred"] == gesture]
                if class_samples:
                    class_correct = sum(1 for r in class_samples if r["matches"])
                    class_acc = class_correct / len(class_samples)
                    avg_conf = np.mean([r["new_conf"] for r in class_samples])
                    f.write(
                        f"{gesture:<15} {len(class_samples):>8d} {class_correct:>8d} "
                        f"{class_acc:>9.2%} {avg_conf:>9.2f}\n"
                    )

            f.write("\nDETAILED MISCLASSIFICATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Frame':<20} {'True Label':<12} {'Predicted':<12} {'Confidence':>8} {'Status'}\n"
            )
            f.write("-" * 80 + "\n")

            for r in results:
                if not r["matches"]:
                    status = "INCORRECT"
                    f.write(
                        f"{r['capture']:<20} {r['original_pred']:<12} {r['new_pred']:<12} "
                        f"{r['new_conf']:>8.2f}  {status}\n"
                    )

        # Create and show visualization
        visualization = self.create_results_visualization(results)

        # Show window with results
        cv2.namedWindow("Gesture Recognition Results", cv2.WINDOW_NORMAL)
        cv2.imshow("Gesture Recognition Results", visualization)
        print("\nShowing results visualization. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\nAnalysis complete!")
        print(
            f"- Detailed report saved to: {os.path.join(self.data_dir, 'analysis_report.txt')}"
        )
        print(
            f"- Confusion matrix saved to: {os.path.join(self.data_dir, 'confusion_matrix.png')}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test captured gesture data")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="captured_data",
        help="Directory containing captured data",
    )

    args = parser.parse_args()

    # Initialize and run tester
    tester = CapturedDataTester(args.model_path, args.data_dir)
    tester.generate_report()


if __name__ == "__main__":
    main()
