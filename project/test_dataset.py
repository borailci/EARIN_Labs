import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import math
import time
import json
from torch.utils.data import DataLoader
from functools import lru_cache
import psutil  # For memory monitoring

from model import get_model
from dataset import get_class_names, LeapGestRecogDataset

# Import performance metrics from utils
try:
    from utils.performance_metrics import (
        calculate_detailed_metrics,
        confidence_histogram,
    )
except ImportError:
    print(
        "Performance metrics module not found. Some advanced analytics will be disabled."
    )

# Configure matplotlib for better visualization
try:
    plt.style.use("seaborn-v0_8-darkgrid")  # For newer matplotlib versions
except:
    try:
        plt.style.use("seaborn-darkgrid")  # For older matplotlib versions
    except:
        print("Could not set seaborn style, using default style.")


class DatasetTester:
    def __init__(self, model_path, data_dir):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = get_model(num_classes=10, hidden_size=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Profile model parameters
        self.model_size = (
            sum(p.numel() for p in self.model.parameters()) / 1e6
        )  # in millions
        print(f"Model parameters: {self.model_size:.2f}M")

        # Get class names
        self.class_names = get_class_names()

        # Define transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Create test dataset and loader
        self.test_dataset = LeapGestRecogDataset(
            data_dir, transform=self.transform, mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True,
        )

        # Track inference timings for performance metrics
        self.inference_times = []

        # Results cache to avoid redundant processing
        self.results_cache = {}

    @lru_cache(maxsize=32)  # Cache recent predictions
    def predict(self, image_tensor):
        """
        Make prediction on a single image with performance tracking.

        Args:
            image_tensor (torch.Tensor): Input image tensor

        Returns:
            tuple: (predicted_class_name, confidence, inference_time)
        """
        # Convert tensor to hashable form for cache check
        image_hash = image_tensor.cpu().numpy().tobytes()

        # Check cache first
        if image_hash in self.results_cache:
            return self.results_cache[image_hash]

        # Time the inference
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)

        # Get predicted class name
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()

        # Only consider high confidence predictions as valid
        if confidence_value < 0.5:
            result_class = "uncertain"
        else:
            result_class = predicted_class

        # Store in cache
        result = (result_class, confidence_value, inference_time)
        self.results_cache[image_hash] = result

        return result

    def create_results_visualization(self, results):
        """Create a stylish grid visualization of all results"""
        # Calculate grid dimensions
        n_images = len(results)
        grid_size = math.ceil(math.sqrt(n_images))

        # Define image display size with more space for info
        display_width = 280  # Increased width
        display_height = 300  # Increased height for more info

        # Create blank canvas with a dark background
        canvas_width = grid_size * display_width
        canvas_height = grid_size * display_height
        canvas = (
            np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 40
        )  # Dark gray background

        # Place images in grid
        for idx, result in enumerate(results):
            # Calculate grid position
            row = idx // grid_size
            col = idx % grid_size

            # Create cell background (slightly lighter than canvas)
            cell_y = row * display_height
            cell_x = col * display_width
            cv2.rectangle(
                canvas,
                (cell_x, cell_y),
                (cell_x + display_width - 2, cell_y + display_height - 2),
                (60, 60, 60),  # Lighter gray
                -1,
            )  # Filled rectangle

            # Add subtle border
            cv2.rectangle(
                canvas,
                (cell_x, cell_y),
                (cell_x + display_width - 2, cell_y + display_height - 2),
                (80, 80, 80),  # Border color
                1,
            )  # Border thickness

            # Convert tensor to numpy image
            img = result["image"].squeeze().numpy()
            img = (img * 0.5 + 0.5) * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Calculate image position (centered in cell)
            img_size = 180  # Fixed size for the gesture image
            img = cv2.resize(img, (img_size, img_size))

            img_x = cell_x + (display_width - img_size) // 2
            img_y = cell_y + 60  # Leave space at top for title

            # Place image
            canvas[img_y : img_y + img_size, img_x : img_x + img_size] = img

            # Get labels and status
            true_label = self.class_names[result["true"]]
            pred_label = result["pred"]
            confidence = result["conf"]
            is_correct = result["correct"]

            # Create title background
            title_height = 40
            title_color = (0, 70, 0) if is_correct else (70, 0, 0)  # Dark green/red
            cv2.rectangle(
                canvas,
                (cell_x, cell_y),
                (cell_x + display_width, cell_y + title_height),
                title_color,
                -1,
            )

            # Add status text
            status_text = "CORRECT" if is_correct else "INCORRECT"
            text_color = (150, 255, 150) if is_correct else (255, 150, 150)

            # Get text size for centering
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[
                0
            ]
            text_x = cell_x + (display_width - text_size[0]) // 2

            cv2.putText(
                canvas,
                status_text,
                (text_x, cell_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

            # Add prediction details
            info_y = img_y + img_size + 25

            # Class information header
            cv2.putText(
                canvas,
                "Classification Result:",
                (cell_x + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),  # Light gray
                1,
            )

            # True label
            cv2.putText(
                canvas,
                f"True Class: {true_label}",
                (cell_x + 10, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),  # Light gray
                1,
            )

            # Always show prediction
            pred_color = (
                (150, 255, 150) if is_correct else (255, 150, 150)
            )  # Green if correct, red if wrong
            cv2.putText(
                canvas,
                f"Predicted: {pred_label}",
                (cell_x + 10, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                pred_color,
                1,
            )

            # Confidence bar
            conf_bar_width = 150
            conf_bar_height = 15
            conf_x = cell_x + 10
            conf_y = info_y + 65  # Adjusted position for new layout

            # Draw confidence bar background
            cv2.rectangle(
                canvas,
                (conf_x, conf_y),
                (conf_x + conf_bar_width, conf_y + conf_bar_height),
                (60, 60, 60),
                -1,
            )

            # Draw confidence level
            conf_width = int(conf_bar_width * confidence)
            conf_color = (
                (0, 255, 0)
                if confidence > 0.8
                else (255, 255, 0) if confidence > 0.5 else (255, 0, 0)
            )
            cv2.rectangle(
                canvas,
                (conf_x, conf_y),
                (conf_x + conf_width, conf_y + conf_bar_height),
                conf_color,
                -1,
            )

            # Add confidence text
            cv2.putText(
                canvas,
                f"Confidence: {confidence:.2%}",
                (conf_x + conf_bar_width + 10, conf_y + conf_bar_height - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return canvas

    def visualize_calibration(self, confidences, correct_predictions, output_dir=None):
        """
        Create and save a calibration plot showing how well model confidence
        correlates with actual accuracy.

        Args:
            confidences (list): List of confidence values
            correct_predictions (list): List of booleans indicating correct predictions
            output_dir (str, optional): Directory to save the visualization
        """
        try:
            # Get histogram data
            bin_edges, accuracies, counts = confidence_histogram(
                confidences, correct_predictions, n_bins=10
            )

            # Create figure
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot histograms of sample counts
            ax1.bar(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                counts,
                width=0.08,
                alpha=0.5,
                color="blue",
                label="Sample Count",
            )
            ax1.set_ylabel("Sample Count", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            # Add second y-axis for accuracy
            ax2 = ax1.twinx()
            ax2.plot(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                accuracies,
                "r-o",
                label="Accuracy",
            )
            # Add perfect calibration line
            ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            ax2.set_ylim([0, 1.05])
            ax2.set_ylabel("Accuracy", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            # Add details
            plt.title("Model Calibration: Confidence vs. Accuracy")
            plt.xlabel("Confidence")

            # Add legend with both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            # Save if output directory specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(output_dir, "calibration_plot.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Could not generate calibration plot: {e}")

    def test_batch(self, batch_size=32):
        """
        Efficiently test the model using batches for faster processing.
        Useful for large test sets.

        Args:
            batch_size (int): Batch size for testing

        Returns:
            dict: Dictionary with test metrics
        """
        # Update dataloader with batch size
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Use more workers for batch processing
            pin_memory=True,
        )

        # Track metrics
        all_preds = []
        all_labels = []
        all_confidences = []
        total_time = 0
        total_samples = 0

        # Process batches
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Time inference
                start_time = time.time()
                outputs = self.model(images)
                batch_time = time.time() - start_time
                total_time += batch_time

                # Get predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)

                # Store results
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                total_samples += images.size(0)

        # Calculate metrics
        correct = sum(np.array(all_preds) == np.array(all_labels))
        accuracy = correct / total_samples
        avg_time_per_sample = total_time * 1000 / total_samples  # in ms

        # Generate detailed metrics if available
        try:
            detailed_metrics = calculate_detailed_metrics(
                all_labels, all_preds, self.class_names
            )
        except:
            detailed_metrics = {"accuracy": accuracy}

        # Add timing information
        detailed_metrics["avg_inference_time_ms"] = avg_time_per_sample
        detailed_metrics["total_samples"] = total_samples
        detailed_metrics["confidences"] = all_confidences

        print(f"\nBatch Testing Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Average inference time: {avg_time_per_sample:.2f}ms per sample")

        return detailed_metrics

    def test_and_visualize(self, num_samples=100, output_dir=None):
        """
        Test the model on the test set and visualize results with enhanced metrics

        Args:
            num_samples (int): Number of samples to test
            output_dir (str, optional): Directory to save results and visualizations

        Returns:
            dict: Test metrics and results
        """
        results = []
        correct = 0
        total = 0
        class_correct = {name: 0 for name in self.class_names.values()}
        class_total = {name: 0 for name in self.class_names.values()}

        # Track additional metrics
        confidences = []
        correct_predictions = []
        inference_times = []
        true_labels = []
        predicted_labels = []

        print("\nTesting model on test set...")
        start_time = time.time()

        # Process test samples
        for i, (image, label) in enumerate(self.test_loader):
            if i >= num_samples:
                break

            # Move to device
            image = image.to(self.device)
            label = label.to(self.device)

            # Get prediction with timing
            pred_class, confidence, pred_time = self.predict(image)

            # Check if prediction is correct
            true_class = self.class_names[label.item()]
            is_correct = pred_class == true_class

            # Update counters
            if is_correct:
                correct += 1
                class_correct[true_class] += 1
            total += 1
            class_total[true_class] += 1

            # Track metrics
            confidences.append(confidence)
            correct_predictions.append(is_correct)
            inference_times.append(pred_time)
            true_labels.append(label.item())
            predicted_labels.append(
                list(self.class_names.values()).index(pred_class)
                if pred_class in self.class_names.values()
                else -1
            )

            # Store result
            results.append(
                {
                    "image": image.cpu(),
                    "true": label.item(),
                    "pred": pred_class,
                    "conf": confidence,
                    "correct": is_correct,
                    "inference_time": pred_time,
                }
            )

        # Calculate accuracy and timing metrics
        accuracy = correct / total if total > 0 else 0
        avg_inference_time = np.mean(inference_times)
        total_test_time = time.time() - start_time

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Print detailed summary
        # Get memory usage
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        # Get cache stats
        cache_info = self.predict.cache_info()
        cache_hit_ratio = (
            cache_info.hits / (cache_info.hits + cache_info.misses)
            if (cache_info.hits + cache_info.misses) > 0
            else 0
        )

        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Avg Inference Time: {avg_inference_time:.2f}ms per sample")
        print(f"Total Test Time: {total_test_time:.2f}s")
        print(f"Memory Usage: {mem_usage:.2f} MB")
        print(
            f"Cache Hit Ratio: {cache_hit_ratio:.2%} ({cache_info.hits} hits, {cache_info.misses} misses)"
        )

        print("\nPer-Class Performance:")
        print("-" * 50)
        print(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total':<15}")
        print("-" * 50)
        for class_name in self.class_names.values():
            if class_total[class_name] > 0:
                class_acc = class_correct[class_name] / class_total[class_name]
                print(
                    f"{class_name:<15} {class_acc:>8.2%}  ({class_correct[class_name]}/{class_total[class_name]})"
                )

        # Generate calibration plot if performance metrics available
        try:
            if output_dir:
                self.visualize_calibration(confidences, correct_predictions, output_dir)
            else:
                self.visualize_calibration(confidences, correct_predictions)
        except Exception as e:
            print(f"Could not generate calibration plot: {e}")

        # Create and show visualization
        visualization = self.create_results_visualization(results)

        # Show window with results
        cv2.namedWindow("Hand Gesture Recognition - Test Results", cv2.WINDOW_NORMAL)
        cv2.imshow("Hand Gesture Recognition - Test Results", visualization)
        print("\nShowing test results visualization. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save visualization and results
        output_path = output_dir if output_dir else "."
        viz_path = os.path.join(output_path, "test_results_visualization.png")
        cv2.imwrite(viz_path, visualization)
        print(f"\nVisualization saved as '{viz_path}'")

        # Save detailed metrics if output directory specified
        if output_dir:
            # Create metrics summary
            metrics = {
                "accuracy": accuracy,
                "total_samples": total,
                "correct_samples": correct,
                "avg_inference_time_ms": avg_inference_time,
                "class_accuracy": {
                    class_name: (
                        (class_correct[class_name] / class_total[class_name])
                        if class_total[class_name] > 0
                        else 0
                    )
                    for class_name in self.class_names.values()
                },
            }

            # Save metrics as JSON
            metrics_path = os.path.join(output_dir, "test_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Detailed metrics saved to {metrics_path}")

        # Return consolidated results
        return {
            "accuracy": accuracy,
            "inference_time": avg_inference_time,
            "confidences": confidences,
            "correct_predictions": correct_predictions,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test and visualize dataset results")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save test results and visualizations",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Use batch processing for faster testing on larger datasets",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for batch mode processing",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the model to ONNX format after testing",
    )
    args = parser.parse_args()

    # Create tester
    tester = DatasetTester(args.model_path, args.data_dir)

    # Run test based on mode
    if args.batch_mode:
        print(f"Running batch test with batch size {args.batch_size}...")
        metrics = tester.test_batch(batch_size=args.batch_size)

        # Save metrics if output directory provided
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            metrics_path = os.path.join(args.output_dir, "batch_test_metrics.json")
            with open(metrics_path, "w") as f:
                # Convert numpy values to Python native types for JSON serialization
                serializable_metrics = {}
                for k, v in metrics.items():
                    if k != "confidences":  # Skip large arrays
                        if isinstance(v, np.ndarray):
                            serializable_metrics[k] = v.tolist()
                        else:
                            serializable_metrics[k] = v

                json.dump(serializable_metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")
    else:
        # Run standard test with visualization
        tester.test_and_visualize(args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()
