import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import time

from model import get_model
from dataset import get_class_names, LeapGestRecogDataset


class TestVisualizer:
    def __init__(self, model_path, custom_data_dir):
        """Initialize the test visualizer"""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = get_model(num_classes=10, hidden_size=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Initialize transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Get class names
        self.class_names = get_class_names()

        # Set data directory
        self.custom_data_dir = custom_data_dir

        # Create output directory
        self.output_dir = "visualization_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_process_dataset(self):
        """Load and process the custom dataset"""
        # Create dataset
        dataset = LeapGestRecogDataset(
            self.custom_data_dir, transform=self.transform, mode="test"
        )

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        return dataset, dataloader

    def test_and_visualize(self):
        """Run tests and create visualizations"""
        print("Loading dataset...")
        dataset, dataloader = self.load_and_process_dataset()

        # Initialize metrics
        all_preds = []
        all_labels = []
        all_confidences = []
        all_images = []
        all_true_labels = []

        print("Running inference...")
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Get predictions
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)

                # Store results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                # Store images and labels for visualization
                all_images.extend(inputs.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)

        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        print(f"\nOverall Accuracy: {accuracy:.2%}")

        # Create visualizations
        self.create_confusion_matrix(all_labels, all_preds)
        self.create_confidence_histogram(all_confidences)
        self.create_class_accuracy_plot(all_labels, all_preds)
        self.create_sample_predictions(
            all_images, all_labels, all_preds, all_confidences
        )
        self.create_error_analysis(all_images, all_labels, all_preds, all_confidences)
        self.create_large_grid_visualization(
            all_images, all_labels, all_preds, all_confidences
        )

        print(f"\nVisualizations saved to {self.output_dir}")

    def create_confusion_matrix(self, true_labels, pred_labels):
        """Create and save confusion matrix"""
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.class_names.values()),
            yticklabels=list(self.class_names.values()),
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def create_confidence_histogram(self, confidences):
        """Create and save confidence histogram"""
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, range=(0, 1))
        plt.title("Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "confidence_histogram.png"))
        plt.close()

    def create_class_accuracy_plot(self, true_labels, pred_labels):
        """Create and save class-wise accuracy plot"""
        class_accuracies = []
        for i in range(10):
            mask = true_labels == i
            if np.any(mask):
                accuracy = np.mean(pred_labels[mask] == true_labels[mask])
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)

        plt.figure(figsize=(12, 6))
        plt.bar(range(10), class_accuracies)
        plt.xticks(range(10), list(self.class_names.values()), rotation=45)
        plt.title("Accuracy by Class")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "class_accuracy.png"))
        plt.close()

    def create_sample_predictions(self, images, true_labels, pred_labels, confidences):
        """Create and save sample predictions visualization"""
        # Select samples for visualization (2 correct and 2 incorrect per class)
        samples = []
        for class_idx in range(10):
            # Get indices for this class
            class_indices = np.where(true_labels == class_idx)[0]

            # Get correct and incorrect predictions
            correct_indices = class_indices[pred_labels[class_indices] == class_idx]
            incorrect_indices = class_indices[pred_labels[class_indices] != class_idx]

            # Add samples
            if len(correct_indices) > 0:
                samples.append((correct_indices[0], True))
            if len(incorrect_indices) > 0:
                samples.append((incorrect_indices[0], False))

        # Create visualization
        n_samples = len(samples)
        fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 6))
        axes = axes.flatten()

        for idx, (sample_idx, is_correct) in enumerate(samples):
            if idx >= len(axes):
                break

            # Get image and predictions
            image = images[sample_idx].squeeze()
            true_label = true_labels[sample_idx]
            pred_label = pred_labels[sample_idx]
            confidence = confidences[sample_idx]

            # Plot image
            axes[idx].imshow(image, cmap="gray")
            axes[idx].axis("off")

            # Add title
            color = "green" if is_correct else "red"
            title = f"True: {self.class_names[f'{true_label+1:02d}']}\n"
            title += f"Pred: {self.class_names[f'{pred_label+1:02d}']}\n"
            title += f"Conf: {confidence:.2%}"
            axes[idx].set_title(title, color=color)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sample_predictions.png"))
        plt.close()

    def create_error_analysis(self, images, true_labels, pred_labels, confidences):
        """Create and save error analysis visualization"""
        # Find misclassified samples
        misclassified = np.where(true_labels != pred_labels)[0]

        if len(misclassified) == 0:
            print("No misclassified samples found!")
            return

        # Create visualization
        n_samples = min(16, len(misclassified))
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.flatten()

        for idx, sample_idx in enumerate(misclassified[:n_samples]):
            if idx >= len(axes):
                break

            # Get image and predictions
            image = images[sample_idx].squeeze()
            true_label = true_labels[sample_idx]
            pred_label = pred_labels[sample_idx]
            confidence = confidences[sample_idx]

            # Plot image
            axes[idx].imshow(image, cmap="gray")
            axes[idx].axis("off")

            # Add title
            title = f"True: {self.class_names[f'{true_label+1:02d}']}\n"
            title += f"Pred: {self.class_names[f'{pred_label+1:02d}']}\n"
            title += f"Conf: {confidence:.2%}"
            axes[idx].set_title(title, color="red")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_analysis.png"))
        plt.close()

    def create_large_grid_visualization(
        self, images, true_labels, pred_labels, confidences
    ):
        """Create a large grid visualization of 50 images with their predictions"""
        # Select 50 samples (5 from each class)
        samples = []
        for class_idx in range(10):
            # Get indices for this class
            class_indices = np.where(true_labels == class_idx)[0]
            # Take first 5 samples
            samples.extend(class_indices[:5])

        # Create figure
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(10, 5, hspace=0.3, wspace=0.3)

        # Plot images
        for idx, sample_idx in enumerate(samples):
            row = idx // 5
            col = idx % 5

            ax = fig.add_subplot(gs[row, col])

            # Get image and predictions
            image = images[sample_idx].squeeze()
            true_label = true_labels[sample_idx]
            pred_label = pred_labels[sample_idx]
            confidence = confidences[sample_idx]

            # Plot image
            ax.imshow(image, cmap="gray")
            ax.axis("off")

            # Add title
            is_correct = true_label == pred_label
            color = "green" if is_correct else "red"
            title = f"True: {self.class_names[f'{true_label+1:02d}']}\n"
            title += f"Pred: {self.class_names[f'{pred_label+1:02d}']}\n"
            title += f"Conf: {confidence:.2%}"
            ax.set_title(title, color=color, fontsize=8)

        plt.suptitle("Sample Predictions (50 Images)", fontsize=16, y=0.95)
        plt.savefig(
            os.path.join(self.output_dir, "large_grid_visualization.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize test results")
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--custom_data_dir",
        type=str,
        default="dataset/custom",
        help="Path to custom dataset directory",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Initialize and run visualizer
    visualizer = TestVisualizer(args.model_path, args.custom_data_dir)
    visualizer.test_and_visualize()


if __name__ == "__main__":
    main()
