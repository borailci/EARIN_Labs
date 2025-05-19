import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from torchvision.utils import make_grid
import pandas as pd
from thop import profile

from model import get_model
from dataset import get_data_loaders, get_class_names


def plot_confusion_matrix(model, test_loader, class_names, device, output_dir):
    """Create and plot confusion matrix"""
    # Get predictions
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return cm


def plot_class_metrics(model, test_loader, class_names, device, output_dir):
    """Plot per-class precision, recall, and F1 scores"""
    # Get predictions
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Get classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # Create DataFrame for plotting
    metrics_df = pd.DataFrame(
        {
            "Precision": [
                report[class_name]["precision"] for class_name in class_names
            ],
            "Recall": [report[class_name]["recall"] for class_name in class_names],
            "F1-Score": [report[class_name]["f1-score"] for class_name in class_names],
        },
        index=class_names,
    )

    # Plot metrics
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind="bar", width=0.8)
    plt.title("Per-class Metrics")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.grid(True, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_metrics.png"))
    plt.close()

    return report


def visualize_predictions(
    model, test_loader, class_names, device, output_dir, num_samples=10
):
    """Visualize model predictions on sample frames"""
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Create figure
    plt.figure(figsize=(15, 2 * num_samples))
    for i in range(num_samples):
        # Plot original image with prediction
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(images[i][0].cpu(), cmap="gray")

        # Add colored title based on correctness
        is_correct = predicted[i] == labels[i]
        color = "green" if is_correct else "red"
        plt.title(
            f"True: {class_names[labels[i]]} | Predicted: {class_names[predicted[i]]}",
            color=color,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"))
    plt.close()


def calculate_model_stats(model, test_loader, device):
    """Calculate model parameters, FLOPs, and inference time"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate FLOPs
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))

    # Measure inference time
    model.eval()
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            start_time = time.time()
            _ = model(images)
            total_time += time.time() - start_time
            num_samples += images.size(0)

    avg_inference_time = (total_time / num_samples) * 1000  # Convert to milliseconds

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "FLOPs": flops,
        "Average Inference Time (ms)": avg_inference_time,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive test results visualization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/best_model.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Directory to save visualizations"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_model(num_classes=10, hidden_size=128).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Get data loaders and class names
    data_loaders = get_data_loaders(args.data, batch_size=32)
    test_loader = data_loaders["test"]
    class_names = list(get_class_names().values())

    print("Generating visualizations...")

    # Create confusion matrix
    print("1. Creating confusion matrix...")
    cm = plot_confusion_matrix(model, test_loader, class_names, device, args.output)

    # Plot class metrics
    print("2. Plotting class metrics...")
    metrics = plot_class_metrics(model, test_loader, class_names, device, args.output)

    # Visualize predictions
    print("3. Visualizing sample predictions...")
    visualize_predictions(model, test_loader, class_names, device, args.output)

    # Calculate and save model statistics
    print("4. Calculating model statistics...")
    stats = calculate_model_stats(model, test_loader, device)

    # Save model statistics to file
    with open(os.path.join(args.output, "model_report.txt"), "w") as f:
        f.write("Model Statistics:\n")
        f.write("-----------------\n")
        for key, value in stats.items():
            if key == "FLOPs":
                f.write(f"{key}: {value:,} operations\n")
            elif "Parameters" in key:
                f.write(f"{key}: {value:,}\n")
            else:
                f.write(f"{key}: {value:.2f}\n")

        # Add overall metrics from classification report
        f.write("\nOverall Metrics:\n")
        f.write("---------------\n")
        f.write(f"Accuracy: {metrics['macro avg']['precision']:.4f}\n")
        f.write(f"Macro Avg Precision: {metrics['macro avg']['precision']:.4f}\n")
        f.write(f"Macro Avg Recall: {metrics['macro avg']['recall']:.4f}\n")
        f.write(f"Macro Avg F1-Score: {metrics['macro avg']['f1-score']:.4f}\n")

    print("\nVisualization complete! Results saved in:", args.output)


if __name__ == "__main__":
    main()
