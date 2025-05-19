import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_model
from dataset import get_data_loaders, get_class_names


def load_model(model_path, device):
    """Load the trained model"""
    model = get_model(num_classes=10, hidden_size=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def create_page(
    images, labels, predictions, confidences, class_names, page_num, total_pages
):
    """Create a single page with 10 frames"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f"Gesture Recognition Test Results (Page {page_num+1}/{total_pages})",
        fontsize=16,
    )

    for idx, (image, true_label, pred_label, confidence) in enumerate(
        zip(images, labels, predictions, confidences)
    ):
        row = idx // 5
        col = idx % 5

        # Get class names
        true_class = class_names[f"{true_label+1:02d}"]
        pred_class = class_names[f"{pred_label+1:02d}"]

        # Plot image
        ax = axes[row, col]
        ax.imshow(image, cmap="gray")

        # Add text with better spacing
        result_color = "green" if true_label == pred_label else "red"

        # Add text boxes with more space between them
        ax.text(
            0.5,
            0.95,
            f"True: {true_class}",
            transform=ax.transAxes,
            color="green",
            ha="center",
            bbox=dict(facecolor="black", alpha=0.7),
        )
        ax.text(
            0.5,
            0.85,
            f"Pred: {pred_class}",
            transform=ax.transAxes,
            color=result_color,
            ha="center",
            bbox=dict(facecolor="black", alpha=0.7),
        )
        ax.text(
            0.5,
            0.75,
            f"Conf: {confidence:.2f}",
            transform=ax.transAxes,
            color=result_color,
            ha="center",
            bbox=dict(facecolor="black", alpha=0.7),
        )

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "results/best_model.pth"
    model = load_model(model_path, device)

    # Get class names
    class_names = get_class_names()

    # Load test data
    data_loaders = get_data_loaders(
        data_dir="data", custom_data_dir="custom_data", batch_size=1, num_workers=0
    )
    test_loader = data_loaders["test"]

    # Process frames
    images = []
    true_labels = []
    pred_labels = []
    confidences = []
    frame_count = 0

    for batch_images, batch_labels in test_loader:
        if frame_count >= 50:
            break

        # Get image and label
        image = batch_images[0].numpy().squeeze()
        true_label = batch_labels[0].item()

        # Get prediction
        with torch.no_grad():
            output = model(batch_images.to(device))
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_label = torch.max(probabilities, 1)
            pred_label = pred_label[0].item()
            confidence = confidence[0].item()

        # Store results
        images.append(image)
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confidences.append(confidence)

        frame_count += 1

    # Create pages (10 frames per page)
    frames_per_page = 10
    total_pages = (len(images) + frames_per_page - 1) // frames_per_page

    for page in range(total_pages):
        start_idx = page * frames_per_page
        end_idx = min((page + 1) * frames_per_page, len(images))

        # Create page
        fig = create_page(
            images[start_idx:end_idx],
            true_labels[start_idx:end_idx],
            pred_labels[start_idx:end_idx],
            confidences[start_idx:end_idx],
            class_names,
            page,
            total_pages,
        )

        # Show page
        plt.show()

        # Wait for user to close the window
        plt.waitforbuttonpress()
        plt.close(fig)


if __name__ == "__main__":
    main()
