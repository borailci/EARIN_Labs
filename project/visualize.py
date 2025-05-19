import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot learning curves (loss and accuracy vs. epochs)

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        output_dir (str): Directory to save the plots
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """
    Plot confusion matrix as a heatmap

    Args:
        conf_matrix (ndarray): Confusion matrix
        class_names (list): List of class names
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_class_metrics(precision, recall, f1, class_names, output_dir):
    """
    Plot per-class precision, recall, and F1 scores

    Args:
        precision (ndarray): Precision values for each class
        recall (ndarray): Recall values for each class
        f1 (ndarray): F1 scores for each class
        class_names (list): List of class names
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-Score")

    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Per-class Performance Metrics")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(output_dir, "class_metrics.png"), dpi=300)
    plt.close()


def plot_sample_predictions(
    model, test_loader, class_names, device, output_dir, num_samples=5
):
    """
    Plot sample predictions with ground truth and predicted labels

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        class_names: List of class names
        device: Device to run inference on
        output_dir: Directory to save the plot
        num_samples: Number of samples to display
    """
    model.eval()

    plt.figure(figsize=(15, 3 * num_samples))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Get the first image in the batch
            img = inputs[0].cpu().squeeze().numpy()
            true_label = class_names[targets[0].item()]
            pred_label = class_names[predicted[0].item()]

            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=300)
    plt.close()


def create_model_report(model_performance, output_dir):
    """
    Create a text report of model performance

    Args:
        model_performance (dict): Dictionary with performance metrics
        output_dir (str): Directory to save the report
    """
    report = "Model Performance Report\n"
    report += "======================\n\n"
    report += f"Accuracy: {model_performance['accuracy']:.4f}\n"
    report += f"Average Precision: {model_performance['precision']:.4f}\n"
    report += f"Average Recall: {model_performance['recall']:.4f}\n"
    report += f"Average F1-Score: {model_performance['f1']:.4f}\n\n"
    report += f"Training Time: {model_performance['training_time']:.2f} seconds\n"
    report += f"Model Parameters: {model_performance['parameters']}\n"

    with open(os.path.join(output_dir, "model_report.txt"), "w") as f:
        f.write(report)
