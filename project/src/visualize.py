import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None, normalize=True):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        confusion_matrix: 2D array of confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        # Normalize by row (true label)
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_class_metrics(precisions, recalls, f1_scores, class_names, save_path=None):
    """
    Plot precision, recall, and F1-score for each class as a bar chart
    
    Args:
        precisions: List of precision values for each class
        recalls: List of recall values for each class
        f1_scores: List of F1-scores for each class
        class_names: List of class names
        save_path: Path to save the figure
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precisions, width, label='Precision', color='#5DA5DA')
    plt.bar(x, recalls, width, label='Recall', color='#FAA43A')
    plt.bar(x + width, f1_scores, width, label='F1-score', color='#60BD68')
    
    plt.ylabel('Score (%)')
    plt.title('Per-class Performance Metrics')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def visualize_sample_predictions(model, data, class_names, device, num_samples=5, save_dir=None):
    """
    Visualize model predictions on sample data
    
    Args:
        model: The trained model
        data: A batch of data (inputs, targets)
        class_names: List of class names
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_dir: Directory to save the figures
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model.eval()
    inputs, targets = data
    
    # Only take up to num_samples
    if len(inputs) > num_samples:
        inputs = inputs[:num_samples]
        targets = targets[:num_samples]
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
    
    for i in range(len(inputs)):
        # For sequences, take first, middle, and last frames
        if inputs[i].dim() == 4:  # [seq_len, channels, H, W]
            seq_len = inputs[i].size(0)
            frame_indices = [0, seq_len//2, seq_len-1] if seq_len > 2 else list(range(seq_len))
            frames = [inputs[i, idx, 0].cpu().numpy() for idx in frame_indices]  # Take first channel
            
            fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
            if len(frames) == 1:
                axes = [axes]
            
            for j, frame in enumerate(frames):
                axes[j].imshow(frame, cmap='gray')
                axes[j].axis('off')
                axes[j].set_title(f"Frame {frame_indices[j]}")
            
            pred_label = class_names[predicted[i].item()]
            true_label = class_names[targets[i].item()]
            prob = probabilities[i, predicted[i]].item() * 100
            
            fig.suptitle(
                f"True: {true_label}, Predicted: {pred_label} ({prob:.1f}%)",
                fontsize=14
            )
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=300, bbox_inches='tight')
            
            plt.close()
        
        # For single images
        else:
            plt.figure(figsize=(6, 6))
            plt.imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            
            pred_label = class_names[predicted[i].item()]
            true_label = class_names[targets[i].item()]
            prob = probabilities[i, predicted[i]].item() * 100
            
            plt.title(
                f"True: {true_label}, Predicted: {pred_label} ({prob:.1f}%)",
                fontsize=14
            )
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=300, bbox_inches='tight')
            
            plt.close()
