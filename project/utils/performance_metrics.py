import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def calculate_detailed_metrics(true_labels, predicted_labels, class_names):
    """
    Calculate detailed performance metrics for model evaluation.
    
    Args:
        true_labels (list): List of true labels
        predicted_labels (list): List of predicted labels
        class_names (dict): Dictionary mapping label indices to class names
        
    Returns:
        dict: Dictionary containing detailed metrics
    """
    # Convert class names dict to list if needed
    if isinstance(class_names, dict):
        classes = list(class_names.values())
    else:
        classes = class_names
        
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate precision, recall, and F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None
    )
    
    # Calculate accuracy
    accuracy = np.sum(np.array(true_labels) == np.array(predicted_labels)) / len(true_labels)
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
        "class_names": classes
    }
    
    return metrics

def confidence_histogram(confidences, correct_predictions, n_bins=10):
    """
    Create data for a confidence histogram to analyze model calibration.
    
    Args:
        confidences (list): List of prediction confidences
        correct_predictions (list): List indicating whether each prediction was correct
        n_bins (int): Number of bins for the histogram
        
    Returns:
        tuple: (bin_edges, accuracies, counts) for plotting
    """
    # Create bins for confidence values
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure within valid range
    
    # Calculate accuracy and count per bin
    accuracies = []
    counts = []
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = np.sum(mask)
        if count > 0:
            accuracy = np.sum(np.array(correct_predictions)[mask]) / count
        else:
            accuracy = 0
        
        accuracies.append(accuracy)
        counts.append(count)
    
    return bin_edges, np.array(accuracies), np.array(counts)
