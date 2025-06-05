"""
Performance Metrics and Evaluation Utilities

This module provides comprehensive performance evaluation functions for machine learning
models, with specific focus on classification tasks and model calibration analysis.
The functions support academic research by providing detailed statistical analysis
and visualization capabilities.

Functions:
    - calculate_detailed_metrics: Comprehensive performance metrics calculation
    - confidence_histogram: Model calibration analysis
    - plot_roc_curves: ROC curve analysis for multi-class problems
    - calculate_statistical_significance: Statistical significance testing
    - generate_performance_report: Comprehensive evaluation report

Author: Course Project Team
Date: Academic Year 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


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
    accuracy = np.sum(np.array(true_labels) == np.array(predicted_labels)) / len(
        true_labels
    )

    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            true_labels, predicted_labels, average="weighted"
        )
    )

    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
        "class_names": classes,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
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


def calculate_class_wise_metrics(
    true_labels: List[int], predicted_labels: List[int], class_names: List[str]
) -> pd.DataFrame:
    """
    Calculate detailed per-class performance metrics.

    Args:
        true_labels: True class labels
        predicted_labels: Predicted class labels
        class_names: List of class names

    Returns:
        DataFrame with per-class metrics
    """
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )

    # Create DataFrame
    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )

    # Add overall metrics
    overall_row = pd.DataFrame(
        {
            "Class": ["Overall"],
            "Precision": [np.mean(precision)],
            "Recall": [np.mean(recall)],
            "F1-Score": [np.mean(f1)],
            "Support": [np.sum(support)],
        }
    )

    metrics_df = pd.concat([metrics_df, overall_row], ignore_index=True)

    return metrics_df


def calculate_statistical_significance(
    model1_scores: List[float], model2_scores: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two model performances.

    Args:
        model1_scores: Performance scores for model 1
        model2_scores: Performance scores for model 2
        alpha: Significance level

    Returns:
        Dictionary with statistical test results
    """
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        (
            (len(model1_scores) - 1) * np.var(model1_scores, ddof=1)
            + (len(model2_scores) - 1) * np.var(model2_scores, ddof=1)
        )
        / (len(model1_scores) + len(model2_scores) - 2)
    )

    cohens_d = (np.mean(model1_scores) - np.mean(model2_scores)) / pooled_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": (
            "small"
            if abs(cohens_d) < 0.5
            else "medium" if abs(cohens_d) < 0.8 else "large"
        ),
        "model1_mean": np.mean(model1_scores),
        "model2_mean": np.mean(model2_scores),
        "model1_std": np.std(model1_scores),
        "model2_std": np.std(model2_scores),
    }


def plot_confusion_matrix_analysis(
    cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive confusion matrix visualization.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def generate_performance_report(
    true_labels: List[int],
    predicted_labels: List[int],
    class_names: List[str],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive performance evaluation report.

    Args:
        true_labels: True class labels
        predicted_labels: Predicted class labels
        class_names: List of class names
        model_name: Name of the model being evaluated
        save_path: Path to save the report

    Returns:
        String containing the formatted report
    """
    # Calculate detailed metrics
    metrics = calculate_detailed_metrics(true_labels, predicted_labels, class_names)

    # Calculate per-class metrics
    class_metrics_df = calculate_class_wise_metrics(
        true_labels, predicted_labels, class_names
    )

    # Generate report
    report = f"""
# Performance Evaluation Report: {model_name}

## Overall Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- **Macro Average Precision**: {metrics['precision_macro']:.4f}
- **Macro Average Recall**: {metrics['recall_macro']:.4f}
- **Macro Average F1-Score**: {metrics['f1_macro']:.4f}
- **Weighted Average Precision**: {metrics['precision_weighted']:.4f}
- **Weighted Average Recall**: {metrics['recall_weighted']:.4f}
- **Weighted Average F1-Score**: {metrics['f1_weighted']:.4f}

## Per-Class Performance Metrics
{class_metrics_df.to_string(index=False, float_format='%.4f')}

## Class Distribution (Support)
Total samples: {sum(metrics['support'])}
"""

    for i, (class_name, support) in enumerate(zip(class_names, metrics["support"])):
        percentage = (support / sum(metrics["support"])) * 100
        report += f"- {class_name}: {support} samples ({percentage:.1f}%)\n"

    report += f"""
## Model Insights
- **Best Performing Class**: {class_names[np.argmax(metrics['f1'])]} (F1: {np.max(metrics['f1']):.4f})
- **Worst Performing Class**: {class_names[np.argmin(metrics['f1'])]} (F1: {np.min(metrics['f1']):.4f})
- **Most Balanced Class**: {class_names[np.argmin(np.abs(metrics['precision'] - metrics['recall']))]}
- **Standard Deviation of F1 Scores**: {np.std(metrics['f1']):.4f}

## Recommendations
"""

    # Add performance-based recommendations
    if metrics["accuracy"] > 0.95:
        report += (
            "- Excellent model performance. Consider deployment for production use.\n"
        )
    elif metrics["accuracy"] > 0.90:
        report += "- Good model performance. Minor optimizations may improve results.\n"
    else:
        report += "- Model performance needs improvement. Consider architecture changes or more data.\n"

    if np.std(metrics["f1"]) > 0.1:
        report += "- High variance in per-class performance. Consider class balancing techniques.\n"

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Performance report saved to: {save_path}")

    return report
