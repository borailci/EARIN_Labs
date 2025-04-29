import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def analyze_results():
    # Load results from CSV
    results_file = "results/experiment_results.csv"
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found!")
        return

    results = pd.read_csv(results_file)
    print(f"Loaded {len(results)} experiment results")

    # Create results directory if it doesn't exist
    os.makedirs("analysis", exist_ok=True)

    # Group results by varied parameter and create comparison plots

    # 1. Learning rate comparison
    lr_results = results[results["batch_size"] == 32][results["hidden_layers"] == 1][
        results["hidden_size"] == 128
    ][results["optimizer_type"] == "sgd_momentum"]

    if not lr_results.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="learning_rate", y="val_accuracy", data=lr_results)
        plt.title("Effect of Learning Rate on Validation Accuracy")
        plt.xlabel("Learning Rate")
        plt.ylabel("Validation Accuracy")
        plt.savefig("analysis/learning_rate_comparison.png")
        plt.close()

        # Training time comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="learning_rate", y="training_time", data=lr_results)
        plt.title("Effect of Learning Rate on Training Time")
        plt.xlabel("Learning Rate")
        plt.ylabel("Training Time (seconds)")
        plt.savefig("analysis/learning_rate_time_comparison.png")
        plt.close()

    # 2. Batch size comparison
    bs_results = results[results["learning_rate"] == 0.01][
        results["hidden_layers"] == 1
    ][results["hidden_size"] == 128][results["optimizer_type"] == "sgd_momentum"]

    if not bs_results.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="batch_size", y="val_accuracy", data=bs_results)
        plt.title("Effect of Batch Size on Validation Accuracy")
        plt.xlabel("Batch Size")
        plt.ylabel("Validation Accuracy")
        plt.savefig("analysis/batch_size_comparison.png")
        plt.close()

        # Training time comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="batch_size", y="training_time", data=bs_results)
        plt.title("Effect of Batch Size on Training Time")
        plt.xlabel("Batch Size")
        plt.ylabel("Training Time (seconds)")
        plt.savefig("analysis/batch_size_time_comparison.png")
        plt.close()

    # 3. Hidden layers comparison
    hl_results = results[results["learning_rate"] == 0.01][results["batch_size"] == 32][
        results["hidden_size"] == 128
    ][results["optimizer_type"] == "sgd_momentum"]

    if not hl_results.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="hidden_layers", y="val_accuracy", data=hl_results)
        plt.title("Effect of Hidden Layers on Validation Accuracy")
        plt.xlabel("Number of Hidden Layers")
        plt.ylabel("Validation Accuracy")
        plt.savefig("analysis/hidden_layers_comparison.png")
        plt.close()

    # 4. Hidden size comparison
    hs_results = results[results["learning_rate"] == 0.01][results["batch_size"] == 32][
        results["hidden_layers"] == 1
    ][results["optimizer_type"] == "sgd_momentum"]

    if not hs_results.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="hidden_size", y="val_accuracy", data=hs_results)
        plt.title("Effect of Hidden Size on Validation Accuracy")
        plt.xlabel("Hidden Size (neurons per layer)")
        plt.ylabel("Validation Accuracy")
        plt.savefig("analysis/hidden_size_comparison.png")
        plt.close()

    # 5. Optimizer type comparison
    opt_results = results[results["learning_rate"] == 0.01][
        results["batch_size"] == 32
    ][results["hidden_layers"] == 1][results["hidden_size"] == 128]

    if not opt_results.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="optimizer_type", y="val_accuracy", data=opt_results)
        plt.title("Effect of Optimizer Type on Validation Accuracy")
        plt.xlabel("Optimizer Type")
        plt.ylabel("Validation Accuracy")
        plt.savefig("analysis/optimizer_comparison.png")
        plt.close()

    # 6. Train vs Validation accuracy comparison for all models
    plt.figure(figsize=(12, 8))

    x = np.arange(len(results))
    width = 0.35

    plt.bar(x - width / 2, results["train_accuracy"], width, label="Train Accuracy")
    plt.bar(x + width / 2, results["val_accuracy"], width, label="Validation Accuracy")

    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy for All Models")
    plt.xticks(x, [f"Exp {i+1}" for i in range(len(results))], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/train_val_comparison.png")
    plt.close()

    # Create summary table
    summary = results[
        [
            "experiment_name",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "training_time",
        ]
    ]
    summary = summary.sort_values(by="val_accuracy", ascending=False)

    # Save summary to CSV
    summary.to_csv("analysis/summary.csv", index=False)

    print("Analysis completed! Results saved to the 'analysis' directory.")


if __name__ == "__main__":
    analyze_results()
