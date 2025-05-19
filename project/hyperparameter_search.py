import os
import json
import torch
import itertools
from tqdm import tqdm
import pandas as pd
from train import train_model


def hyperparameter_search(data_dir, output_dir):
    """
    Perform hyperparameter search over specified configurations

    Args:
        data_dir (str): Path to data directory
        output_dir (str): Path to save output
    """
    # Create output directory for hyperparameter search
    hp_search_dir = os.path.join(output_dir, "hyperparameter_search")
    os.makedirs(hp_search_dir, exist_ok=True)

    # Define hyperparameter search space
    learning_rates = [1e-2, 1e-3, 1e-4]
    hidden_sizes = [64, 128, 256]
    batch_sizes = [16, 32]

    # Generate all combinations
    configs = list(itertools.product(learning_rates, hidden_sizes, batch_sizes))
    print(f"Total hyperparameter configurations to try: {len(configs)}")

    # Store results
    results = []

    # Run experiments
    for i, (lr, hidden_size, batch_size) in enumerate(configs):
        print(f"\nExperiment {i+1}/{len(configs)}")
        print(
            f"Learning Rate: {lr}, Hidden Size: {hidden_size}, Batch Size: {batch_size}"
        )

        config = {
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "max_epochs": 30,
            "patience": 5,
        }

        # Create directory for this experiment
        exp_dir = os.path.join(hp_search_dir, f"exp_{i+1}")
        os.makedirs(exp_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Run training with this configuration
        _, performance = train_model(data_dir, exp_dir, config)

        # Add experiment info
        performance["experiment_id"] = i + 1
        performance["learning_rate"] = lr
        performance["hidden_size"] = hidden_size
        performance["batch_size"] = batch_size

        # Store results
        results.append(performance)

        # Save results so far
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(hp_search_dir, "results.csv"), index=False)

    # Find best configuration based on F1 score
    best_idx = pd.DataFrame(results)["f1"].idxmax()
    best_config = results[best_idx]

    print("\nHyperparameter search completed!")
    print(f"Best configuration (Experiment {best_config['experiment_id']}):")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Hidden Size: {best_config['hidden_size']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  F1 Score: {best_config['f1']:.4f}")
    print(f"  Accuracy: {best_config['accuracy']:.4f}")

    # Compare all configurations
    print("\nAll configurations:")
    for result in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(
            f"  Exp {result['experiment_id']}: LR={result['learning_rate']}, "
            f"Hidden={result['hidden_size']}, Batch={result['batch_size']}, "
            f"F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}"
        )

    # Save best configuration
    with open(os.path.join(hp_search_dir, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=4)

    return best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter search for hand gesture recognition"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Path to save output"
    )

    args = parser.parse_args()

    print("Starting hyperparameter search...")
    best_config = hyperparameter_search(args.data_dir, args.output_dir)

    print("\nTraining final model with best configuration...")
    final_config = {
        "learning_rate": best_config["learning_rate"],
        "hidden_size": best_config["hidden_size"],
        "batch_size": best_config["batch_size"],
        "max_epochs": 30,
        "patience": 5,
    }

    # Train final model with best configuration
    final_output_dir = os.path.join(args.output_dir, "final_model")
    train_model(args.data_dir, final_output_dir, final_config)
