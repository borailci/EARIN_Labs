import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import random
import os
from datetime import datetime
import csv


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load KMNIST dataset
def load_kmnist_data(batch_size):
    # Define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the KMNIST dataset
    train_dataset = torchvision.datasets.KMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torchvision.datasets.KMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, output_size):
        super(MLP, self).__init__()

        layers = []

        # Input layer
        if hidden_layers == 0:
            # No hidden layers, just input -> output (linear model)
            layers.append(nn.Linear(input_size, output_size))
        else:
            # Add input layer to first hidden layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())

            # Add hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())

            # Add output layer
            layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        return self.network(x)


# Training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs, experiment_name
):
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save step loss for detailed tracking
            train_losses.append(loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return train_losses, train_accuracies, val_accuracies, training_time


# Evaluate the model on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# Plot training curves
def plot_curves(
    train_losses, train_accuracies, val_accuracies, experiment_name, save_path="results"
):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Plot loss curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f"Training Loss - {experiment_name}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    epochs = len(train_accuracies)
    x = np.arange(1, epochs + 1)
    plt.plot(x, train_accuracies, label="Train Accuracy")
    plt.plot(x, val_accuracies, label="Validation Accuracy")
    plt.title(f"Accuracy - {experiment_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_path}/{experiment_name}_{timestamp}.png")
    plt.close()


# Run experiment with different parameters
def run_experiment(
    learning_rate, batch_size, hidden_layers, hidden_size, optimizer_type, epochs=10
):
    # Load data
    train_loader, val_loader, test_loader = load_kmnist_data(batch_size)

    # Create model
    input_size = 28 * 28  # KMNIST images are 28x28
    output_size = 10  # 10 classes in KMNIST
    model = MLP(input_size, hidden_layers, hidden_size, output_size)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Experiment name
    experiment_name = f"lr{learning_rate}_bs{batch_size}_hl{hidden_layers}_hs{hidden_size}_{optimizer_type}"

    # Train model
    train_losses, train_accuracies, val_accuracies, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, experiment_name
    )

    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader)

    # Plot and save curves
    plot_curves(train_losses, train_accuracies, val_accuracies, experiment_name)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{experiment_name}.pth")

    # Return results for comparison
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "hidden_size": hidden_size,
        "optimizer_type": optimizer_type,
        "train_accuracy": train_accuracies[-1],
        "val_accuracy": val_accuracies[-1],
        "test_accuracy": test_accuracy,
        "training_time": training_time,
        "experiment_name": experiment_name,
    }


# Run all experiments
def run_experiments():
    # Define the variations to test
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [1, 32, 128]
    hidden_layers_options = [0, 1, 3]  # 0 = linear model
    hidden_sizes = [64, 128, 256]
    optimizer_types = ["sgd", "sgd_momentum", "adam"]

    # Default configuration (for experiments where we vary just one parameter)
    default_lr = 0.01
    default_bs = 32
    default_hl = 1
    default_hs = 128
    default_opt = "sgd_momentum"
    default_epochs = 10

    results = []

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # 1. Vary learning rate
    print("\n--- VARYING LEARNING RATE ---")
    for lr in learning_rates:
        print(f"\nExperiment with learning rate = {lr}")
        result = run_experiment(
            learning_rate=lr,
            batch_size=default_bs,
            hidden_layers=default_hl,
            hidden_size=default_hs,
            optimizer_type=default_opt,
            epochs=default_epochs,
        )
        results.append(result)

    # 2. Vary batch size
    print("\n--- VARYING BATCH SIZE ---")
    for bs in batch_sizes:
        print(f"\nExperiment with batch size = {bs}")
        result = run_experiment(
            learning_rate=default_lr,
            batch_size=bs,
            hidden_layers=default_hl,
            hidden_size=default_hs,
            optimizer_type=default_opt,
            epochs=default_epochs,
        )
        results.append(result)

    # 3. Vary number of hidden layers
    print("\n--- VARYING HIDDEN LAYERS ---")
    for hl in hidden_layers_options:
        print(f"\nExperiment with hidden layers = {hl}")
        result = run_experiment(
            learning_rate=default_lr,
            batch_size=default_bs,
            hidden_layers=hl,
            hidden_size=default_hs,
            optimizer_type=default_opt,
            epochs=default_epochs,
        )
        results.append(result)

    # 4. Vary hidden size
    print("\n--- VARYING HIDDEN SIZE ---")
    for hs in hidden_sizes:
        print(f"\nExperiment with hidden size = {hs}")
        result = run_experiment(
            learning_rate=default_lr,
            batch_size=default_bs,
            hidden_layers=default_hl,
            hidden_size=hs,
            optimizer_type=default_opt,
            epochs=default_epochs,
        )
        results.append(result)

    # 5. Vary optimizer type
    print("\n--- VARYING OPTIMIZER TYPE ---")
    for opt in optimizer_types:
        print(f"\nExperiment with optimizer = {opt}")
        result = run_experiment(
            learning_rate=default_lr,
            batch_size=default_bs,
            hidden_layers=default_hl,
            hidden_size=default_hs,
            optimizer_type=opt,
            epochs=default_epochs,
        )
        results.append(result)

    # Save experiment results to CSV
    with open("results/experiment_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(
        "\nAll experiments completed. Results saved to results/experiment_results.csv"
    )


# Analyze experiment results
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


def main():
    set_seed(42)
    print(f"Using device: {device}")

    # Choose what to do
    print("\nWhat would you like to do?")
    print("1. Run experiments (train models)")
    print("2. Analyze results")
    print("3. Run both experiments and analysis")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_experiments()
    elif choice == "2":
        analyze_results()
    elif choice == "3":
        run_experiments()
        analyze_results()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
