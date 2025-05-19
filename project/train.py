import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score

from dataset import get_data_loaders
from model import get_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc


def train_model(data_dir, output_dir, config):
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get data loaders
    data_loaders = get_data_loaders(data_dir, batch_size=config["batch_size"])

    # Initialize model
    model = get_model(
        num_classes=10, hidden_size=config["hidden_size"], dropout_rate=0.5
    ).to(device)
    print(f"Model initialized")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # Initialize training variables
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    print("Starting training...")

    try:
        for epoch in range(config["max_epochs"]):
            # Train and validate
            train_loss, train_acc = train_one_epoch(
                model, data_loaders["train"], criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, data_loaders["val"], criterion, device)

            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Print epoch summary
            print(f"Epoch {epoch+1}/{config['max_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}")
            print(f"Learning Rate: {current_lr:.2e}")
            print("-" * 60)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_model.pth")
                )
                print(f"Saved best model with validation accuracy: {val_acc:.2%}")
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        # Print final summary
        total_time = time.time() - start_time
        print("\nTraining Complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {os.path.join(output_dir, 'best_model.pth')}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a CNN for hand gesture recognition"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Path to save output"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )

    args = parser.parse_args()

    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
    }

    train_model(args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
