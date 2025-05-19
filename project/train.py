import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_data_loaders
from model import get_model
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import gc
import psutil
import torch.cuda
from torch.cuda.amp import autocast, GradScaler
import time


def clear_memory():
    """Clear unused memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
        )
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2, 0


def plot_training_history(history, output_dir):
    """Plot training history including learning rates"""
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Plot learning rates
    plt.subplot(2, 2, 3)
    plt.semilogy(history["learning_rates"])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate (log scale)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get data loaders with optimized settings for 24GB memory
    num_workers = min(4, os.cpu_count())  # Reduced workers
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        custom_data_dir=args.custom_data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Initialize model with gradient checkpointing
    model = get_model(num_classes=10, hidden_size=args.hidden_size).to(device)
    model.use_checkpointing = True  # Enable gradient checkpointing
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, threshold=1e-4
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "best_val_loss": float("inf"),
    }

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_start_time = time.time()

        # Clear memory before each epoch
        clear_memory()
        mem_used, mem_reserved = get_memory_usage()
        print(f"Memory usage: {mem_used:.1f}MB used, {mem_reserved:.1f}MB reserved")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loop = tqdm(data_loaders["train"], desc="Training")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Use mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_loop.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*train_correct/train_total:.2f}%",
                }
            )

            # Clear memory after each batch
            del outputs, loss, predicted
            torch.cuda.empty_cache()  # More efficient than full clear_memory()

        train_loss = train_loss / len(data_loaders["train"])
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_loop = tqdm(data_loaders["val"], desc="Validation")
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                # Use mixed precision for validation
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Update progress bar
                val_loop.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100.*val_correct/val_total:.2f}%",
                    }
                )

                # Clear memory after each batch
                del outputs, loss, predicted
                torch.cuda.empty_cache()  # More efficient than full clear_memory()

        val_loss = val_loss / len(data_loaders["val"])
        val_acc = 100.0 * val_correct / val_total

        # Update learning rate
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Print learning rate change if it changed
        if new_lr != old_lr:
            print(f"\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}")
            print(
                f"Validation loss: {val_loss:.4f} (Previous best: {history['best_val_loss']:.4f})"
            )

        # Update best validation loss
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            print(f"New best validation loss: {val_loss:.4f}")

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {new_lr:.6f}")
        print(f"Best Val Loss: {history['best_val_loss']:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(new_lr)

        # Save history to file
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)

        # Plot training history
        plot_training_history(history, args.output_dir)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pth")
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Clear memory after each epoch
        clear_memory()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train hand gesture recognition model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to original dataset directory",
    )
    parser.add_argument(
        "--custom_data_dir",
        type=str,
        default="dataset/custom",
        help="Path to custom captured dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save model and results",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Maximum number of epochs"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
