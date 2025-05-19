import os
import argparse
import torch

from train import train_model
from hyperparameter_search import hyperparameter_search
from visualize import plot_sample_predictions, create_model_report
from dataset import get_data_loaders, get_class_names
from model import get_model


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition with CNN")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "hyperparam", "test", "predict"],
        help="Mode to run: train, hyperparam, test, or predict",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Path to save results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (for test/predict mode)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (for train mode)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (for train/test mode)"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden layer size (for train mode)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Max epochs (for train mode)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "train":
        print("Training model with specified configuration...")
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "max_epochs": args.epochs,
            "patience": 5,
        }

        print("Training configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        model, performance = train_model(args.data_dir, args.output_dir, config)
        create_model_report(performance, args.output_dir)

    elif args.mode == "hyperparam":
        print("Running hyperparameter search...")
        best_config = hyperparameter_search(args.data_dir, args.output_dir)

        print("\nRetraining model with best configuration...")
        final_config = {
            "learning_rate": best_config["learning_rate"],
            "hidden_size": best_config["hidden_size"],
            "batch_size": best_config["batch_size"],
            "max_epochs": args.epochs,
            "patience": 5,
        }

        final_output_dir = os.path.join(args.output_dir, "final_model")
        model, performance = train_model(args.data_dir, final_output_dir, final_config)
        create_model_report(performance, final_output_dir)

    elif args.mode == "test":
        if args.model_path is None:
            args.model_path = os.path.join(args.output_dir, "best_model.pth")
            print(f"No model path specified, using: {args.model_path}")

        print(f"Testing model from: {args.model_path}")

        # Load model
        model = get_model(num_classes=10, hidden_size=args.hidden_size).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        # Get test data loader
        data_loaders = get_data_loaders(args.data_dir, batch_size=args.batch_size)
        test_loader = data_loaders["test"]

        # Test model
        from train import test

        test_metrics = test(model, test_loader, device)

        # Print test results
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Average Precision: {test_metrics['avg_precision']:.4f}")
        print(f"Average Recall: {test_metrics['avg_recall']:.4f}")
        print(f"Average F1-Score: {test_metrics['avg_f1']:.4f}")

        # Plot confusion matrix and class metrics
        from visualize import plot_confusion_matrix, plot_class_metrics

        class_names = list(get_class_names().values())
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names=class_names,
            output_dir=args.output_dir,
        )

        plot_class_metrics(
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            class_names=class_names,
            output_dir=args.output_dir,
        )

    elif args.mode == "predict":
        if args.model_path is None:
            args.model_path = os.path.join(args.output_dir, "best_model.pth")
            print(f"No model path specified, using: {args.model_path}")

        print(f"Loading model from: {args.model_path}")

        # Load model
        model = get_model(num_classes=10, hidden_size=args.hidden_size).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        # Get test data loader
        data_loaders = get_data_loaders(args.data_dir, batch_size=args.batch_size)
        test_loader = data_loaders["test"]

        # Plot sample predictions
        class_names = get_class_names()
        plot_sample_predictions(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            output_dir=args.output_dir,
            num_samples=5,
        )
        print(f"Sample predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
