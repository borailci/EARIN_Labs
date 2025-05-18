import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path

from src.utils.helpers import load_config, set_seed
from src.data_loader import make_dataloaders
from src.utils.metrics import get_metrics, get_confusion_matrix
from src.models.cnn_lstm import CNNLSTM

def get_model(config, n_classes):
    """Factory function to create model based on config"""
    mcfg = config['model']
    if mcfg['type'] == 'cnn_lstm':
        return CNNLSTM(
            cnn_filters=mcfg['cnn_lstm']['cnn_filters'],
            lstm_hidden=mcfg['cnn_lstm']['lstm_hidden'],
            lstm_layers=mcfg['cnn_lstm']['lstm_layers'],
            n_classes=n_classes,
            img_size=config['data']['img_size'],
            dropout=mcfg['cnn_lstm']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {mcfg['type']}")

def evaluate(model, data_loader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_outputs = np.vstack(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    return all_outputs, all_targets

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get data loaders
    _, _, test_loader = make_dataloaders(config)
    
    # Get class names
    class_names = test_loader.dataset.class_names
    n_classes = len(class_names)
    
    # Create model
    model = get_model(config, n_classes)
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Evaluate model
    outputs, targets = evaluate(model, test_loader, device)
    
    # Compute metrics
    metrics = get_metrics(outputs, targets, class_names)
    confusion = get_confusion_matrix(outputs, targets)
    
    # Print results
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    
    # Save results
    results = {
        'metrics': metrics,
        'confusion_matrix': confusion.tolist()
    }
    
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Evaluation results saved to {output_file}")
    
    # Optional: Generate visualizations
    from src.visualize import plot_confusion_matrix, plot_class_metrics
    
    figure_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(confusion, class_names, figure_path)
    
    metrics_path = os.path.join(args.output_dir, 'class_metrics.png')
    if 'per_class' in metrics:
        plot_class_metrics(
            metrics['per_class']['precision'],
            metrics['per_class']['recall'],
            metrics['per_class']['f1'],
            class_names,
            metrics_path
        )

if __name__ == '__main__':
    main()
