import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime

from src.utils.helpers import load_config, set_seed, ensure_dir, Timer
from src.utils.logger import get_logger, TensorboardWriter
from src.utils.metrics import accuracy, precision, recall, f1
from src.data_loader import make_dataloaders
from src.models.cnn_lstm import CNNLSTM

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc='Training')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100 * correct / total
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Store outputs and targets for metrics
            running_loss += loss.item() * batch_size
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate loss
    total = len(all_targets)
    val_loss = running_loss / total
    
    # Calculate metrics
    val_acc = accuracy(all_outputs, all_targets) * 100
    val_precision = precision(all_outputs, all_targets) * 100
    val_recall = recall(all_outputs, all_targets) * 100
    val_f1 = f1(all_outputs, all_targets) * 100
    
    metrics = {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1
    }
    
    return metrics

def get_model(config, n_classes):
    """Get model based on config"""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'cnn_lstm':
        cnn_filters = model_config['cnn_lstm']['cnn_filters']
        lstm_hidden = model_config['cnn_lstm']['lstm_hidden']
        lstm_layers = model_config['cnn_lstm']['lstm_layers']
        dropout = model_config['cnn_lstm']['dropout']
        img_size = config['data']['img_size']
        
        return CNNLSTM(
            cnn_filters=cnn_filters,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_classes=n_classes,
            img_size=img_size,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train(config, experiment_dir, logger, device):
    """
    Train model
    
    Args:
        config: Configuration dictionary
        experiment_dir: Directory to save experiment results
        logger: Logger instance
        device: Device to train on
    """
    # Create data loaders
    train_loader, val_loader, _ = make_dataloaders(config)
    
    # Get class names and number of classes
    n_classes = len(train_loader.dataset.class_names)
    class_names = train_loader.dataset.class_names
    
    logger.info(f"Number of classes: {n_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Create model
    model = get_model(config, n_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True
    )
    
    # Tensorboard writer
    tb_logdir = os.path.join(experiment_dir, 'tensorboard')
    writer = TensorboardWriter(tb_logdir)
    
    # Training loop
    epochs = config['training']['epochs']
    logger.info(f"Starting training for {epochs} epochs")
    
    # Track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Start training
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        
        # Train epoch
        with Timer() as train_timer:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        with Timer() as val_timer:
            val_metrics = validate(model, val_loader, criterion, device)
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Log to tensorboard
        writer.set_step(epoch)
        writer.add_scalar('loss/train', train_loss)
        writer.add_scalar('loss/val', val_loss)
        writer.add_scalar('accuracy/train', train_acc)
        writer.add_scalar('accuracy/val', val_acc)
        writer.add_scalar('precision/val', val_metrics['precision'])
        writer.add_scalar('recall/val', val_metrics['recall'])
        writer.add_scalar('f1/val', val_metrics['f1'])
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered after epoch {epoch}")
            break
        
        # Save checkpoint
        checkpoint_path = os.path.join(experiment_dir, 'checkpoints', f'model_epoch_{epoch}.pth')
        ensure_dir(os.path.dirname(checkpoint_path))
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save best model
        if val_loss == early_stopping.best_loss:
            best_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model checkpoint to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(experiment_dir, 'checkpoints', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model checkpoint to {final_model_path}")
    
    # Save learning curves
    from src.visualize import plot_learning_curves
    curves_path = os.path.join(experiment_dir, 'learning_curves.png')
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    logger.info(f"Saved learning curves to {curves_path}")
    
    # Close tensorboard writer
    writer.close()
    
    return model, val_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Directory to save experiment results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    if args.experiment_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = config['model']['type']
        args.experiment_dir = f"experiments/{model_type}_{timestamp}"
    
    ensure_dir(args.experiment_dir)
    
    # Create logger
    log_dir = os.path.join(args.experiment_dir, 'logs')
    logger = get_logger(name='train', log_dir=log_dir)
    
    # Log arguments and config
    logger.info(f"Arguments: {args}")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Train model
    try:
        model, val_metrics = train(config, args.experiment_dir, logger, device)
        logger.info(f"Final validation metrics: {val_metrics}")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
