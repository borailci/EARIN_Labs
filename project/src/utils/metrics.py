import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def accuracy(outputs, targets):
    """Calculate accuracy between predicted outputs and targets"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # For softmax outputs
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
    
    return accuracy_score(targets, outputs)

def precision(outputs, targets, average='macro'):
    """Calculate precision between predicted outputs and targets"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # For softmax outputs
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
    
    return precision_score(targets, outputs, average=average, zero_division=0)

def recall(outputs, targets, average='macro'):
    """Calculate recall between predicted outputs and targets"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # For softmax outputs
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
    
    return recall_score(targets, outputs, average=average, zero_division=0)

def f1(outputs, targets, average='macro'):
    """Calculate F1 score between predicted outputs and targets"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # For softmax outputs
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
    
    return f1_score(targets, outputs, average=average, zero_division=0)

def get_confusion_matrix(outputs, targets, normalize=None):
    """Get confusion matrix between predicted outputs and targets"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    
    # For softmax outputs
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
    
    return confusion_matrix(targets, outputs, normalize=normalize)

def get_metrics(outputs, targets, class_names=None):
    """Get a dictionary of metrics"""
    metrics = {
        'accuracy': accuracy(outputs, targets) * 100,
        'precision': precision(outputs, targets) * 100,
        'recall': recall(outputs, targets) * 100,
        'f1': f1(outputs, targets) * 100
    }
    
    # Add per-class metrics if class names are provided
    if class_names is not None:
        per_class_precision = precision(outputs, targets, average=None) * 100
        per_class_recall = recall(outputs, targets, average=None) * 100
        per_class_f1 = f1(outputs, targets, average=None) * 100
        
        metrics['per_class'] = {
            'class_names': class_names,
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1': per_class_f1
        }
    
    return metrics
