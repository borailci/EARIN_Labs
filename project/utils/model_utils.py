"""
Model Utility Functions for Hand Gesture Recognition

This module provides utility functions for model analysis, visualization, and
performance evaluation. These functions support the academic research workflow
by providing comprehensive model introspection and analysis capabilities.

Functions:
    - calculate_model_complexity: Analyze model parameters and computational requirements
    - visualize_model_architecture: Create visual representations of model structure
    - benchmark_inference_time: Measure and analyze model inference performance
    - analyze_feature_maps: Extract and visualize intermediate feature representations
    - model_summary_report: Generate comprehensive model analysis report

Author: Course Project Team
Date: Academic Year 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import time
from typing import Dict, List, Tuple, Any
import pandas as pd


def calculate_model_complexity(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 64, 64)
) -> Dict[str, Any]:
    """
    Calculate comprehensive model complexity metrics.

    Args:
        model: PyTorch model to analyze
        input_shape: Input tensor shape for FLOP calculation

    Returns:
        Dictionary containing complexity metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # Estimate FLOPs (simplified calculation)
    flops = estimate_flops(model, input_shape)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "estimated_flops": flops,
        "parameters_per_class": total_params / 10,  # Assuming 10 classes
    }


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for the model (simplified calculation).

    Args:
        model: PyTorch model
        input_shape: Input tensor shape

    Returns:
        Estimated FLOPs
    """
    model.eval()
    flops = 0

    def conv_flop_count(module, input_shape):
        """Calculate FLOPs for convolutional layer."""
        if isinstance(module, nn.Conv2d):
            output_h = (
                input_shape[2] - module.kernel_size[0] + 2 * module.padding[0]
            ) // module.stride[0] + 1
            output_w = (
                input_shape[3] - module.kernel_size[1] + 2 * module.padding[1]
            ) // module.stride[1] + 1
            kernel_flops = (
                module.kernel_size[0] * module.kernel_size[1] * input_shape[1]
            )
            output_elements = output_h * output_w * module.out_channels
            return kernel_flops * output_elements
        return 0

    # Simplified FLOP calculation for major operations
    # This is a rough estimate - for precise calculations, use tools like ptflops
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            flops += conv_flop_count(module, input_shape)
            # Update input shape for next layer (simplified)
            input_shape = (
                input_shape[0],
                module.out_channels,
                input_shape[2] // 2,
                input_shape[3] // 2,
            )  # Assuming pooling

    return flops


def benchmark_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 64, 64),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference time with statistical analysis.

    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs to exclude

    Returns:
        Dictionary containing timing statistics
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    times = np.array(times)

    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "median_time_ms": np.median(times),
        "fps_estimate": 1000 / np.mean(times),
    }


def analyze_layer_outputs(
    model: nn.Module, input_tensor: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate layer outputs for analysis.

    Args:
        model: PyTorch model
        input_tensor: Input tensor to pass through model

    Returns:
        Dictionary mapping layer names to their outputs
    """
    model.eval()
    layer_outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach()

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return layer_outputs


def create_model_summary_table(model: nn.Module) -> pd.DataFrame:
    """
    Create a detailed model summary table.

    Args:
        model: PyTorch model to analyze

    Returns:
        Pandas DataFrame with layer-wise model summary
    """
    summary_data = []

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module_type = module.__class__.__name__

            # Count parameters
            param_count = sum(p.numel() for p in module.parameters())
            trainable_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

            # Get module details
            details = str(module).replace("\n", " ").replace("  ", " ")
            if len(details) > 80:
                details = details[:77] + "..."

            summary_data.append(
                {
                    "Layer": name,
                    "Type": module_type,
                    "Parameters": param_count,
                    "Trainable": trainable_params,
                    "Details": details,
                }
            )

    return pd.DataFrame(summary_data)


def save_model_analysis_report(
    model: nn.Module, save_path: str, input_shape: Tuple[int, ...] = (1, 1, 64, 64)
):
    """
    Generate and save a comprehensive model analysis report.

    Args:
        model: PyTorch model to analyze
        save_path: Path to save the analysis report
        input_shape: Input tensor shape for analysis
    """
    # Calculate complexity metrics
    complexity = calculate_model_complexity(model, input_shape)

    # Benchmark inference time
    timing = benchmark_inference_time(model, input_shape)

    # Create model summary
    summary_df = create_model_summary_table(model)

    # Create report
    report = f"""
# Model Analysis Report

## Model Complexity
- Total Parameters: {complexity['total_parameters']:,}
- Trainable Parameters: {complexity['trainable_parameters']:,}
- Model Size: {complexity['model_size_mb']:.2f} MB
- Estimated FLOPs: {complexity['estimated_flops']:,}

## Performance Metrics
- Average Inference Time: {timing['mean_time_ms']:.2f} ± {timing['std_time_ms']:.2f} ms
- Estimated FPS: {timing['fps_estimate']:.1f}
- Min/Max Time: {timing['min_time_ms']:.2f}/{timing['max_time_ms']:.2f} ms

## Layer Summary
{summary_df.to_string(index=False)}

## Architecture Visualization
The model follows a progressive channel expansion pattern:
Input (1 channel) → Conv Block 1 (64 channels) → Conv Block 2 (128 channels) → Conv Block 3 (256 channels) → Classifier

Each convolutional block includes:
- Two convolutional layers with batch normalization
- ReLU activation functions
- Dropout regularization
- Max pooling for spatial dimension reduction
"""

    with open(save_path, "w") as f:
        f.write(report)

    print(f"Model analysis report saved to: {save_path}")
