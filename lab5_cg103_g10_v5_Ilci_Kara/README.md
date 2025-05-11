# EARIN Lab 5: Artificial Neural Networks

## Project Description

This project implements a multilayer perceptron (MLP) for image classification using the KMNIST dataset. The neural network is trained with the mini-batch gradient descent method, and the codebase evaluates the effects of various hyperparameters on model performance.

## Features

- MLP implementation with configurable architecture
- Custom training loop for model training
- Evaluation of multiple hyperparameters:
  - Learning rates: 0.001, 0.01, 0.1
  - Mini-batch sizes: 1, 32, 128
  - Number of hidden layers: 0 (linear model), 1, 3
  - Width (neurons per layer): 64, 128, 256
  - Optimizer types: SGD, SGD with momentum, Adam
- Visualization of training and validation metrics
- Comprehensive analysis of results

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- pandas
- seaborn

## Installation and Setup

### Option 1: Using Virtual Environment (Recommended)

```bash
# Run the setup script to create a virtual environment and install dependencies
python setup.py

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Option 2: Direct Installation

```bash
pip install torch torchvision numpy matplotlib pandas seaborn
```

## Usage

1. Run the main script to train and evaluate the models:

```bash
python mlp_kmnist.py
```

2. After the experiments are completed, analyze the results:

```bash
python analyze_results.py
```

3. View the generated plots and results in the `results` and `analysis` directories.

## Project Structure

- `mlp_kmnist.py`: Main script for MLP implementation and hyperparameter experimentation
- `analyze_results.py`: Script for analyzing and visualizing experiment results
- `setup.py`: Script to set up a virtual environment and install dependencies
- `results/`: Directory containing raw experiment results and per-experiment plots
- `analysis/`: Directory containing comparative analysis plots and summaries
- `models/`: Directory containing saved model weights
- `data/`: Directory where the KMNIST dataset will be downloaded
- `venv/`: Virtual environment directory (created by setup.py)

## Implementation Details

- The MLP architecture is configurable in terms of hidden layers and layer width
- Training is performed with a custom mini-batch gradient descent implementation
- Data is split into training (80%) and validation (20%) sets
- Loss and accuracy are tracked for each epoch
- All experiments are conducted with the same random seed for reproducibility

## Output

- Training/validation plots for each experiment configuration
- Comparative plots showing the effect of each hyperparameter
- Summary CSV file with accuracy and training time metrics

## Authors

- Bora Ilci
- Kaan Emre Kara
