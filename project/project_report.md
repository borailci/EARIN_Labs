# Hand Gesture Recognition System - Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Training System](#training-system)
5. [Real-time Recognition](#real-time-recognition)
6. [Testing and Evaluation](#testing-and-evaluation)
7. [Data Capture System](#data-capture-system)
8. [System Utilities](#system-utilities)
9. [Performance Optimizations](#performance-optimizations)
10. [Project Structure](#project-structure)
11. [Dependencies and Environment](#dependencies-and-environment)

## System Overview

The Hand Gesture Recognition System is a deep learning-based solution for real-time hand gesture recognition using a webcam. The system can recognize 10 different hand gestures with high accuracy and low latency.

### Key Features

- Real-time gesture recognition
- High accuracy (99.82% validation accuracy)
- Low latency processing
- Robust to variations in hand position and lighting
- Comprehensive testing and visualization tools
- Custom dataset generation capabilities

## Model Architecture

### CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256, dropout_rate=0.3):
        # Input: 1x64x64 (grayscale)

        # Block 1: 1 → 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 64x64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64x64
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32

        # Block 2: 64 → 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 32x32
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16

        # Block 3: 128 → 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 16x16
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 16x16
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, hidden_size)  # 16384 → 256
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 256 → 128
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # 128 → 10
```

### Model Specifications

- Input: 64x64 grayscale images
- Architecture: 3 convolutional blocks with double conv layers
- Channel progression: 1 → 64 → 128 → 256
- Spatial dimensions: 64 → 32 → 16 → 8
- Fully connected layers: 16384 → 256 → 128 → 10
- Total parameters: ~2.5M
- Model size: 4.4MB (regular) / 12MB (best models)

## Data Processing Pipeline

### Dataset Structure

```
custom_data/
├── 00/                  # Main folder 0
│   ├── 01_palm/        # Gesture 1: palm
│   │   ├── frame_00_01_0001.png
│   │   ├── frame_00_01_0002.png
│   │   └── ...
│   ├── 02_l/           # Gesture 2: l
│   └── ...
├── 01/                 # Main folder 1
└── ...
```

### Data Augmentation

```python
train_transform = transforms.Compose([
    # Resize to model input size
    transforms.Resize((64, 64)),

    # Geometric transformations
    transforms.RandomApply([
        transforms.RandomRotation(15),  # ±15 degrees
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),      # 10% translation
            scale=(0.9, 1.1),          # 10% scaling
            shear=10                   # 10 degrees shear
        )
    ], p=0.5),                        # 50% probability

    # Horizontal flip
    transforms.RandomHorizontalFlip(p=0.3),  # 30% probability

    # Blur and noise
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)          # Variable blur strength
        )
    ], p=0.2),                        # 20% probability

    # Color adjustments
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2,           # 20% brightness variation
            contrast=0.2              # 20% contrast variation
        )
    ], p=0.2),                        # 20% probability

    # Normalization
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Mean=0.5, Std=0.5
])
```

### Data Splitting

- Training set: 70%
- Validation set: 15%
- Test set: 15%
- Stratified splitting to maintain class distribution

## Training System

### Training Configuration

```python
# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,           # Learning rate
    weight_decay=1e-4    # L2 regularization
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Monitor validation loss
    factor=0.5,          # Halve learning rate
    patience=2,          # Wait 2 epochs
    min_lr=1e-6,         # Minimum learning rate
    threshold=1e-4       # Minimum change threshold
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Mixed precision training
scaler = GradScaler()
```

### Training Progress

```json
{
    "train_loss": [0.8442, 0.2052, 0.1231],
    "train_acc": [77.00%, 94.30%, 96.49%],
    "val_loss": [0.0820, 0.0205, 0.0093],
    "val_acc": [99.39%, 99.58%, 99.82%],
    "learning_rates": [0.0005, 0.0005, 0.0005]
}
```

## Real-time Recognition

### Hand Detection

- Framework: MediaPipe
- Parameters:
  - Static image mode: False
  - Max hands: 1
  - Detection confidence: 0.5
  - Tracking confidence: 0.5

### Prediction Pipeline

1. Frame Processing:

   - Hand ROI extraction
   - Grayscale conversion
   - Resize to 64x64
   - Normalization

2. Prediction Smoothing:
   - History buffer size: 5
   - Confidence threshold: 0.7
   - Cooldown period: 0.5s
   - Minimum agreement: 60%

### Performance Optimizations

1. Memory Management:

   - Gradient checkpointing
   - Mixed precision inference
   - Efficient data loading
   - Memory cleanup routines

2. Real-time Optimizations:
   - Non-blocking tensor operations
   - Efficient hand detection
   - Prediction history smoothing
   - Frame skipping during cooldown

## Testing and Evaluation

### Test Visualizer

```python
class TestVisualizer:
    def __init__(self, model_path, custom_data_dir):
        # Initialize model and transforms
        self.model = get_model(num_classes=10, hidden_size=128)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
```

### Evaluation Metrics

1. Confusion Matrix
2. Confidence Histogram
3. Class-wise Accuracy
4. Sample Predictions
5. Error Analysis
6. Large Grid Visualization

## Data Capture System

### Dataset Generator

```python
class DatasetGenerator:
    def __init__(self, output_dir, gestures, samples_per_gesture=100, countdown=3):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
```

### Capture Parameters

- Samples per gesture: 100
- Countdown timer: 3 seconds
- Hand detection confidence: 0.7
- Tracking confidence: 0.7
- Capture delay: 0.1 seconds

## System Utilities

### Memory Management

```python
def clear_memory():
    # Clear Python garbage collector
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clear system cache
    if os.name == "posix":
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1")
```

### Process Management

```python
def kill_python_processes():
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
        if proc.info["name"] == "python" and proc.info["pid"] != current_pid:
            if proc.info["cpu_percent"] > 1.0:
                os.kill(proc.info["pid"], signal.SIGTERM)
```

## Project Structure

```
project/
├── model.py              # CNN architecture
├── dataset.py           # Dataset handling
├── train.py             # Training pipeline
├── realtime_recognition.py  # Real-time system
├── capture_dataset.py   # Data capture
├── visualize_test.py    # Testing visualization
├── cleanup.py          # System utilities
├── generate_dataset.py  # Dataset generation
├── visualize.py        # Visualization tools
├── results/            # Model checkpoints
│   ├── best_model.pth
│   ├── eniyisibu.pth
│   └── training_history.json
└── custom_data/        # Custom dataset
```

## Dependencies and Environment

### Core Dependencies

```
# Core
numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=8.0.0
opencv-python>=4.5.0
mediapipe>=0.8.0
scikit-learn>=0.24.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
psutil>=5.9.0
```

### System Requirements

- Python 3.8+
- CUDA-capable GPU (optional)
- Webcam
- 8GB+ RAM
- 1GB+ free disk space

### Environment Setup

1. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
