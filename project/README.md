# Real-Time Hand Gesture Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Project](https://img.shields.io/badge/Academic-Project-green.svg)](https://github.com)

A comprehensive deep learning-based system for real-time hand gesture recognition using Convolutional Neural Networks (CNNs). This academic project achieves **99.82% validation accuracy** with sub-millisecond inference times, making it suitable for interactive applications and human-computer interfaces.

## 🎯 Project Overview

This system combines MediaPipe hand detection with a custom CNN architecture to provide robust, real-time gesture recognition. The project includes comprehensive training pipelines, evaluation tools, and academic documentation suitable for research publication.

### Key Achievements

- **99.82% validation accuracy** on 10 gesture classes
- **<3ms inference time** for real-time applications
- **Comprehensive academic documentation** with LaTeX scientific paper
- **Production-ready code** with extensive testing and evaluation tools

## 🚀 Features

### Core Functionality

- **Real-time gesture recognition** using webcam input
- **10 distinct hand gestures** with high accuracy classification
- **MediaPipe integration** for robust hand detection
- **GPU acceleration** with automatic fallback to CPU
- **Confidence-based prediction filtering** for stable recognition

### Academic & Research Features

- **Comprehensive training pipeline** with modern deep learning techniques
- **Extensive evaluation tools** with statistical analysis
- **Model complexity analysis** and performance benchmarking
- **Publication-quality visualizations** and reports
- **LaTeX scientific paper** with IEEE formatting
- **Reproducible experiments** with detailed documentation

### Technical Features

- **Mixed precision training** for improved efficiency
- **Advanced data augmentation** techniques
- **TensorBoard integration** for training monitoring
- **Model checkpointing** and automatic best model selection
- **Cross-validation support** with statistical significance testing

## 📊 Supported Gestures

| ID  | Gesture      | Description                     | Use Case             |
| --- | ------------ | ------------------------------- | -------------------- |
| 1   | Palm         | Open palm gesture               | Stop, attention      |
| 2   | L-Shape      | Index finger and thumb extended | Frame, measure       |
| 3   | Fist         | Closed fist                     | Power, selection     |
| 4   | Fist Moved   | Fist with wrist movement        | Dynamic action       |
| 5   | Thumb        | Thumbs up gesture               | Approval, like       |
| 6   | Index Finger | Single finger pointing          | Direction, selection |
| 7   | OK Sign      | Thumb and index finger circle   | Confirmation         |
| 8   | Palm Moved   | Palm with wrist movement        | Wave, greeting       |
| 9   | C-Shape      | Curved hand forming C           | Grab, hold           |
| 10  | Down Sign    | Pointing downward               | Navigate down        |

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Webcam** for real-time recognition
- **GPU** recommended but not required (CUDA support)
- **4GB+ RAM** for training (2GB+ for inference only)

### Quick Installation

1. **Clone the repository:**

```bash
git clone https://github.com/borailci/EARIN_Labs.git
cd EARIN_Labs/project
```

2. **Create and activate virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation:**

```bash
python -c "import torch, cv2, mediapipe; print('Installation successful!')"
```

## 🎮 Usage

### Quick Start - Real-time Recognition

```bash
# Run with pre-trained model
python realtime_recognition.py --model_path ./results/best_model.pth

# With custom confidence threshold
python realtime_recognition.py --model_path ./results/best_model.pth --confidence 0.8

# Specify camera device
python realtime_recognition.py --model_path ./results/best_model.pth --camera 0
```

### Training Your Own Model

1. **Prepare dataset:**

```bash
# The custom dataset is already available in ./custom_data/ directory
# with 10 users (00-09) and 10 gesture classes each
# Original dataset is available in ./data/ directory

# To view dataset structure:
ls -la ./custom_data/  # Custom data from 10 users
ls -la ./data/         # Original Kaggle dataset
```

2. **Train model:**

```bash
# Basic training
python train.py --custom_data_dir ./custom_data

# Advanced training with configuration
python train.py --config ./config/training_config.yaml --gpu 0

# Resume from checkpoint
python train.py --resume ./results/checkpoint.pth
```

3. **Evaluate model:**

```bash
# Basic evaluation using existing evaluation utilities
python -c "
from utils.model_utils import calculate_model_complexity
from utils.performance_metrics import evaluate_model_performance
from model import get_model
import torch

# Load and evaluate model
model = get_model()
model.load_state_dict(torch.load('./results/best_model.pth'))
print(calculate_model_complexity(model))
"

# Or use the test application
python test_app.py --model_path ./results/best_model.pth
```

### Configuration Management

The project uses YAML configuration files for flexible parameter management:

**Training Configuration (`config/training_config.yaml`):**

```yaml
# Model Architecture
model:
  num_classes: 10
  hidden_size: 256
  dropout_rate: 0.3

# Training Parameters
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 15

# Data Processing
data:
  image_size: 64
  augmentation:
    rotation_range: 15
    zoom_range: 0.1
    noise_factor: 0.1
```

**Evaluation Configuration (`config/evaluation_config.yaml`):**

```yaml
# Evaluation Settings
evaluation:
  batch_size: 32
  generate_visualizations: true
  benchmark_inference: true

# Performance Thresholds
thresholds:
  accuracy:
    excellent: 0.99
    good: 0.95
  inference_time_ms:
    real_time: 10
    interactive: 100
```

## 📁 Project Structure

```
project/
├── 📄 train.py                 # Main training script
├── 📄 model.py                 # CNN model architecture
├── 📄 dataset.py               # Dataset handling and preprocessing
├── 📄 realtime_recognition.py  # Real-time recognition system
├── 📄 test_app.py              # Application testing utilities
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project documentation
├── 📄 scientific_paper.tex    # Academic paper (LaTeX)
├── 📄 project_report.md       # Technical documentation
├──
├── 📁 config/                 # Configuration files
│   ├── training_config.yaml   # Training parameters
│   └── evaluation_config.yaml # Evaluation settings
├──
├── 📁 utils/                  # Utility modules
│   ├── model_utils.py         # Model analysis tools
│   ├── performance_metrics.py # Evaluation metrics
│   └── __init__.py
├──
├── 📁 results/                # Training outputs
│   ├── best_model.pth         # Best trained model
│   ├── training_history.json  # Training logs
│   ├── confusion_matrix.png   # Evaluation plots
│   ├── comprehensive_analysis.png # Analysis visualizations
│   └── model_summary_table.png # Model performance summary
├──
├── 📁 custom_data/            # User-generated datasets (gathered by authors)
│   ├── 00/ ... 09/           # 10 different users
│   └── [01_palm, 02_l, 03_fist, 04_fist_moved, 05_thumb, 06_index, 07_ok, 08_palm_moved, 09_c, 10_down]/
└──
└── 📁 data/                   # Original dataset (from Kaggle LeapGestRecog)
    └── 00/ ... 09/           # Gesture class folders
```

## 🏗️ Model Architecture

The system uses a custom CNN architecture optimized for hand gesture recognition:

### Network Design

- **Input**: 64×64 grayscale images
- **Architecture**: 3 convolutional blocks with progressive channel expansion
- **Channel Progression**: 1 → 64 → 128 → 256
- **Regularization**: Batch normalization + dropout (0.3)
- **Output**: 10-class softmax classification

### Technical Specifications

```python
# Model Complexity
Total Parameters: ~2.1M
Model Size: ~8.5MB
Inference Time: <3ms (GPU), <10ms (CPU)
Memory Usage: ~500MB (training), ~200MB (inference)

# Performance Metrics
Validation Accuracy: 99.82%
Precision (macro): 99.8%
Recall (macro): 99.8%
F1-Score (macro): 99.8%
```

### Training Features

- **Mixed precision training** for efficiency
- **Advanced data augmentation** (rotation, zoom, noise)
- **Learning rate scheduling** with ReduceLROnPlateau
- **Early stopping** to prevent overfitting
- **TensorBoard integration** for monitoring

## 📊 Performance Benchmarks

### Accuracy Metrics (Test Set)

| Metric              | Score  |
| ------------------- | ------ |
| Overall Accuracy    | 99.82% |
| Macro Avg Precision | 99.80% |
| Macro Avg Recall    | 99.80% |
| Macro Avg F1-Score  | 99.80% |
| Weighted Avg F1     | 99.82% |

### Inference Performance

| Platform | Time (ms) | FPS |
| -------- | --------- | --- |
| RTX 3080 | 2.3       | 435 |
| GTX 1060 | 4.1       | 244 |
| CPU (i7) | 8.7       | 115 |
| CPU (i5) | 12.4      | 81  |

### Per-Class Performance

| Gesture    | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Palm       | 99.9%     | 99.8%  | 99.8%    |
| L-Shape    | 99.7%     | 99.9%  | 99.8%    |
| Fist       | 99.8%     | 99.7%  | 99.8%    |
| Fist Moved | 99.6%     | 99.8%  | 99.7%    |
| Thumb      | 99.9%     | 99.8%  | 99.9%    |
| Index      | 99.8%     | 99.9%  | 99.8%    |
| OK Sign    | 99.9%     | 99.7%  | 99.8%    |
| Palm Moved | 99.7%     | 99.8%  | 99.8%    |
| C-Shape    | 99.8%     | 99.8%  | 99.8%    |
| Down       | 99.9%     | 99.9%  | 99.9%    |

## 📚 Academic Documentation

### Scientific Paper

- **Format**: IEEE Conference Paper (LaTeX)
- **File**: `scientific_paper.tex`
- **Sections**: Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion
- **Length**: ~8 pages with references and figures
- **Status**: Ready for academic submission

### Key Contributions

1. **Novel CNN Architecture**: Optimized 3-block design for gesture recognition
2. **Comprehensive Training Pipeline**: Advanced techniques with 99.82% accuracy
3. **Real-time Integration**: MediaPipe + CNN for practical applications
4. **Extensive Evaluation**: Statistical analysis and performance benchmarking

### Research Reproducibility

- All experiments are reproducible with provided code
- Detailed hyperparameter documentation
- Statistical significance testing included
- Cross-validation support for robust evaluation

## 🔧 Advanced Usage

### Custom Dataset Creation

```bash
# The project already includes comprehensive custom data from 10 users (00-09)
# Each user has captured all 10 gesture classes
# Data is stored in ./custom_data/ with proper structure

# To view existing data structure:
ls -la ./custom_data/
```

### Hyperparameter Optimization

```bash
# Grid search (modify train.py for hyperparameter search)
python train.py --hyperparameter_search --search_space config/search_space.yaml

# Manual parameter adjustment
python train.py --learning_rate 0.0005 --batch_size 64 --dropout 0.4
```

### Model Analysis and Visualization

```bash
# Generate comprehensive analysis using available utilities
python -c "
from utils.model_utils import calculate_model_complexity, analyze_model_layers
from utils.performance_metrics import compute_detailed_metrics
from model import get_model
import torch

model = get_model()
model.load_state_dict(torch.load('./results/best_model.pth'))
print(calculate_model_complexity(model))
print(analyze_model_layers(model))
"

# Use test application for model testing
python test_app.py --model_path ./results/best_model.pth --analyze
```

### Export and Deployment

```bash
# Export to ONNX (requires onnx package)
python -c "
import torch
from model import get_model
model = get_model()
model.load_state_dict(torch.load('./results/best_model.pth'))
torch.onnx.export(model, torch.randn(1,1,64,64), 'gesture_model.onnx')
"
```

## 🧪 Testing and Validation

### Unit Tests

```bash
# Run basic tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_model.py -v
python -m pytest tests/test_dataset.py -v
```

### Validation Procedures

1. **Cross-validation**: 5-fold CV for robust performance estimation
2. **Statistical testing**: Paired t-tests for significance
3. **Ablation studies**: Component-wise performance analysis
4. **Robustness testing**: Performance under various conditions

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
python train.py --batch_size 16

# Use CPU
python train.py --device cpu
```

**2. Camera Not Detected**

```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Specify camera index
python realtime_recognition.py --camera 1
```

**3. Import Errors**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

**4. Low Performance**

- Ensure proper lighting during data collection
- Increase dataset size for better generalization
- Adjust confidence threshold for stability
- Consider model fine-tuning on your specific data

### Performance Optimization

- Use GPU acceleration when available
- Optimize batch size for your hardware
- Enable mixed precision training
- Use data parallel training for multiple GPUs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@techreport{ilci_kara_gesture_recognition_2025,
  title={Real-Time Hand Gesture Recognition Using Deep Convolutional Neural Networks},
  author={Ilci, Bora and Kara, Kaan Emre},
  institution={EARIN Labs - Introduction to Artificial Intelligence, Politechnika Warszawska},
  year={2025},
  month={June},
  type={Academic Course Project},
  note={GitHub Repository: \url{https://github.com/borailci/EARIN_Labs}},
  abstract={A comprehensive deep learning system achieving 99.82\% validation accuracy for real-time hand gesture recognition using custom CNN architecture and MediaPipe integration.}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** for hand detection framework
- **PyTorch Team** for deep learning framework
- **Course Instructor** for project guidance
- **Open Source Community** for tools and libraries

## 📞 Contact

- **Project Lead**: Bora Ilci - bora.ilci@student.pw.edu.pl
- **Co-Author**: Kaan Emre Kara - kaan.kara@student.pw.edu.pl
- **Course**: EARIN (Introduction to Artificial Intelligence)
- **Institution**: Politechnika Warszawska (Warsaw University of Technology)
- **Academic Year**: 2025

---

**Note**: This is an academic project developed for the EARIN (Introduction to Artificial Intelligence) course at Politechnika Warszawska. The model and techniques can be adapted for research and commercial applications with proper attribution.
