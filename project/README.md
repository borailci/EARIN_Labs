# Hand Gesture Recognition

A real-time hand gesture recognition system using deep learning and computer vision. The system can detect and classify hand gestures in real-time using a webcam.

## Features

- Real-time hand gesture recognition using webcam
- Support for 10 different hand gestures
- Pre-trained model included
- Training pipeline for custom datasets
- Comprehensive testing and visualization tools
- Memory-efficient processing with caching

## Supported Gestures

1. Palm
2. L Shape
3. Fist
4. Fist (moved)
5. Thumb
6. Index Finger
7. OK Sign
8. Palm (moved)
9. C Shape
10. Down Sign

## Requirements

- Python 3.11 or higher
- Webcam (for real-time recognition)
- GPU recommended but not required

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

The project uses a YAML configuration file (`config/training_config.yaml`) to manage all hyperparameters and settings. You can modify this file to customize:

- Model architecture (layers, sizes, activation functions)
- Training parameters (learning rate, batch size, optimizer)
- Data processing (augmentation, preprocessing)
- Hardware utilization (number of workers, CUDA device)
- Logging and checkpoint settings

Example configuration modification:

```yaml
# Modify batch size and learning rate
training:
  batch_size: 64
  learning_rate: 0.0005

# Enable data augmentation
data:
  augmentation:
    enabled: true
    rotation_range: 20
```

### Real-time Recognition

To run real-time gesture recognition using your webcam:

```bash
python realtime_recognition.py --model results/best_model.pth
```

Controls:

- Press 's' to save the current frame and its processed versions
- Press 'q' to quit

### Training

To train a new model:

```bash
python train.py --data_dir ./data --output_dir ./results
```

Optional arguments:

- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 32)
- `--hidden_size`: Hidden layer size (default: 128)
- `--epochs`: Maximum number of epochs (default: 50)
- `--patience`: Early stopping patience (default: 5)

### Testing

To test the model on the dataset:

```bash
python test_dataset.py --model_path results/best_model.pth --data_dir data
```

Optional arguments:

- `--num_samples`: Number of test samples to process (default: 100)
- `--output_dir`: Directory to save test results and visualizations
- `--batch_mode`: Use batch processing for faster testing
- `--batch_size`: Batch size for batch mode (default: 32)

## Project Structure

```
.
├── data/                  # Dataset directory
├── results/              # Trained models and results
├── captured_data/        # Saved frames from real-time recognition
├── realtime_recognition.py  # Real-time recognition script
├── train.py             # Training script
├── test_dataset.py      # Testing and validation script
├── model.py             # Model architecture
├── dataset.py           # Dataset handling
└── requirements.txt     # Project dependencies
```

## Performance Metrics

The system provides comprehensive performance metrics including:

- Overall accuracy
- Per-class accuracy
- Inference time
- Memory usage
- Confidence calibration
- Confusion matrix

## Troubleshooting

1. If you encounter CUDA out of memory errors:

   - Reduce batch size
   - Use CPU mode if GPU memory is limited

2. If webcam doesn't work:

   - Check webcam permissions
   - Try different webcam index (modify cv2.VideoCapture(0))

3. If hand detection is unstable:
   - Ensure good lighting conditions
   - Keep hand within the marked ROI
   - Adjust min_brightness and max_brightness parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand detection
- PyTorch team for the deep learning framework
- OpenCV team for computer vision tools
