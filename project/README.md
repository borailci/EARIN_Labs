# Hand Gesture Recognition

A deep learning system for hand gesture recognition using a CNN-LSTM architecture.

## Project Structure

```
├── README.md
├── requirements.txt
├── setup.py
├── config/                  # Hyperparameter and path configs
│   └── default.yaml
├── data/
│   ├── raw/                 # unprocessed LeapGestRecog CSVs or videos
│   ├── processed/           # resized frames organized by label
│   │   └── <gesture>/<seq_id>/frame_000.png
│   └── processed_aug/       # augmented data
├── scripts/
│   ├── prepare_data.py      # raw → processed + splitting/shuffling
│   ├── augment_data.py      # data augmentation pipeline
│   └── capture_live.py      # webcam/MediaPipe capture utility
├── src/
│   ├── data_loader.py       # Dataset classes & PyTorch DataLoaders
│   ├── transforms.py        # torchvision transforms & custom aug
│   ├── models/
│   │   └── cnn_lstm.py      # CNN+LSTM architecture
│   ├── train.py             # training driver, CLI entrypoint
│   ├── evaluate.py          # evaluation & metrics scripts
│   ├── visualize.py         # plotting curves, confusion
│   ├── inference.py         # batch & real-time inference code
│   └── utils/
│       ├── logger.py        # logger/TensorBoard writer
│       ├── metrics.py       # accuracy, F1, confusion matrix
│       └── helpers.py       # I/O, seed setting, timing
├── experiments/             # output directory for experiment results
└── exported_models/         # saved .pt model files
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

1. Prepare the processed data from raw CSV files or videos:

   ```
   python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed --data_type csv
   ```

2. (Optional) Augment the data for better model generalization:
   ```
   python scripts/augment_data.py --input_dir data/processed --output_dir data/processed_aug --include_original
   ```

## Training

Train the CNN-LSTM model:

```
python src/train.py --config config/default.yaml
```

## Evaluation

Evaluate the trained model:

```
python src/evaluate.py --config config/default.yaml --checkpoint experiments/cnn_lstm_*/checkpoints/best_model.pth
```

## Inference

Run real-time inference using a webcam:

```
python src/inference.py --model experiments/cnn_lstm_*/checkpoints/best_model.pth --config config/default.yaml
```

Or run inference on a video file:

```
python src/inference.py --model experiments/cnn_lstm_*/checkpoints/best_model.pth --config config/default.yaml --video path/to/video.mp4 --output results.mp4
```

## Model Architecture

The project implements a CNN-LSTM architecture:

- CNN layers extract spatial features from each frame
- LSTM layers model the temporal dynamics of the gesture sequence
- Fully connected layers for final classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.
