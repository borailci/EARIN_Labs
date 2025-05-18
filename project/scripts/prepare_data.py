#!/usr/bin/env python
import os
import argparse
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path to import from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import ensure_dir, set_seed

def parse_leap_csv(file_path, img_size=64, normalize=True):
    """
    Parse CSV file from LeapGestRecog dataset and extract frames as images
    
    Args:
        file_path: Path to CSV file
        img_size: Size of output images
        normalize: Whether to normalize images to [0, 255]
    
    Returns:
        List of numpy arrays representing frames
    """
    # Read CSV
    data = pd.read_csv(file_path, header=None)
    
    # Extract frames
    frames = []
    for _, row in data.iterrows():
        # Reshape to 28x28 (LeapGestRecog format)
        img = np.array(row).reshape(28, 28).astype(np.float32)
        
        # Normalize to [0, 255]
        if normalize:
            img = (img - img.min()) / (img.max() - img.min()) * 255
        
        # Resize to img_size
        img = cv2.resize(img, (img_size, img_size))
        
        frames.append(img.astype(np.uint8))
    
    return frames

def extract_frames_from_video(video_path, img_size=64, gray=True):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        img_size: Size of output images
        gray: Whether to convert to grayscale
    
    Returns:
        List of numpy arrays representing frames
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if requested
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to img_size
        frame = cv2.resize(frame, (img_size, img_size))
        
        frames.append(frame)
    
    cap.release()
    return frames

def save_frames(frames, output_dir, prefix="frame"):
    """
    Save frames as PNG images
    
    Args:
        frames: List of numpy arrays representing frames
        output_dir: Directory to save frames to
        prefix: Prefix for frame filenames
    """
    ensure_dir(output_dir)
    
    for i, frame in enumerate(frames):
        # Create PIL image
        img = Image.fromarray(frame)
        
        # Save as PNG
        img.save(os.path.join(output_dir, f"{prefix}_{i:03d}.png"))

def process_leap_dataset(input_dir, output_dir, img_size=64, seq_len=None):
    """
    Process LeapGestRecog dataset
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed frames
        img_size: Size of output images
        seq_len: Number of frames to extract (None for all)
    
    Returns:
        List of (gesture, sequence_id) tuples
    """
    # Find all CSV files in input directory
    csv_files = list(Path(input_dir).glob("**/*.csv"))
    
    # Group by gesture
    gesture_files = {}
    for file_path in csv_files:
        # Extract gesture from parent directory name
        gesture = file_path.parent.name
        
        if gesture not in gesture_files:
            gesture_files[gesture] = []
        
        gesture_files[gesture].append(file_path)
    
    # Process each gesture
    samples = []
    for gesture, files in tqdm(gesture_files.items(), desc="Processing gestures"):
        # Create output directory for gesture
        gesture_dir = os.path.join(output_dir, gesture)
        ensure_dir(gesture_dir)
        
        # Process each CSV file
        for i, file_path in enumerate(files):
            # Create sequence directory
            seq_dir = os.path.join(gesture_dir, f"seq_{i:03d}")
            
            # Parse CSV and extract frames
            frames = parse_leap_csv(file_path, img_size)
            
            # Subsample frames if seq_len is set
            if seq_len is not None and len(frames) > seq_len:
                # Evenly spaced indices
                indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Save frames
            save_frames(frames, seq_dir)
            
            # Add to samples
            samples.append((gesture, f"seq_{i:03d}"))
    
    return samples

def process_video_dataset(input_dir, output_dir, img_size=64, seq_len=None):
    """
    Process video dataset
    
    Args:
        input_dir: Directory containing raw video files
        output_dir: Directory to save processed frames
        img_size: Size of output images
        seq_len: Number of frames to extract (None for all)
    
    Returns:
        List of (gesture, sequence_id) tuples
    """
    # Find all video files in input directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mpeg', '.flv', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
    
    # Group by gesture
    gesture_files = {}
    for file_path in video_files:
        # Extract gesture from parent directory name
        gesture = file_path.parent.name
        
        if gesture not in gesture_files:
            gesture_files[gesture] = []
        
        gesture_files[gesture].append(file_path)
    
    # Process each gesture
    samples = []
    for gesture, files in tqdm(gesture_files.items(), desc="Processing gestures"):
        # Create output directory for gesture
        gesture_dir = os.path.join(output_dir, gesture)
        ensure_dir(gesture_dir)
        
        # Process each video file
        for i, file_path in enumerate(files):
            # Create sequence directory
            seq_dir = os.path.join(gesture_dir, f"seq_{i:03d}")
            
            # Extract frames from video
            frames = extract_frames_from_video(file_path, img_size)
            
            # Subsample frames if seq_len is set
            if seq_len is not None and len(frames) > seq_len:
                # Evenly spaced indices
                indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Save frames
            save_frames(frames, seq_dir)
            
            # Add to samples
            samples.append((gesture, f"seq_{i:03d}"))
    
    return samples

def create_dataset_splits(samples, output_dir, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    """
    Create train/val/test splits and save to files
    
    Args:
        samples: List of (gesture, sequence_id) tuples
        output_dir: Directory to save split files
        train_split: Proportion of samples for training
        val_split: Proportion of samples for validation
        test_split: Proportion of samples for testing
        seed: Random seed for reproducibility
    """
    # Verify splits sum to 1
    if abs(train_split + val_split + test_split - 1.0) > 1e-5:
        raise ValueError(f"Splits must sum to 1, got {train_split + val_split + test_split}")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Calculate split indices
    n_samples = len(samples)
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)
    
    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Create split directory
    split_dir = os.path.join(output_dir, 'splits')
    ensure_dir(split_dir)
    
    # Save splits to files
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        for gesture, seq_id in train_samples:
            f.write(f"{gesture}/{seq_id}\n")
    
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        for gesture, seq_id in val_samples:
            f.write(f"{gesture}/{seq_id}\n")
    
    with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
        for gesture, seq_id in test_samples:
            f.write(f"{gesture}/{seq_id}\n")
    
    print(f"Created splits: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare data for gesture recognition')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--img_size', type=int, default=64, help='Size of output images')
    parser.add_argument('--seq_len', type=int, default=None, help='Number of frames to extract (None for all)')
    parser.add_argument('--data_type', type=str, choices=['csv', 'video'], default='csv', help='Type of raw data (csv for LeapGestRecog, video for video files)')
    parser.add_argument('--train_split', type=float, default=0.7, help='Proportion of samples for training')
    parser.add_argument('--val_split', type=float, default=0.15, help='Proportion of samples for validation')
    parser.add_argument('--test_split', type=float, default=0.15, help='Proportion of samples for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Process data
    if args.data_type == 'csv':
        print(f"Processing LeapGestRecog CSV dataset from {args.input_dir}")
        samples = process_leap_dataset(args.input_dir, args.output_dir, args.img_size, args.seq_len)
    else:
        print(f"Processing video dataset from {args.input_dir}")
        samples = process_video_dataset(args.input_dir, args.output_dir, args.img_size, args.seq_len)
    
    # Create dataset splits
    create_dataset_splits(
        samples, 
        args.output_dir, 
        args.train_split, 
        args.val_split, 
        args.test_split, 
        args.seed
    )
    
    print(f"Data preparation complete. Processed data saved to {args.output_dir}")

if __name__ == '__main__':
    main()
