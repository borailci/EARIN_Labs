import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional, Union

from src.utils.helpers import load_config, ensure_dir
from src.transforms import (
    get_train_transforms, 
    get_test_transforms, 
    TemporalRandomCrop, 
    TemporalCenterCrop,
    FrameTransform,
    stack_frames
)

class GestureDataset(Dataset):
    """
    Dataset for gesture recognition from image sequences
    """
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        transform=None, 
        seq_len: int = 20,
        img_size: int = 64,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory containing processed data
            split: 'train', 'val', or 'test'
            transform: Transform to apply to each frame
            seq_len: Target sequence length
            img_size: Target image size
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.img_size = img_size
        self.seed = seed
        
        # Set transform
        if transform is None:
            if split == 'train':
                frame_transform = get_train_transforms(img_size)
                self.temporal_transform = TemporalRandomCrop(seq_len)
            else:
                frame_transform = get_test_transforms(img_size)
                self.temporal_transform = TemporalCenterCrop(seq_len)
                
            self.transform = FrameTransform(frame_transform)
        else:
            self.transform = transform
        
        # Get class folders
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() 
                                  if d.is_dir() and not d.name.startswith('.')])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Get samples
        self.samples = self._make_dataset()
        
    def _make_dataset(self) -> List[Tuple[str, int]]:
        """
        Create a list of (sequence_path, label) tuples
        
        Returns:
            List of (sequence_path, label) tuples
        """
        samples = []
        
        # For each class folder
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]
            
            # Get all sequence folders
            sequence_folders = [d for d in class_dir.iterdir() 
                               if d.is_dir() and not d.name.startswith('.')]
            
            # Filter sequences based on split
            random.Random(self.seed).shuffle(sequence_folders)
            
            total = len(sequence_folders)
            
            if self.split == 'train':
                # Use first 70% for training
                split_folders = sequence_folders[:int(0.7 * total)]
            elif self.split == 'val':
                # Use next 15% for validation
                split_folders = sequence_folders[int(0.7 * total):int(0.85 * total)]
            else:  # 'test'
                # Use last 15% for testing
                split_folders = sequence_folders[int(0.85 * total):]
            
            # Add each sequence to samples
            for seq_folder in split_folders:
                samples.append((str(seq_folder), label))
        
        return samples
    
    def _load_sequence(self, path: str) -> List[Image.Image]:
        """
        Load a sequence of frames from path
        
        Args:
            path: Path to sequence folder
        
        Returns:
            List of PIL images
        """
        # Get all image files, sorted by name
        files = sorted([f for f in Path(path).iterdir() 
                      if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        
        # Load each image
        frames = []
        for file_path in files:
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            frames.append(img)
        
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (sequence tensor, label)
                sequence tensor: shape [seq_len, channels, height, width]
                label: Class index
        """
        path, label = self.samples[idx]
        
        # Load sequence
        frames = self._load_sequence(path)
        
        # Apply temporal transform
        frames = self.temporal_transform(frames)
        
        # Apply frame transform
        frames = self.transform(frames)
        
        # Stack frames into tensor
        sequence = stack_frames(frames)
        
        return sequence, label

def make_dataloaders(config):
    """
    Create data loaders based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    train_config = config['training']
    
    # Get paths and parameters
    data_dir = data_config['processed_dir']
    seq_len = data_config['seq_len']
    img_size = data_config['img_size']
    batch_size = train_config['batch_size']
    
    # Create datasets
    train_dataset = GestureDataset(
        root_dir=data_dir,
        split='train',
        seq_len=seq_len,
        img_size=img_size
    )
    
    val_dataset = GestureDataset(
        root_dir=data_dir,
        split='val',
        seq_len=seq_len,
        img_size=img_size
    )
    
    test_dataset = GestureDataset(
        root_dir=data_dir,
        split='test',
        seq_len=seq_len,
        img_size=img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
