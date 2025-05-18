import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

class TemporalRandomCrop:
    """
    Randomly crop a sequence to a specified length
    
    If sequence is shorter than target length, duplicates frames
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def __call__(self, frames):
        """
        Args:
            frames: List of PIL images or torch tensors
        
        Returns:
            Cropped sequence of length seq_len
        """
        if len(frames) == 0:
            raise ValueError("Empty frame sequence")
        
        # If sequence is too short, duplicate frames
        if len(frames) < self.seq_len:
            # Duplicate frames with indices
            idx = np.linspace(0, len(frames) - 1, self.seq_len).astype(int)
            return [frames[i] for i in idx]
        
        # If sequence is longer than target, randomly crop
        if len(frames) > self.seq_len:
            start_idx = random.randint(0, len(frames) - self.seq_len)
            return frames[start_idx:start_idx + self.seq_len]
        
        # If sequence is exactly the right length
        return frames

class TemporalCenterCrop:
    """
    Center crop a sequence to a specified length
    
    If sequence is shorter than target length, duplicates frames
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def __call__(self, frames):
        """
        Args:
            frames: List of PIL images or torch tensors
        
        Returns:
            Center-cropped sequence of length seq_len
        """
        if len(frames) == 0:
            raise ValueError("Empty frame sequence")
        
        # If sequence is too short, duplicate frames
        if len(frames) < self.seq_len:
            # Duplicate frames with indices
            idx = np.linspace(0, len(frames) - 1, self.seq_len).astype(int)
            return [frames[i] for i in idx]
        
        # If sequence is longer than target, center crop
        if len(frames) > self.seq_len:
            start_idx = (len(frames) - self.seq_len) // 2
            return frames[start_idx:start_idx + self.seq_len]
        
        # If sequence is exactly the right length
        return frames

class FrameTransform:
    """
    Apply transformation to each frame in a sequence
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, frames):
        """
        Args:
            frames: List of PIL images
        
        Returns:
            Transformed frames
        """
        return [self.transform(frame) for frame in frames]

def get_train_transforms(img_size=64):
    """
    Get training transforms with augmentation
    """
    frame_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=15,  # Rotation ±15 degrees
            translate=(0.1, 0.1),  # Horizontal/vertical shift ±10%
            scale=(0.8, 1.2),  # Zoom 0.8-1.2x
            fill=0
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return frame_transform

def get_test_transforms(img_size=64):
    """
    Get test transforms without augmentation
    """
    frame_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return frame_transform

def stack_frames(frames_list):
    """
    Stack a list of tensors into a single tensor
    
    Args:
        frames_list: List of tensors, each of shape [C, H, W]
    
    Returns:
        Tensor of shape [seq_len, C, H, W]
    """
    if not frames_list:
        raise ValueError("Empty frames list")
    
    return torch.stack(frames_list, dim=0)
