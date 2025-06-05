"""
Hand Gesture Recognition Dataset Module

This module provides comprehensive dataset handling capabilities for hand gesture recognition,
including data loading, preprocessing, augmentation, and train/validation splitting.

Key Features:
- Support for both original and custom gesture datasets
- Comprehensive data augmentation pipeline
- Balanced dataset creation with stratified sampling
- Multi-class gesture classification support
- Efficient data loading with PyTorch DataLoader integration

Classes:
    LeapGestRecogDataset: Main dataset class for gesture recognition

Functions:
    get_data_loaders: Create train/validation DataLoaders with proper augmentation
    get_class_names: Extract class names from dataset directory structure

Author: Course Project Team
Date: Academic Year 2024
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import re
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")  # Prevent sharing issues


class LeapGestRecogDataset(Dataset):
    def __init__(
        self,
        data_dir,
        custom_data_dir=None,
        transform=None,
        mode="train",
        log_callback=None,
    ):
        """
        Args:
            data_dir (str): Directory with original gesture classes
            custom_data_dir (str): Directory with custom captured gesture classes
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train', 'val', or 'test'
            log_callback (callable, optional): Function to call for logging messages
        """
        self.data_dir = data_dir
        self.custom_data_dir = custom_data_dir
        self.transform = transform
        self.mode = mode
        self.log_callback = log_callback

        # Get all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []

        # Class mapping based on the frame numbers
        self.class_names = {
            "01": "palm",
            "02": "l",
            "03": "fist",
            "04": "fist_moved",
            "05": "thumb",
            "06": "index",
            "07": "ok",
            "08": "palm_moved",
            "09": "c",
            "10": "down",
        }

        # Load original dataset
        self._load_dataset(data_dir)

        # Load custom dataset if provided
        if custom_data_dir and os.path.exists(custom_data_dir):
            self._load_dataset(custom_data_dir)

        # Convert labels to tensor for faster access
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # Initialize indices array
        self.indices = np.arange(len(self.image_paths))

        # Create stratified train/val/test split
        if len(self.image_paths) > 0:  # Only split if we have data
            train_idx, temp_idx = train_test_split(
                self.indices, test_size=0.3, stratify=self.labels, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=self.labels[temp_idx], random_state=42
            )

            # Select appropriate split based on mode
            if mode == "train":
                self.indices = train_idx
            elif mode == "val":
                self.indices = val_idx
            else:  # test
                self.indices = test_idx

            # Pre-load images into memory for faster access if in training mode
            if mode == "train":
                if self.log_callback:
                    self.log_callback("Pre-loading training images into memory...")
                self.images = []
                for idx in self.indices:
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert("L")
                    self.images.append(image)
            else:
                self.images = None
        else:
            if self.log_callback:
                self.log_callback(f"Warning: No images found in {data_dir}")
            self.images = None

    def _load_dataset(self, data_dir):
        """Load images and labels from a dataset directory"""
        if self.log_callback:
            self.log_callback(f"\nLoading dataset from: {data_dir}")
        total_images = 0
        class_counts = {i: 0 for i in range(10)}  # Count images per class

        # Both datasets use 00-09 folders
        for folder_idx in range(10):  # 00-09
            folder = f"{folder_idx:02d}"
            folder_path = os.path.join(data_dir, folder)

            if os.path.exists(folder_path):
                # Find all PNG files in this folder and its subdirectories
                frames = glob.glob(
                    os.path.join(folder_path, "**/*.png"), recursive=True
                )

                if self.log_callback:
                    self.log_callback(f"Found {len(frames)} images in folder {folder}")

                for frame_path in frames:
                    frame_name = os.path.basename(frame_path)
                    # Extract class number from filename using regex
                    match = re.match(r"^frame_\d+_(\d+)_\d+\.png$", frame_name)
                    if match:
                        class_num = match.group(1).zfill(2)
                        if class_num in self.class_names:
                            self.image_paths.append(frame_path)
                            class_idx = int(class_num) - 1  # Convert to 0-based index
                            self.labels.append(class_idx)
                            class_counts[class_idx] += 1
                            total_images += 1
                        else:
                            if self.log_callback:
                                self.log_callback(
                                    f"Warning: Invalid class number {class_num} in file {frame_name}"
                                )
                    else:
                        if self.log_callback:
                            self.log_callback(
                                f"Warning: Filename {frame_name} doesn't match expected pattern"
                            )

        if self.log_callback:
            self.log_callback(f"\nDataset Summary for {data_dir}:")
            self.log_callback(f"Total images loaded: {total_images}")
            self.log_callback("Images per class:")
            for class_idx, count in class_counts.items():
                class_name = self.class_names[
                    f"{class_idx+1:02d}"
                ]  # Add 1 to convert from 0-based to 1-based index
                self.log_callback(f"  {class_name}: {count} images")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]

        # Get image
        if self.images is not None:
            # Use pre-loaded image for training
            image = self.images[idx]
        else:
            # Load image for validation/test
            img_path = self.image_paths[true_idx]
            image = Image.open(img_path).convert("L")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[true_idx]

        return image, label


def get_data_loaders(
    data_dir, custom_data_dir=None, batch_size=32, num_workers=4, log_callback=None
):
    """
    Create data loaders for train, validation, and test sets

    Args:
        data_dir (str): Path to original data directory
        custom_data_dir (str): Path to custom captured data directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        log_callback (callable, optional): Function to call for logging messages

    Returns:
        dict: Dictionary containing data loaders
    """
    # Define transformations for training (with less aggressive augmentation)
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomApply(
                [
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=10,
                    ),
                ],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ],
                p=0.2,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Define transformations for validation/test (no augmentation)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets with appropriate transforms
    train_dataset = LeapGestRecogDataset(
        data_dir,
        custom_data_dir,
        transform=train_transform,
        mode="train",
        log_callback=log_callback,
    )
    val_dataset = LeapGestRecogDataset(
        data_dir,
        custom_data_dir,
        transform=eval_transform,
        mode="val",
        log_callback=log_callback,
    )
    test_dataset = LeapGestRecogDataset(
        data_dir,
        custom_data_dir,
        transform=eval_transform,
        mode="test",
        log_callback=log_callback,
    )

    # Determine if pin_memory should be used (not supported on MPS)
    use_pin_memory = torch.cuda.is_available()

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if log_callback:
        log_callback(f"\nDataloader Configuration:")
        log_callback(f"Number of workers: {num_workers}")
        log_callback(f"Batch size - Train: {batch_size}, Val/Test: {batch_size * 2}")
        log_callback(f"Train set size: {len(train_dataset)}")
        log_callback(f"Validation set size: {len(val_dataset)}")
        log_callback(f"Test set size: {len(test_dataset)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_class_names():
    """Get the mapping of class indices to class names"""
    return {
        "01": "palm",
        "02": "l",
        "03": "fist",
        "04": "fist_moved",
        "05": "thumb",
        "06": "index",
        "07": "ok",
        "08": "palm_moved",
        "09": "c",
        "10": "down",
    }
