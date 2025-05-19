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
    def __init__(self, data_dir, transform=None, mode="train"):
        """
        Args:
            data_dir (str): Directory with all gesture classes
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

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

        # Collect all image paths and labels
        for folder_idx in range(10):  # 10 main folders (00 to 09)
            folder = f"{folder_idx:02d}"
            folder_path = os.path.join(data_dir, folder)

            if os.path.exists(folder_path):
                # Find all PNG files in this folder and its subdirectories
                frames = glob.glob(
                    os.path.join(folder_path, "**/*.png"), recursive=True
                )

                for frame_path in frames:
                    frame_name = os.path.basename(frame_path)
                    # Extract class number from filename using regex
                    match = re.match(r"^frame_\d+_(\d+)_\d+\.png$", frame_name)
                    if match:
                        class_num = match.group(1).zfill(2)
                        if class_num in self.class_names:
                            self.image_paths.append(frame_path)
                            class_idx = int(class_num) - 1
                            self.labels.append(class_idx)

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
                print("Pre-loading training images into memory...")
                self.images = []
                for idx in self.indices:
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert("L")
                    self.images.append(image)
            else:
                self.images = None
        else:
            print(f"Warning: No images found in {data_dir}")
            self.images = None

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


def get_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for train, validation, and test sets

    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for data loaders

    Returns:
        dict: Dictionary containing data loaders
    """
    # Define transformations for training (with less aggressive augmentation)
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomApply(
                [
                    transforms.RandomRotation(15),  # Reduced rotation angle
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),  # Reduced translation
                        scale=(0.9, 1.1),  # Reduced scale range
                        shear=10,  # Reduced shear
                    ),
                ],
                p=0.5,  # Reduced probability
            ),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced flip probability
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3, sigma=(0.1, 1.0)
                    ),  # Reduced blur
                ],
                p=0.2,  # Reduced probability
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2
                    ),  # Reduced jitter
                ],
                p=0.2,  # Reduced probability
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

    # Get number of CPU cores for data loading
    num_workers = min(8, os.cpu_count())

    # Create datasets with appropriate transforms
    train_dataset = LeapGestRecogDataset(
        data_dir, transform=train_transform, mode="train"
    )
    val_dataset = LeapGestRecogDataset(data_dir, transform=eval_transform, mode="val")
    test_dataset = LeapGestRecogDataset(data_dir, transform=eval_transform, mode="test")

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch size for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"\nDataloader Configuration:")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size - Train: {batch_size}, Val/Test: {batch_size * 2}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_class_names():
    """Return the mapping of class indices to human-readable names"""
    return {
        0: "palm",  # 01
        1: "l",  # 02
        2: "fist",  # 03
        3: "fist_moved",  # 04
        4: "thumb",  # 05
        5: "index",  # 06
        6: "ok",  # 07
        7: "palm_moved",  # 08
        8: "c",  # 09
        9: "down",  # 10
    }
