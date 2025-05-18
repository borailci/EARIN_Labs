#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
import cv2
import shutil

# Add parent directory to path to import from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import ensure_dir, set_seed

def random_crop(img, min_factor=0.8):
    """
    Randomly crop and resize image
    
    Args:
        img: PIL image
        min_factor: Minimum crop size as fraction of original image
    
    Returns:
        Cropped and resized PIL image
    """
    width, height = img.size
    crop_factor = random.uniform(min_factor, 1.0)
    
    crop_width = int(width * crop_factor)
    crop_height = int(height * crop_factor)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    cropped = img.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height))

def random_rotate(img, max_angle=15):
    """
    Randomly rotate image
    
    Args:
        img: PIL image
        max_angle: Maximum rotation angle in degrees
    
    Returns:
        Rotated PIL image
    """
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)

def random_shift(img, max_shift=0.1):
    """
    Randomly shift image
    
    Args:
        img: PIL image
        max_shift: Maximum shift as fraction of image size
    
    Returns:
        Shifted PIL image
    """
    width, height = img.size
    
    x_shift = int(width * random.uniform(-max_shift, max_shift))
    y_shift = int(height * random.uniform(-max_shift, max_shift))
    
    shifted = Image.new(img.mode, img.size, 0)
    shifted.paste(img, (x_shift, y_shift))
    
    return shifted

def random_brightness(img, factor_range=(0.8, 1.2)):
    """
    Randomly adjust image brightness
    
    Args:
        img: PIL image
        factor_range: Range of brightness adjustment factors
    
    Returns:
        Brightness-adjusted PIL image
    """
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def random_contrast(img, factor_range=(0.8, 1.2)):
    """
    Randomly adjust image contrast
    
    Args:
        img: PIL image
        factor_range: Range of contrast adjustment factors
    
    Returns:
        Contrast-adjusted PIL image
    """
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def random_scale(img, scale_range=(0.8, 1.2)):
    """
    Randomly scale image
    
    Args:
        img: PIL image
        scale_range: Range of scaling factors
    
    Returns:
        Scaled PIL image
    """
    scale = random.uniform(*scale_range)
    width, height = img.size
    new_width, new_height = int(width * scale), int(height * scale)
    
    # Resize
    resized = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Create new image and paste resized image in center
    result = Image.new(img.mode, img.size, 0)
    
    left = max(0, (width - new_width) // 2)
    top = max(0, (height - new_height) // 2)
    
    # Crop if necessary
    if new_width > width or new_height > height:
        crop_left = max(0, (new_width - width) // 2)
        crop_top = max(0, (new_height - height) // 2)
        crop_right = crop_left + width
        crop_bottom = crop_top + height
        resized = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
        left, top = 0, 0
    
    result.paste(resized, (left, top))
    return result

def augment_frame(img, augmentation_strength=1.0):
    """
    Apply random augmentations to a single frame
    
    Args:
        img: PIL image
        augmentation_strength: Strength of augmentation (0.0 to 1.0)
    
    Returns:
        Augmented PIL image
    """
    # Skip augmentation with probability 1 - augmentation_strength
    if random.random() > augmentation_strength:
        return img
    
    # Apply random transformations with 50% probability each
    if random.random() < 0.5:
        img = random_crop(img)
    
    if random.random() < 0.5:
        img = random_rotate(img)
    
    if random.random() < 0.5:
        img = random_shift(img)
    
    if random.random() < 0.5:
        img = random_brightness(img)
    
    if random.random() < 0.5:
        img = random_contrast(img)
    
    if random.random() < 0.5:
        img = random_scale(img)
    
    return img

def random_frame_drop(frames, drop_prob=0.1, min_frames=10):
    """
    Randomly drop frames from a sequence
    
    Args:
        frames: List of frames
        drop_prob: Probability of dropping a frame
        min_frames: Minimum number of frames to keep
    
    Returns:
        List of frames with some frames dropped
    """
    # Ensure we have at least min_frames frames
    if len(frames) <= min_frames:
        return frames
    
    # Randomly select frames to keep
    keep_indices = []
    for i in range(len(frames)):
        if random.random() > drop_prob or len(keep_indices) < min_frames:
            keep_indices.append(i)
    
    # Ensure we have at least min_frames frames
    while len(keep_indices) < min_frames:
        # Find indices not in keep_indices
        drop_indices = [i for i in range(len(frames)) if i not in keep_indices]
        if not drop_indices:
            break
        # Add a random dropped index
        keep_indices.append(random.choice(drop_indices))
    
    keep_indices.sort()  # Preserve frame order
    
    return [frames[i] for i in keep_indices]

def random_frame_repeat(frames, repeat_prob=0.1, max_repeats=2):
    """
    Randomly repeat frames in a sequence
    
    Args:
        frames: List of frames
        repeat_prob: Probability of repeating a frame
        max_repeats: Maximum number of times to repeat a frame
    
    Returns:
        List of frames with some frames repeated
    """
    result = []
    
    for frame in frames:
        result.append(frame)
        
        # Randomly repeat frame
        if random.random() < repeat_prob:
            repeats = random.randint(1, max_repeats)
            for _ in range(repeats):
                result.append(frame)
    
    return result

def augment_sequence(input_dir, output_dir, target_len=None, augmentation_strength=1.0):
    """
    Augment a sequence of frames
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save augmented frames
        target_len: Target sequence length (None to keep original)
        augmentation_strength: Strength of augmentation (0.0 to 1.0)
    """
    # Create output directory
    ensure_dir(output_dir)
    
    # Find all image files
    image_files = sorted([f for f in Path(input_dir).iterdir() 
                         if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    
    # Load frames
    frames = []
    for file_path in image_files:
        img = Image.open(file_path).convert('L')  # Convert to grayscale
        frames.append(img)
    
    # Apply temporal augmentations
    if random.random() < 0.5 and len(frames) > 1:
        frames = random_frame_drop(frames)
    
    if random.random() < 0.5:
        frames = random_frame_repeat(frames)
    
    # Adjust sequence length if target_len is specified
    if target_len is not None:
        if len(frames) < target_len:
            # Repeat frames to reach target_len
            indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) > target_len:
            # Sample frames to reach target_len
            indices = sorted(random.sample(range(len(frames)), target_len))
            frames = [frames[i] for i in indices]
    
    # Apply spatial augmentations to each frame
    aug_frames = []
    for img in frames:
        aug_img = augment_frame(img, augmentation_strength)
        aug_frames.append(aug_img)
    
    # Save augmented frames
    for i, img in enumerate(aug_frames):
        img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

def augment_dataset(input_dir, output_dir, target_len=None, n_augmentations=5, augmentation_strength=1.0):
    """
    Augment all sequences in a dataset
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory to save augmented data
        target_len: Target sequence length (None to keep original)
        n_augmentations: Number of augmented copies to create for each sequence
        augmentation_strength: Strength of augmentation (0.0 to 1.0)
    """
    # Find all gesture class directories
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        # Create output directory for class
        class_name = class_dir.name
        class_output_dir = os.path.join(output_dir, class_name)
        ensure_dir(class_output_dir)
        
        # Find all sequence directories
        seq_dirs = [d for d in class_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Process each sequence
        for seq_dir in tqdm(seq_dirs, desc=f"Processing {class_name} sequences", leave=False):
            # Augment sequence multiple times
            for aug_idx in range(n_augmentations):
                # Create output directory for augmented sequence
                seq_name = seq_dir.name
                aug_output_dir = os.path.join(class_output_dir, f"{seq_name}_aug_{aug_idx}")
                
                # Augment sequence
                augment_sequence(
                    seq_dir, 
                    aug_output_dir, 
                    target_len,
                    augmentation_strength
                )

def copy_original_data(input_dir, output_dir):
    """
    Copy original data to output directory
    
    Args:
        input_dir: Directory containing original data
        output_dir: Directory to copy data to
    """
    # Find all gesture class directories
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Copying original data"):
        # Create output directory for class
        class_name = class_dir.name
        class_output_dir = os.path.join(output_dir, class_name)
        ensure_dir(class_output_dir)
        
        # Find all sequence directories
        seq_dirs = [d for d in class_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Process each sequence
        for seq_dir in seq_dirs:
            # Create output directory for sequence
            seq_name = seq_dir.name
            seq_output_dir = os.path.join(class_output_dir, seq_name)
            
            # Copy sequence directory
            if not os.path.exists(seq_output_dir):
                shutil.copytree(seq_dir, seq_output_dir)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Augment data for gesture recognition')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented data')
    parser.add_argument('--seq_len', type=int, default=None, help='Target sequence length (None to keep original)')
    parser.add_argument('--n_augmentations', type=int, default=5, help='Number of augmented copies to create for each sequence')
    parser.add_argument('--augmentation_strength', type=float, default=1.0, help='Strength of augmentation (0.0 to 1.0)')
    parser.add_argument('--include_original', action='store_true', help='Include original data in output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Copy original data if requested
    if args.include_original:
        print("Copying original data...")
        copy_original_data(args.input_dir, args.output_dir)
    
    # Augment data
    print(f"Augmenting data with {args.n_augmentations} copies per sequence...")
    augment_dataset(
        args.input_dir,
        args.output_dir,
        args.seq_len,
        args.n_augmentations,
        args.augmentation_strength
    )
    
    print(f"Data augmentation complete. Augmented data saved to {args.output_dir}")

if __name__ == '__main__':
    main()
