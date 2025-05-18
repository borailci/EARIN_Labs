import os
import time
import random
import numpy as np
import torch
import yaml
from pathlib import Path

def set_seed(seed):
    """Set random seeds for reproducibility across libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_config(path):
    """Load YAML configuration file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name} took {self.interval:.4f} seconds")
