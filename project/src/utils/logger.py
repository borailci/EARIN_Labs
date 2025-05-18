import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def get_logger(name='gesture_recognition', log_dir=None, level=logging.INFO):
    """Get logger instance with reasonable defaults"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_dir is provided
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

class TensorboardWriter:
    """Wrapper for tensorboard SummaryWriter with useful methods"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step = 0
        
    def set_step(self, step):
        """Set current step"""
        self.step = step
    
    def update_step(self):
        """Increment step by 1"""
        self.step += 1
        
    def add_scalar(self, tag, value):
        """Add scalar with current step"""
        self.writer.add_scalar(tag, value, self.step)
        
    def add_scalars(self, tag, values):
        """Add group of scalars with current step"""
        self.writer.add_scalars(tag, values, self.step)
        
    def add_image(self, tag, image):
        """Add image with current step"""
        self.writer.add_image(tag, image, self.step)
        
    def add_figure(self, tag, figure):
        """Add matplotlib figure with current step"""
        self.writer.add_figure(tag, figure, self.step)
    
    def close(self):
        """Close writer"""
        self.writer.close()
