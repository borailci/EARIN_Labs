import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from collections import deque
import time
from PIL import Image
import torchvision.transforms as transforms

from src.utils.helpers import load_config, ensure_dir
from src.models.cnn_lstm import CNNLSTM
from src.transforms import get_test_transforms, stack_frames

class GesturePredictor:
    """
    Class for gesture prediction using a trained model
    
    Supports both batch prediction and real-time inference
    """
    def __init__(self, model_path, config_path, device=None):
        """
        Initialize gesture predictor
        
        Args:
            model_path: Path to trained model
            config_path: Path to config file
            device: Device to run inference on (None for auto-detection)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load config
        self.config = load_config(config_path)
        
        # Get model parameters
        self.img_size = self.config['data']['img_size']
        self.seq_len = self.config['data']['seq_len']
        
        # Load class names from processed directory
        data_dir = Path(self.config['data']['processed_dir'])
        self.class_names = sorted([d.name for d in data_dir.iterdir() 
                                 if d.is_dir() and not d.name.startswith('.')])
        self.n_classes = len(self.class_names)
        
        # Create model
        model_config = self.config['model']
        cnn_filters = model_config['cnn_lstm']['cnn_filters']
        lstm_hidden = model_config['cnn_lstm']['lstm_hidden']
        lstm_layers = model_config['cnn_lstm']['lstm_layers']
        dropout = model_config['cnn_lstm']['dropout']
        
        self.model = CNNLSTM(
            cnn_filters=cnn_filters,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            n_classes=self.n_classes,
            img_size=self.img_size,
            dropout=dropout
        )
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize transform
        self.transform = get_test_transforms(self.img_size)
        
        # Initialize buffer for real-time prediction
        self.frame_buffer = deque(maxlen=self.seq_len)
        
        # Smoothing buffer for prediction (majority voting)
        self.prediction_buffer = deque(maxlen=5)
        
        print(f"Loaded model with {self.n_classes} classes: {self.class_names}")
        print(f"Using device: {self.device}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input
        
        Args:
            frame: CV2 BGR frame
        
        Returns:
            Preprocessed tensor of shape [1, C, H, W]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        pil_image = Image.fromarray(frame_rgb).convert('L')  # Convert to grayscale
        
        # Apply transform
        tensor = self.transform(pil_image)
        
        return tensor
    
    def predict_sequence(self, frames_tensor):
        """
        Predict class for a sequence of frames
        
        Args:
            frames_tensor: Tensor of shape [1, seq_len, C, H, W]
        
        Returns:
            Tuple of (class_id, class_name, confidence)
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.model(frames_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class
            confidence, predicted = torch.max(probabilities, dim=1)
            class_id = predicted.item()
            confidence = confidence.item() * 100
            
            return class_id, self.class_names[class_id], confidence
    
    def add_frame(self, frame):
        """
        Add a frame to the buffer and make prediction if buffer is full
        
        Args:
            frame: CV2 BGR frame
        
        Returns:
            Tuple of (class_id, class_name, confidence) or None if buffer not full
        """
        # Preprocess frame
        tensor = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(tensor)
        
        # If buffer is full, make prediction
        if len(self.frame_buffer) == self.seq_len:
            # Stack frames
            frames_tensor = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0)
            frames_tensor = frames_tensor.to(self.device)
            
            # Predict
            prediction = self.predict_sequence(frames_tensor)
            
            # Add to prediction buffer
            self.prediction_buffer.append(prediction[0])  # Add class ID
            
            # Get smoothed prediction (majority voting)
            if len(self.prediction_buffer) > 0:
                # Count occurrences of each class
                counts = {}
                for p in self.prediction_buffer:
                    counts[p] = counts.get(p, 0) + 1
                
                # Get class with most votes
                smoothed_class = max(counts, key=counts.get)
                confidence = prediction[2]  # Use confidence from current prediction
                
                return smoothed_class, self.class_names[smoothed_class], confidence
        
        return None
    
    def predict_from_video(self, video_path, output_path=None, display=True):
        """
        Run prediction on a video file
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video (None to skip saving)
            display: Whether to display the video with predictions
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if output path provided
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Clear buffers
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
        
        # Process video
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make prediction
            prediction = self.add_frame(frame)
            
            # Draw prediction on frame
            if prediction:
                class_id, class_name, confidence = prediction
                label = f"{class_name}: {confidence:.1f}%"
                cv2.putText(frame, label, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display progress
            progress = f"Frame: {frame_idx}/{frame_count}"
            cv2.putText(frame, progress, (10, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            if display:
                cv2.imshow('Prediction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame
            if output_path:
                out.write(frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()
    
    def run_webcam(self, camera_index=0, output_path=None):
        """
        Run real-time prediction on webcam
        
        Args:
            camera_index: Camera index
            output_path: Path to save video (None to skip saving)
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_index}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if output path provided
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Clear buffers
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
        
        # Process video
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Make prediction
            prediction = self.add_frame(frame)
            
            # Draw prediction on frame
            if prediction:
                class_id, class_name, confidence = prediction
                label = f"{class_name}: {confidence:.1f}%"
                cv2.putText(frame, label, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {current_fps}", (width - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Gesture Recognition', frame)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Gesture recognition inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--video', type=str, default=None, help='Path to video file (None for webcam)')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for webcam')
    args = parser.parse_args()
    
    # Create predictor
    predictor = GesturePredictor(args.model, args.config)
    
    # Run inference
    if args.video:
        print(f"Running inference on video: {args.video}")
        predictor.predict_from_video(args.video, args.output)
    else:
        print(f"Running inference on webcam")
        predictor.run_webcam(args.camera, args.output)

if __name__ == '__main__':
    main()

