import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for hand gesture recognition
    
    Architecture:
    1. CNN for feature extraction from each frame
    2. LSTM for temporal modeling of the sequence
    3. Fully connected layers for classification
    """
    def __init__(self, cnn_filters, lstm_hidden, lstm_layers, n_classes, img_size=64, dropout=0.5):
        super(CNNLSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn_layers = self._make_cnn_layers(cnn_filters)
        
        # Calculate CNN output size
        cnn_out_size = self._get_cnn_output_size(cnn_filters, img_size)
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_classes)
        )
    
    def _make_cnn_layers(self, cnn_filters):
        """Create CNN layers for feature extraction"""
        layers = []
        in_channels = 1  # Grayscale input
        
        for out_channels in cnn_filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _get_cnn_output_size(self, cnn_filters, img_size):
        """Calculate CNN output size based on input dimensions and network architecture"""
        # For each MaxPool2d layer, the spatial dimensions are halved
        output_size = img_size // (2 ** len(cnn_filters))
        
        # Final output channels from CNN
        output_channels = cnn_filters[-1]
        
        # Total flattened feature size
        return output_channels * output_size * output_size
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channels, height, width]
        
        Returns:
            Logits of shape [batch_size, n_classes]
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for CNN processing
        # Combine batch and sequence dimensions
        cnn_input = x.view(-1, channels, height, width)
        
        # Pass through CNN layers
        cnn_output = self.cnn_layers(cnn_input)
        
        # Flatten CNN output
        cnn_output = cnn_output.view(cnn_output.size(0), -1)
        
        # Reshape back to [batch_size, seq_len, features]
        lstm_input = cnn_output.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(lstm_input)
        
        # Take the output from the last time step
        lstm_last_output = lstm_output[:, -1, :]
        
        # Classification
        output = self.fc(lstm_last_output)
        
        return output 