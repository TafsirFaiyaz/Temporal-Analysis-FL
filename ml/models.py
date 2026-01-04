"""
Neural network models for federated learning experiments.
Author: [Your Name]
Date: [Today's Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple 2-layer CNN for MNIST
    
    Architecture:
        Conv1(1→32) → ReLU → MaxPool → 
        Conv2(32→64) → ReLU → MaxPool → 
        Flatten → FC1(3136→128) → ReLU → 
        FC2(128→10)
    
    Parameters: ~101,770
    Input: (batch, 1, 28, 28)
    Output: (batch, 10)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two pools: 28→14→7, so 64*7*7=3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)  # (batch, 3136)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def num_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


def test_model():
    """Test model creation and forward pass"""
    model = SimpleCNN()
    print(f"Model created: {model.num_parameters():,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 28, 28)  # Batch of 4 images
    output = model(dummy_input)
    
    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
    print(f"Forward pass OK: {dummy_input.shape} → {output.shape}")
    

if __name__ == "__main__":
    test_model()