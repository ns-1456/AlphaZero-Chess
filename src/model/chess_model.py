import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """A residual block that maintains the input dimensions."""
    
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)

class ChessModel(nn.Module):
    """Neural network for chess position evaluation and move prediction.
    
    Architecture:
    1. Input layer: 73 planes of 8x8 board representation
    2. Initial convolution block
    3. Multiple residual blocks for pattern recognition
    4. Two heads:
       - Policy head: Predicts move probabilities
       - Value head: Evaluates the position
    """
    
    def __init__(self, num_residual_blocks=10):
        super().__init__()
        
        # Initial processing
        self.input_block = ConvBlock(73, 256)  # Changed from 13 to 73 to match move encoding
        
        # Residual tower
        self.residual_tower = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 73, kernel_size=1)  # Output matches move encoding
        )
        
        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 73, 8, 8) representing board states
            
        Returns:
            policy: Move probabilities of shape (batch_size, 73, 8, 8)
            value: Position evaluations of shape (batch_size, 1)
        """
        # Initial convolution
        x = self.input_block(x)
        
        # Residual tower
        for block in self.residual_tower:
            x = block(x)
        
        # Policy head
        policy = self.policy_head(x)
        
        # Value head
        value = self.value_head(x)
        
        return policy, value 