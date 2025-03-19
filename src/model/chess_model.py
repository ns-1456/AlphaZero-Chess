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
    1. Input layer: 12 planes of 8x8 board representation
    2. Initial convolution block
    3. Multiple residual blocks for pattern recognition
    4. Two heads:
       - Policy head: Predicts move probabilities
       - Value head: Evaluates the position
    """
    
    def __init__(self, device=None):
        super(ChessModel, self).__init__()
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input: 8x8x12 (6 piece types x 2 colors)
        self.conv1 = ConvBlock(12, 256)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        
        # Policy head
        self.policy_conv = ConvBlock(256, 256)
        self.policy_fc = nn.Linear(256 * 8 * 8, 1968)  # 1968 possible moves
        
        # Value head
        self.value_conv = ConvBlock(256, 1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Move model to device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Shared layers
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(-1, 256 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(x)
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def predict(self, board_tensor):
        # Ensure input is a batch
        if len(board_tensor.shape) == 3:
            board_tensor = board_tensor.unsqueeze(0)
        
        # Move input to same device as model
        board_tensor = board_tensor.to(self.device)
        
        self.eval()
        with torch.no_grad():
            policy, value = self(board_tensor)
        return policy, value 