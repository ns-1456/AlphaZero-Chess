import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 8x8x12 (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(19)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.policy_bn = nn.BatchNorm2d(256)
        self.policy_fc = nn.Linear(256 * 8 * 8, 1968)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 256 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

def encode_board(board):
    """Convert chess board to input tensor."""
    piece_chars = 'pnbrqkPNBRQK'
    piece_map = {piece: i for i, piece in enumerate(piece_chars)}
    
    # Initialize 12 planes of 8x8
    planes = torch.zeros(12, 8, 8)
    
    # Fill in piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            piece_idx = piece_map[piece.symbol()]
            planes[piece_idx][rank][file] = 1
            
    return planes
