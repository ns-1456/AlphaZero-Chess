import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChessDataset(Dataset):
    """Dataset for training the chess model."""
    
    def __init__(self, training_data):
        """Initialize dataset from training data.
        
        Args:
            training_data: List of (state, policy_target, value_target) tuples
        """
        self.states = []
        self.policy_targets = []
        self.value_targets = []
        
        for state, policy, value in training_data:
            self.states.append(state)
            self.policy_targets.append(policy)
            self.value_targets.append(value)
            
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return (self.states[idx], 
                self.policy_targets[idx], 
                torch.tensor([self.value_targets[idx]], dtype=torch.float32))

class Trainer:
    """Trains the chess model using self-play data."""
    
    def __init__(self, model, batch_size=128, learning_rate=0.001):
        """Initialize trainer.
        
        Args:
            model: Neural network model to train
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.batch_size = batch_size
        
        # Loss functions
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, training_data):
        """Train for one epoch on the given data.
        
        Args:
            training_data: List of (state, policy_target, value_target) tuples
            
        Returns:
            tuple: (average policy loss, average value loss)
        """
        dataset = ChessDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=True)
        
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        self.model.train()
        for states, policy_targets, value_targets in dataloader:
            # Forward pass
            policy_output, value_output = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_loss(
                policy_output.view(-1, 73*8*8),
                policy_targets.view(-1, 73*8*8)
            )
            value_loss = self.value_loss(value_output, value_targets)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
        return (total_policy_loss / num_batches,
                total_value_loss / num_batches)
    
    def save_model(self, path):
        """Save model to file."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model from file."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval() 