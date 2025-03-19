import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class ChessDataset(Dataset):
    """Dataset for training the chess model."""
    
    def __init__(self, training_data):
        """Initialize dataset from training data.
        
        Args:
            training_data: List of (state, policy_target, value_target) tuples
        """
        self.training_data = training_data
        
    def __len__(self):
        return len(self.training_data)
        
    def __getitem__(self, idx):
        data = self.training_data[idx]
        return {
            'board': data['board_tensor'],
            'policy': data['policy'],
            'value': torch.tensor([data['value']], dtype=torch.float32).to(data['board_tensor'].device)
        }

class Trainer:
    """Trains the chess model using self-play data."""
    
    def __init__(self, model, batch_size=256, learning_rate=0.001):
        """Initialize trainer.
        
        Args:
            model: Neural network model to train
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.device = next(model.parameters()).device
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
        if not training_data:
            raise ValueError("No training data provided")
            
        # Create dataset and dataloader
        dataset = ChessDataset(training_data)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        for batch in dataloader:
            # Get data
            boards = batch['board']
            target_policies = batch['policy']
            target_values = batch['value']
            
            # Forward pass
            policy_logits, value_pred = self.model(boards)
            
            # Calculate losses
            policy_loss = -torch.mean(torch.sum(target_policies * policy_logits, dim=1))
            value_loss = torch.mean((value_pred - target_values) ** 2)
            total_loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        # Calculate average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        return avg_policy_loss, avg_value_loss
    
    def save_model(self, path):
        """Save model to file."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model from file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval() 