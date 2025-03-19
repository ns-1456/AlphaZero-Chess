import torch
import chess
import numpy as np
from tqdm import tqdm
from model import ChessNet, encode_board
from mcts import MCTS, move_to_index
import random
from torch.utils.data import Dataset, DataLoader

class ChessDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play_game(model, mcts, device):
    """Play a single game and return training data."""
    board = chess.Board()
    states, policies, values = [], [], []
    
    while not board.is_game_over():
        # Get current state
        state_tensor = encode_board(board).to(device)
        states.append(state_tensor)
        
        # Get MCTS policy
        policy = mcts.search(board)
        policies.append(policy)
        
        # Select move
        if len(board.move_stack) < 30:  # Temperature = 1
            probs = policy.numpy()
            move_idx = np.random.choice(len(probs), p=probs)
        else:  # Temperature = 0
            move_idx = policy.argmax().item()
        
        # Convert move index back to chess move
        for move in board.legal_moves:
            if move_to_index(move) == move_idx:
                board.push(move)
                break
    
    # Get game result
    outcome = board.outcome()
    if outcome is None:
        game_value = 0
    else:
        game_value = 1 if outcome.winner else -1
    
    # Assign values to all states
    current_value = game_value
    for _ in range(len(states)):
        values.append(current_value)
        current_value = -current_value  # Value flips between positions
        
    return states, policies, values

def train_epoch(model, optimizer, dataloader, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    policy_loss_total = 0
    value_loss_total = 0
    num_batches = 0
    
    for states, policies, values in dataloader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        policy_pred, value_pred = model(states)
        
        # Calculate loss
        policy_loss = -torch.sum(policies * torch.log_softmax(policy_pred, dim=1)) / policies.shape[0]
        value_loss = torch.mean((value_pred.squeeze() - values) ** 2)
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, policy_loss_total / num_batches, value_loss_total / num_batches

def train_model(num_iterations=50, games_per_iteration=100, num_epochs=10, batch_size=32):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    mcts = MCTS(model, num_simulations=800, device=device)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Self-play phase
        all_states = []
        all_policies = []
        all_values = []
        
        print("Generating self-play games...")
        for game in tqdm(range(games_per_iteration)):
            states, policies, values = self_play_game(model, mcts, device)
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        # Convert to tensors
        states_tensor = torch.stack(all_states)
        policies_tensor = torch.stack([torch.tensor(p) for p in all_policies])
        values_tensor = torch.tensor(all_values)
        
        # Create dataset and dataloader
        dataset = ChessDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training phase
        print("Training on collected data...")
        for epoch in range(num_epochs):
            total_loss, policy_loss, value_loss = train_epoch(model, optimizer, dataloader, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Loss: {total_loss:.4f} "
                  f"(Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
        
        # Save model checkpoint
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_checkpoint_{iteration + 1}.pt')
        
        print(f"Saved model checkpoint {iteration + 1}")

if __name__ == '__main__':
    train_model()
