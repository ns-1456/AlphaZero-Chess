import os
import sys
import torch
from model import ChessNet
from mcts import MCTS
from train import train_model

def main():
    # Print current directory and contents
    print(f"Current directory: {os.getcwd()}")
    print("Directory contents:", os.listdir())
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # Training configuration
    config = {
        'num_iterations': 2,      # Start with just 2 iterations for testing
        'games_per_iteration': 5, # 5 games per iteration
        'num_epochs': 3,         # 3 training epochs per iteration
        'batch_size': 64         # Batch size of 64
    }
    
    print('\nTraining Configuration:')
    for k, v in config.items():
        print(f'{k}: {v}')
    
    # Start training
    try:
        train_model(
            num_iterations=config['num_iterations'],
            games_per_iteration=config['games_per_iteration'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size']
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
