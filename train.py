import torch
from src.model.chess_model import ChessModel
from src.model.move_encoding import MoveEncoder
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer
from tqdm import tqdm

def train_model(num_games=10, num_epochs=5):
    print("Initializing model and components...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and move it to device
    model = ChessModel()
    model = model.to(device)
    move_encoder = MoveEncoder(device=device)
    
    print(f"\nGenerating {num_games} self-play games...")
    self_play = SelfPlay(model, move_encoder, games_to_play=num_games)
    
    # Generate training data
    all_training_data = []
    for game_num in tqdm(range(num_games), desc="Generating games"):
        try:
            game_data = self_play.generate_game()
            all_training_data.extend(game_data)
            print(f"\nGame {game_num + 1} completed with {len(game_data)} positions")
        except Exception as e:
            print(f"\nError in game {game_num + 1}: {str(e)}")
            continue
    
    print(f"\nCollected {len(all_training_data)} training positions")
    
    if not all_training_data:
        print("No training data collected. Exiting...")
        return
    
    print("\nTraining model...")
    trainer = Trainer(model)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        try:
            policy_loss, value_loss = trainer.train_epoch(all_training_data)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {str(e)}")
            continue
    
    print("\nSaving model...")
    try:
        trainer.save_model('model.pth')
        print("Training complete! Model saved as model.pth")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == '__main__':
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}") 