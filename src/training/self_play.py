import torch
import chess
from src.environment.chess_env import ChessEnv
from src.mcts.search import MCTS
from tqdm import tqdm

class SelfPlay:
    """Generates training data through self-play games."""
    
    def __init__(self, model, move_encoder, games_to_play=100, mcts_sims=800):
        """Initialize self-play generator.
        
        Args:
            model: Neural network model
            move_encoder: MoveEncoder instance
            games_to_play: Number of games to generate
            mcts_sims: Number of MCTS simulations per move
        """
        self.model = model
        self.move_encoder = move_encoder
        self.games_to_play = games_to_play
        self.mcts_sims = mcts_sims
        self.device = next(model.parameters()).device
        
    def generate_game(self):
        """Play a single game and generate training data.
        
        Returns:
            list: List of dictionaries containing board_tensor, policy, and turn
        """
        env = ChessEnv()
        mcts = MCTS(self.model, self.move_encoder, num_simulations=self.mcts_sims)
        training_data = []
        
        while not env.is_game_over():
            # Get board tensor and move it to the correct device
            board_tensor = env.get_board_tensor()
            board_tensor = board_tensor.to(self.device)
            
            # Run MCTS
            root = mcts.search(env.board, board_tensor)
            
            # Get policy from visit counts
            policy = torch.zeros(1968, device=self.device)
            total_visits = sum(child.visit_count for child in root.children.values())
            
            for move, child in root.children.items():
                move_idx = self.move_encoder.move_to_policy_index(move)
                if move_idx is not None:
                    policy[move_idx] = child.visit_count / total_visits
            
            # Store position
            training_data.append({
                'board_tensor': board_tensor,
                'policy': policy,
                'turn': env.board.turn
            })
            
            # Select and make move
            if len(env.board.move_stack) < 30:  # Temperature = 1
                probs = torch.tensor([child.visit_count for child in root.children.values()], device=self.device)
                probs = probs / probs.sum()
                move_idx = torch.multinomial(probs, 1).item()
                move = list(root.children.keys())[move_idx]
            else:  # Temperature = 0
                move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            
            env.make_move(move)
        
        # Add game result to all positions
        outcome = env.board.outcome()
        if outcome is None:
            result = 0
        elif outcome.winner is None:
            result = 0
        else:
            result = 1 if outcome.winner else -1
            
        for data in training_data:
            data['value'] = result if data['turn'] else -result
        
        return training_data
    
    def generate_training_data(self):
        """Generate training data from multiple self-play games.
        
        Returns:
            list: List of dictionaries containing board_tensor, policy, and value
        """
        all_training_data = []
        for _ in tqdm(range(self.games_to_play), desc="Generating games"):
            try:
                game_data = self.generate_game()
                all_training_data.extend(game_data)
            except Exception as e:
                print(f"\nError in game: {str(e)}")
                continue
        return all_training_data
    
    def square_to_coordinates(self, square):
        """Convert a chess.Square to (row, col) coordinates."""
        return (square // 8, square % 8) 