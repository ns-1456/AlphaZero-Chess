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
            mcts_sims: Number of simulations per move
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.move_encoder = move_encoder
        self.move_encoder.set_device(self.device)
        self.games_to_play = games_to_play
        self.mcts_sims = mcts_sims
        
    def generate_game(self):
        """Play a single game and generate training data.
        
        Returns:
            list: List of dictionaries containing board_tensor, policy, and value
        """
        env = ChessEnv(device=self.device)
        mcts = MCTS(self.model, self.move_encoder, num_simulations=self.mcts_sims)
        training_data = []
        
        while not env.is_game_over():
            # Get board tensor (already on correct device)
            board_tensor = env.get_board_tensor()
            
            # Run MCTS
            root = mcts.search(env.board, board_tensor)
            
            # Get policy from visit counts
            policy = torch.zeros(1968, device=self.device)
            total_visits = sum(child.visit_count for child in root.children.values())
            
            if total_visits > 0:  # Only update policy if there are visits
                for move, child in root.children.items():
                    try:
                        policy_idx = self.move_encoder.move_to_policy_index(move)
                        policy[policy_idx] = child.visit_count / total_visits
                    except ValueError:
                        continue
            
            # Store position
            training_data.append({
                'board_tensor': board_tensor.clone(),  # Clone to avoid memory sharing
                'policy': policy.clone(),  # Clone to avoid memory sharing
                'turn': env.board.turn
            })
            
            # Select and make move
            if len(env.board.move_stack) < 30:  # Temperature = 1
                probs = torch.tensor([child.visit_count for child in root.children.values()], device=self.device)
                if len(probs) > 0:  # Only sample if there are moves
                    probs = probs / probs.sum()
                    move_idx = torch.multinomial(probs, 1).item()
                    move = list(root.children.keys())[move_idx]
                    env.make_move(move)
                else:
                    break  # No legal moves
            else:  # Temperature = 0
                visits = [(move, child.visit_count) for move, child in root.children.items()]
                if visits:  # Only select if there are moves
                    move = max(visits, key=lambda x: x[1])[0]
                    env.make_move(move)
                else:
                    break  # No legal moves
        
        # Add game result to all positions
        outcome = env.board.outcome()
        if outcome is None:
            result = 0
        elif outcome.winner is None:
            result = 0
        else:
            result = 1 if outcome.winner else -1
            
        for data in training_data:
            data['value'] = torch.tensor([result if data['turn'] else -result], dtype=torch.float32, device=self.device)
        
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