import chess
import numpy as np
import torch
from collections import deque
from ..mcts.search import MCTS

class SelfPlay:
    """Generates training data through self-play games."""
    
    def __init__(self, model, move_encoder, games_to_play=100, 
                 mcts_simulations=800, max_moves=200):
        """Initialize self-play generator.
        
        Args:
            model: Neural network model
            move_encoder: MoveEncoder instance
            games_to_play: Number of games to generate
            mcts_simulations: Number of MCTS simulations per move
            max_moves: Maximum number of moves per game
        """
        self.model = model
        self.move_encoder = move_encoder
        self.games_to_play = games_to_play
        self.mcts = MCTS(model, move_encoder, mcts_simulations)
        self.max_moves = max_moves
        
    def generate_game(self):
        """Play a single game and generate training data.
        
        Returns:
            list: List of (state, policy_target, value_target) tuples
        """
        board = chess.Board()
        states = []
        policies = []
        
        # Keep track of positions and their MCTS visit counts
        game_history = []
        
        # Play until game is over or max moves reached
        move_count = 0
        while not board.is_game_over() and move_count < self.max_moves:
            # Get current state
            state = self.move_encoder.encode_moves(list(board.legal_moves), board)
            states.append(state)
            
            # Run MCTS with temperature
            temperature = 1.0 if move_count < 30 else 0.0
            root = self.mcts.search(board, temperature)
            
            # Store visit count distribution as policy target
            visits = root.get_visit_counts()
            total_visits = sum(count for _, count in visits)
            policy = torch.zeros(73, 8, 8)
            
            for move, count in visits:
                try:
                    square_idx, move_type_idx = self.move_encoder.move_to_policy_index(move, board)
                    row, col = self.square_to_coordinates(square_idx)
                    policy[move_type_idx, row, col] = count / total_visits
                except ValueError:
                    continue
                    
            policies.append(policy)
            
            # Select and make move
            move = root.get_best_move(temperature)
            if move is None:
                break
                
            board.push(move)
            move_count += 1
            
            # Store position and policy
            game_history.append((state, policy))
        
        # Get game result
        if board.is_checkmate():
            value = -1.0 if board.turn else 1.0
        else:
            value = 0.0
            
        # Create training data with correct value targets
        training_data = []
        for state, policy in game_history:
            training_data.append((state, policy, value))
            value = -value  # Flip value for opponent's position
            
        return training_data
    
    def generate_training_data(self):
        """Generate training data from multiple self-play games.
        
        Returns:
            list: List of (state, policy_target, value_target) tuples
        """
        all_training_data = []
        
        for game in range(self.games_to_play):
            game_data = self.generate_game()
            all_training_data.extend(game_data)
            
        return all_training_data
    
    def square_to_coordinates(self, square):
        """Convert a chess.Square to (row, col) coordinates."""
        return (square // 8, square % 8) 