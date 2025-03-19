import chess
import numpy as np
import torch
from .node import Node

class MCTS:
    """Monte Carlo Tree Search algorithm guided by neural network."""
    
    def __init__(self, model, move_encoder, num_simulations=800):
        """Initialize MCTS search.
        
        Args:
            model: Neural network model (policy + value)
            move_encoder: MoveEncoder instance
            num_simulations: Number of simulations per move
        """
        self.model = model
        self.move_encoder = move_encoder
        self.num_simulations = num_simulations
        
    def search(self, board, temperature=0.0):
        """Search for the best move from current position.
        
        Args:
            board: chess.Board object
            temperature: Temperature for move selection
                        0.0 = select best move
                        1.0 = sample proportionally to visit counts
                        
        Returns:
            Node: Root node of the search tree
        """
        # Create root node
        root = Node(board.copy())
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            
            # Selection: traverse tree until we find an unexpanded node
            while node.is_expanded and not node.board.is_game_over():
                child, move = node.select_child()
                scratch_board.push(move)
                node = child
                
            # Check if game is over
            if node.board.is_game_over():
                value = self._get_game_result(node.board)
                node.backup(value)
                continue
                
            # Expansion and evaluation
            state = self._prepare_state(scratch_board)
            with torch.no_grad():  # Don't track gradients during MCTS
                policy_output, value_output = self.model(state)
            
            # Convert policy output to moves
            policy_probs = self.move_encoder.decode_policy(policy_output[0], scratch_board)
            
            # Expand node with policy predictions
            node.expand(policy_probs)
            
            # Backup value
            value = value_output.item()
            node.backup(value)
            
        return root
    
    def _prepare_state(self, board):
        """Convert board to neural network input format."""
        # Get legal moves for policy target
        legal_moves = list(board.legal_moves)
        
        # Create policy target tensor
        policy_target = self.move_encoder.encode_moves(legal_moves, board)
        
        # Add batch dimension
        return policy_target.unsqueeze(0)
    
    def _get_game_result(self, board):
        """Get game result from terminal position."""
        if board.is_checkmate():
            # Return -1 if side to move is checkmated
            return -1.0 if board.turn else 1.0
        return 0.0  # Draw 