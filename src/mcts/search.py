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
        self.device = next(model.parameters()).device
        
    def search(self, board, board_tensor):
        """Search for the best move from current position.
        
        Args:
            board: chess.Board object
            board_tensor: Preprocessed board tensor
                        
        Returns:
            Node: Root node of the search tree
        """
        # Create root node
        root = Node(board.copy())
        
        # If game is over or no legal moves, return root
        if board.is_game_over() or not list(board.legal_moves):
            return root
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            
            # Selection: traverse tree until we find an unexpanded node
            while node.is_expanded and not node.board.is_game_over():
                child, move = node.select_child()
                if child is None or move is None:  # No legal moves
                    break
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
        # Create board tensor
        state = torch.zeros((12, 8, 8), dtype=torch.float32, device=self.device)
        
        # Piece placement planes (12 planes: 6 white pieces + 6 black pieces)
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_plane = piece_idx[piece.symbol()]
                state[piece_plane, rank, file] = 1.0
                
        # Add batch dimension
        return state.unsqueeze(0)
    
    def _get_game_result(self, board):
        """Get game result from terminal position."""
        outcome = board.outcome()
        if outcome is None:
            return 0.0
        elif outcome.winner is None:
            return 0.0
        else:
            return 1.0 if outcome.winner else -1.0 