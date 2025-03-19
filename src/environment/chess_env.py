import chess
import torch
import numpy as np

class ChessEnv:
    """Chess environment that handles game state and move generation."""
    
    def __init__(self, device=None):
        """Initialize a new chess game.
        
        Args:
            device: torch.device to use for tensors
        """
        self.board = chess.Board()
        self.device = device if device is not None else torch.device('cpu')
        
    def reset(self):
        """Reset the board to initial position."""
        self.board = chess.Board()
        return self.get_board_tensor()
    
    def get_board_tensor(self):
        """Convert current board state to neural network input format.
        
        Returns:
            torch.Tensor of shape (12, 8, 8) representing:
            - 6 planes for white pieces (pawn, knight, bishop, rook, queen, king)
            - 6 planes for black pieces
        """
        # Initialize tensor on the correct device
        state = torch.zeros((12, 8, 8), dtype=torch.float32, device=self.device)
        
        # Piece placement planes (12 planes: 6 white pieces + 6 black pieces)
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_plane = piece_idx[piece.symbol()]
                state[piece_plane, rank, file] = 1.0
                
        return state
    
    def get_valid_moves(self):
        """Get list of valid moves in current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        """Make a move on the board.
        
        Args:
            move: A chess.Move object representing the move to make
            
        Returns:
            done (bool): Whether the game is over
            reward (float): Game outcome if game is over (1.0 white wins, -1.0 black wins, 0.0 draw)
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
            
        self.board.push(move)
        
        # Check if game is over
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome is None:
                return True, 0.0
            elif outcome.winner is None:
                return True, 0.0
            else:
                return True, 1.0 if outcome.winner else -1.0
            
        return False, 0.0
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
        
    def render(self):
        """Return string representation of the board."""
        return str(self.board) 