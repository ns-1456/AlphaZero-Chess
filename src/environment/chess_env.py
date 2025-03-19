import chess
import numpy as np

class ChessEnvironment:
    """Chess environment that handles game state and move generation."""
    
    def __init__(self):
        """Initialize a new chess game."""
        self.board = chess.Board()
        
    def reset(self):
        """Reset the board to initial position."""
        self.board = chess.Board()
        return self.get_state()
    
    def get_state(self):
        """Convert current board state to neural network input format.
        
        Returns:
            numpy array of shape (8, 8, 13) representing:
            - 6 planes for white pieces (pawn, knight, bishop, rook, queen, king)
            - 6 planes for black pieces
            - 1 plane for move number
        """
        state = np.zeros((8, 8, 13), dtype=np.float32)
        
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
                state[rank, file, piece_plane] = 1
                
        # Move number plane
        state[:, :, 12] = self.board.fullmove_number / 100.0  # Normalized
        
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
        self.board.push(move)
        
        # Check if game is over
        if self.board.is_game_over():
            if self.board.is_checkmate():
                # Return -1 if black wins, 1 if white wins
                return True, -1.0 if self.board.turn else 1.0
            return True, 0.0  # Draw
            
        return False, 0.0
    
    def render(self):
        """Return string representation of the board."""
        return str(self.board) 