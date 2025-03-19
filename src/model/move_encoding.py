import chess
import torch
import numpy as np

class MoveEncoder:
    """Handles conversion between chess moves and neural network policy outputs."""
    
    def __init__(self, device=None):
        """Initialize move encoder.
        
        Args:
            device: torch.device to use for tensors
        """
        self.device = device if device is not None else torch.device('cpu')
        
        # Define move directions
        self.KNIGHT_MOVES = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        self.QUEEN_MOVES = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Index mappings for move types
        self.move_index = {
            'queen': list(range(8)),      # First 8 indices for queen-like moves
            'knight': list(range(8, 16)), # Next 8 indices for knight moves
            'underpromotion': list(range(16, 19))  # Last 3 for underpromotions (N, B, R)
        }
        
        # Promotion pieces (excluding queen, which is handled in queen-like moves)
        self.promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    
    def set_device(self, device):
        """Set the device for tensor operations."""
        self.device = device
    
    def square_to_coordinates(self, square):
        """Convert a chess.Square to (row, col) coordinates."""
        return (square // 8, square % 8)
    
    def coordinates_to_square(self, row, col):
        """Convert (row, col) coordinates to a chess.Square."""
        if 0 <= row < 8 and 0 <= col < 8:
            return row * 8 + col
        return None
    
    def move_to_policy_index(self, move, board=None):
        """Convert a chess move to policy index.
        
        Args:
            move: chess.Move object
            board: chess.Board object (optional)
        
        Returns:
            int: Policy index (0-1967)
        """
        if move is None:
            raise ValueError("Move cannot be None")
            
        from_square = move.from_square
        to_square = move.to_square
        
        from_row, from_col = self.square_to_coordinates(from_square)
        to_row, to_col = self.square_to_coordinates(to_square)
        
        # Calculate move direction
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # Check if it's a knight move
        if (row_diff, col_diff) in self.KNIGHT_MOVES:
            move_type_idx = self.move_index['knight'][self.KNIGHT_MOVES.index((row_diff, col_diff))]
            return from_square * 73 + move_type_idx
        
        # Check for underpromotions (not to queen)
        if move.promotion and move.promotion != chess.QUEEN:
            try:
                promo_idx = self.promotion_pieces.index(move.promotion)
                move_type_idx = self.move_index['underpromotion'][promo_idx]
                return from_square * 73 + move_type_idx
            except ValueError:
                raise ValueError(f"Invalid promotion piece: {move.promotion}")
        
        # Handle queen-like moves
        if max(abs(row_diff), abs(col_diff)) > 0:
            # Normalize direction
            if row_diff != 0:
                row_diff = row_diff // abs(row_diff)
            if col_diff != 0:
                col_diff = col_diff // abs(col_diff)
            
            if (row_diff, col_diff) in self.QUEEN_MOVES:
                move_type_idx = self.move_index['queen'][self.QUEEN_MOVES.index((row_diff, col_diff))]
                return from_square * 73 + move_type_idx
        
        # If we get here, something went wrong
        raise ValueError(f"Could not encode move {move}")
    
    def policy_index_to_moves(self, policy_idx, board):
        """Convert policy index to possible chess moves.
        
        Args:
            policy_idx: Policy index (0-1967)
            board: chess.Board object for context
        
        Returns:
            list: Possible chess.Move objects
        """
        square_idx = policy_idx // 73
        move_type_idx = policy_idx % 73
        
        from_row, from_col = self.square_to_coordinates(square_idx)
        possible_moves = []
        
        # Handle queen-like moves
        if move_type_idx < 8:
            direction = self.QUEEN_MOVES[move_type_idx]
            curr_row, curr_col = from_row, from_col
            
            # Keep going in the direction until we hit a piece or the board edge
            while True:
                curr_row += direction[0]
                curr_col += direction[1]
                to_square = self.coordinates_to_square(curr_row, curr_col)
                
                if to_square is None:  # Off the board
                    break
                    
                # Create move and check if it's legal
                move = chess.Move(square_idx, to_square)
                if move in board.legal_moves:
                    possible_moves.append(move)
                    
                    # Check for possible queen promotion
                    if board.piece_at(square_idx) and board.piece_at(square_idx).piece_type == chess.PAWN:
                        if (board.turn and curr_row == 7) or (not board.turn and curr_row == 0):
                            move = chess.Move(square_idx, to_square, promotion=chess.QUEEN)
                            if move in board.legal_moves:
                                possible_moves.append(move)
                
                # Stop if we hit a piece
                if board.piece_at(to_square) is not None:
                    break
        
        # Handle knight moves
        elif move_type_idx < 16:
            knight_idx = move_type_idx - 8
            direction = self.KNIGHT_MOVES[knight_idx]
            to_row = from_row + direction[0]
            to_col = from_col + direction[1]
            to_square = self.coordinates_to_square(to_row, to_col)
            
            if to_square is not None:
                move = chess.Move(square_idx, to_square)
                if move in board.legal_moves:
                    possible_moves.append(move)
        
        # Handle underpromotions
        else:
            promo_idx = move_type_idx - 16
            if promo_idx < len(self.promotion_pieces):
                piece = self.promotion_pieces[promo_idx]
                
                # Try both forward moves for pawns
                pawn_moves = [(1, 0), (1, 1), (1, -1)] if board.turn else [(-1, 0), (-1, 1), (-1, -1)]
                for direction in pawn_moves:
                    to_row = from_row + direction[0]
                    to_col = from_col + direction[1]
                    to_square = self.coordinates_to_square(to_row, to_col)
                    
                    if to_square is not None and ((board.turn and to_row == 7) or (not board.turn and to_row == 0)):
                        move = chess.Move(square_idx, to_square, promotion=piece)
                        if move in board.legal_moves:
                            possible_moves.append(move)
        
        return possible_moves

    def encode_moves(self, legal_moves, board):
        """Create a policy target tensor for a given position.
        
        Args:
            legal_moves: List of legal chess.Move objects
            board: chess.Board object
            
        Returns:
            torch.Tensor: Policy target of shape (1968,)
        """
        policy = torch.zeros(1968, dtype=torch.float32, device=self.device)
        
        for move in legal_moves:
            try:
                policy_idx = self.move_to_policy_index(move)
                policy[policy_idx] = 1.0
            except ValueError:
                continue
                
        return policy
    
    def decode_policy(self, policy_output, board):
        """Convert policy network output to a list of moves with probabilities.
        
        Args:
            policy_output: torch.Tensor of shape (1968,)
            board: chess.Board object
            
        Returns:
            list: (move, probability) tuples sorted by probability
        """
        moves_with_probs = []
        
        # Convert policy output to probabilities
        policy_probs = torch.softmax(policy_output, dim=0)
        
        # Find all non-zero probabilities
        for idx in range(1968):
            prob = policy_probs[idx].item()
            if prob > 0:
                possible_moves = self.policy_index_to_moves(idx, board)
                for move in possible_moves:
                    if move in board.legal_moves:
                        moves_with_probs.append((move, prob))
        
        # Sort by probability
        moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        return moves_with_probs 