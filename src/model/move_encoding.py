import chess
import torch
import numpy as np

class MoveEncoder:
    """Handles conversion between chess moves and neural network policy outputs."""
    
    def __init__(self):
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
    
    def square_to_coordinates(self, square):
        """Convert a chess.Square to (row, col) coordinates."""
        return (square // 8, square % 8)
    
    def coordinates_to_square(self, row, col):
        """Convert (row, col) coordinates to a chess.Square."""
        if 0 <= row < 8 and 0 <= col < 8:
            return row * 8 + col
        return None
    
    def move_to_policy_index(self, move, board):
        """Convert a chess move to policy indices (square_index, move_type_index).
        
        Args:
            move: chess.Move object
            board: chess.Board object for context
        
        Returns:
            tuple: (square_index, move_type_index) where:
                  square_index is the starting square (0-63)
                  move_type_index is the type of move (0-72)
        """
        from_square = move.from_square
        to_square = move.to_square
        
        from_row, from_col = self.square_to_coordinates(from_square)
        to_row, to_col = self.square_to_coordinates(to_square)
        
        # Calculate move direction
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # Check if it's a knight move
        if (row_diff, col_diff) in self.KNIGHT_MOVES:
            move_index = self.move_index['knight'][self.KNIGHT_MOVES.index((row_diff, col_diff))]
            return (from_square, move_index)
        
        # Check if it's a queen-like move
        if max(abs(row_diff), abs(col_diff)) > 0:
            # Normalize direction
            if row_diff != 0:
                row_diff //= abs(row_diff)
            if col_diff != 0:
                col_diff //= abs(col_diff)
            
            if (row_diff, col_diff) in self.QUEEN_MOVES:
                move_index = self.move_index['queen'][self.QUEEN_MOVES.index((row_diff, col_diff))]
                return (from_square, move_index)
        
        # Check for underpromotions (not to queen)
        if move.promotion and move.promotion != chess.QUEEN:
            try:
                promo_idx = self.promotion_pieces.index(move.promotion)
                return (from_square, self.move_index['underpromotion'][promo_idx])
            except ValueError:
                pass
        
        # If we get here, something went wrong
        raise ValueError(f"Could not encode move {move}")
    
    def policy_index_to_moves(self, square_index, move_type_index, board):
        """Convert policy indices to possible chess moves.
        
        Args:
            square_index: Starting square (0-63)
            move_type_index: Type of move (0-72)
            board: chess.Board object for context
        
        Returns:
            list: Possible chess.Move objects
        """
        from_row, from_col = self.square_to_coordinates(square_index)
        possible_moves = []
        
        # Handle queen-like moves
        if move_type_index < 8:
            direction = self.QUEEN_MOVES[move_type_index]
            curr_row, curr_col = from_row, from_col
            
            # Keep going in the direction until we hit a piece or the board edge
            while True:
                curr_row += direction[0]
                curr_col += direction[1]
                to_square = self.coordinates_to_square(curr_row, curr_col)
                
                if to_square is None:  # Off the board
                    break
                    
                # Create move and check if it's legal
                move = chess.Move(square_index, to_square)
                if move in board.legal_moves:
                    possible_moves.append(move)
                    
                    # Check for possible queen promotion
                    if board.piece_at(square_index) and board.piece_at(square_index).piece_type == chess.PAWN:
                        if (board.turn and curr_row == 7) or (not board.turn and curr_row == 0):
                            move = chess.Move(square_index, to_square, promotion=chess.QUEEN)
                            if move in board.legal_moves:
                                possible_moves.append(move)
                
                # Stop if we hit a piece
                if board.piece_at(to_square) is not None:
                    break
        
        # Handle knight moves
        elif move_type_index < 16:
            knight_idx = move_type_index - 8
            direction = self.KNIGHT_MOVES[knight_idx]
            to_row = from_row + direction[0]
            to_col = from_col + direction[1]
            to_square = self.coordinates_to_square(to_row, to_col)
            
            if to_square is not None:
                move = chess.Move(square_index, to_square)
                if move in board.legal_moves:
                    possible_moves.append(move)
        
        # Handle underpromotions
        else:
            promo_idx = move_type_index - 16
            if promo_idx < len(self.promotion_pieces):
                piece = self.promotion_pieces[promo_idx]
                
                # Try both forward moves for pawns
                pawn_moves = [(1, 0), (1, 1), (1, -1)] if board.turn else [(-1, 0), (-1, 1), (-1, -1)]
                for direction in pawn_moves:
                    to_row = from_row + direction[0]
                    to_col = from_col + direction[1]
                    to_square = self.coordinates_to_square(to_row, to_col)
                    
                    if to_square is not None and ((board.turn and to_row == 7) or (not board.turn and to_row == 0)):
                        move = chess.Move(square_index, to_square, promotion=piece)
                        if move in board.legal_moves:
                            possible_moves.append(move)
        
        return possible_moves

    def encode_moves(self, legal_moves, board):
        """Create a policy target tensor for a given position.
        
        Args:
            legal_moves: List of legal chess.Move objects
            board: chess.Board object
            
        Returns:
            torch.Tensor: Policy target of shape (73, 8, 8)
        """
        policy = torch.zeros(73, 8, 8)
        
        for move in legal_moves:
            try:
                square_idx, move_type_idx = self.move_to_policy_index(move, board)
                row, col = self.square_to_coordinates(square_idx)
                policy[move_type_idx, row, col] = 1.0
            except ValueError:
                continue
                
        return policy
    
    def decode_policy(self, policy_output, board):
        """Convert policy network output to a list of moves with probabilities.
        
        Args:
            policy_output: torch.Tensor of shape (73, 8, 8)
            board: chess.Board object
            
        Returns:
            list: (move, probability) tuples sorted by probability
        """
        moves_with_probs = []
        
        # Convert policy output to probabilities
        policy_probs = torch.softmax(policy_output.view(-1), dim=0).view(73, 8, 8)
        
        # Find all non-zero probabilities
        for move_type in range(73):
            for row in range(8):
                for col in range(8):
                    prob = policy_probs[move_type, row, col].item()
                    if prob > 0:
                        square_idx = self.coordinates_to_square(row, col)
                        possible_moves = self.policy_index_to_moves(square_idx, move_type, board)
                        
                        for move in possible_moves:
                            if move in board.legal_moves:
                                moves_with_probs.append((move, prob))
        
        # Sort by probability
        moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        return moves_with_probs 