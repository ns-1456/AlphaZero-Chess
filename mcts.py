import math
import torch
import chess
import numpy as np
from model import encode_board

class Node:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.to_play = True  # True for white, False for black

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, state, to_play, policy):
        """Expand node with new state and policy."""
        self.state = state
        self.to_play = to_play
        policy_sum = 1e-8
        
        # Add children for each legal move
        for move in state.legal_moves:
            move_idx = move_to_index(move)
            if 0 <= move_idx < 1968:  # Valid move index
                prob = policy[move_idx].item()
                policy_sum += prob
                self.children[move] = Node(prior=prob)

        # Normalize probabilities
        for move in self.children:
            self.children[move].prior /= policy_sum

    def select_child(self):
        """Select child node using PUCT algorithm."""
        c_puct = 2.0
        
        best_score = float('-inf')
        best_move = None
        best_child = None

        # Calculate UCB score for each child
        for move, child in self.children.items():
            ucb_score = child.get_ucb_score(self.visit_count, c_puct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child

        return best_move, best_child

    def get_ucb_score(self, parent_visit_count, c_puct):
        """Calculate UCB score for node selection."""
        prior_score = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        value_score = -self.value() if self.visit_count > 0 else 0
        return prior_score + value_score

class MCTS:
    def __init__(self, model, num_simulations=800, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.device = device

    def search(self, state):
        """Perform MCTS search and return policy."""
        root = Node(0)
        
        # Evaluate root state
        tensor_board = encode_board(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(tensor_board)
            policy = torch.softmax(policy, dim=1).squeeze()
        
        # Expand root node
        root.expand(state, state.turn, policy)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # Selection
            while node.children and not current_state.is_game_over():
                move, node = node.select_child()
                current_state.push(move)
                search_path.append(node)
            
            # Expansion and evaluation
            if not current_state.is_game_over():
                tensor_board = encode_board(current_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, value = self.model(tensor_board)
                    policy = torch.softmax(policy, dim=1).squeeze()
                node.expand(current_state, current_state.turn, policy)
                value = value.item()
            else:
                # Game is over, use actual outcome
                outcome = current_state.outcome()
                if outcome is None:
                    value = 0
                else:
                    value = 1 if outcome.winner else -1
                    if not current_state.turn:
                        value = -value
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == state.turn else -value
                node.visit_count += 1
                value = -value  # Value flips between layers
        
        # Calculate policy from visit counts
        policy = torch.zeros(1968)
        for move, child in root.children.items():
            policy[move_to_index(move)] = child.visit_count
        
        # Temperature=1 for first 30 moves, then 0
        if len(state.move_stack) < 30:
            policy = policy / policy.sum()
        else:
            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            policy = torch.zeros(1968)
            policy[move_to_index(best_move)] = 1
            
        return policy

def move_to_index(move):
    """Convert chess move to policy index."""
    from_square = move.from_square
    to_square = move.to_square
    
    # Calculate direction
    from_rank = from_square // 8
    from_file = from_square % 8
    to_rank = to_square // 8
    to_file = to_square % 8
    
    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file
    
    # Handle knight moves
    knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    if (rank_diff, file_diff) in knight_moves:
        move_type = 8 + knight_moves.index((rank_diff, file_diff))
        return from_square * 19 + move_type
    
    # Handle queen moves (including rook and bishop moves)
    queen_moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Normalize direction for queen moves
    if rank_diff != 0:
        rank_diff = rank_diff // abs(rank_diff)
    if file_diff != 0:
        file_diff = file_diff // abs(file_diff)
        
    if (rank_diff, file_diff) in queen_moves:
        move_type = queen_moves.index((rank_diff, file_diff))
        return from_square * 19 + move_type
    
    # Handle promotions (other than queen)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        if move.promotion in promo_pieces:
            move_type = 16 + promo_pieces.index(move.promotion)
            return from_square * 19 + move_type
            
    # If we get here, something went wrong
    return 0  # Safe default
