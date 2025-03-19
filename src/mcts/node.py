import math
import chess
import numpy as np

class Node:
    """A node in the MCTS search tree."""
    
    def __init__(self, board, parent=None, prior_probability=0, move=None):
        """Initialize a new node.
        
        Args:
            board: chess.Board object representing the position
            parent: Parent node (None for root)
            prior_probability: Prior probability of selecting this node (from policy network)
            move: chess.Move that led to this position (None for root)
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior_probability = prior_probability
        
        # Tree statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # Map of moves to child nodes
        self.is_expanded = False
        
        # Constants for UCB score calculation
        self.C_PUCT = 1.0  # Exploration constant
        
    def expand(self, policy_probs):
        """Expand the node using policy network output.
        
        Args:
            policy_probs: List of (move, probability) tuples from policy network
        """
        if self.is_expanded:
            return
            
        for move, prob in policy_probs:
            # Create a new board position
            new_board = self.board.copy()
            new_board.push(move)
            
            # Create child node
            self.children[move] = Node(
                board=new_board,
                parent=self,
                prior_probability=prob,
                move=move
            )
            
        self.is_expanded = True
    
    def select_child(self):
        """Select the child node with the highest UCB score."""
        best_score = float('-inf')
        best_child = None
        best_move = None
        
        for move, child in self.children.items():
            ucb_score = self._ucb_score(child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_move = move
                
        return best_child, best_move
    
    def _ucb_score(self, child):
        """Calculate the UCB score for a child node.
        
        UCB = Q + U where:
        Q = child's value estimate
        U = C_PUCT * P * sqrt(N) / (1 + n)
        P = prior probability
        N = parent visit count
        n = child visit count
        """
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = child.value_sum / child.visit_count
            
        # Flip Q-value for opponent's turn (we want to minimize opponent's score)
        if not self.board.turn:
            q_value = -q_value
            
        # Calculate exploration bonus
        u_value = (self.C_PUCT * child.prior_probability * 
                  math.sqrt(self.visit_count) / (1 + child.visit_count))
                  
        return q_value + u_value
    
    def backup(self, value):
        """Update node statistics after a simulation.
        
        Args:
            value: The value to backup (from value network)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent
            
    def is_root(self):
        """Check if this is the root node."""
        return self.parent is None
        
    def get_visit_counts(self):
        """Get distribution of visit counts for all moves.
        
        Returns:
            List of (move, visit_count) tuples sorted by visit count
        """
        visits = [(move, child.visit_count) 
                 for move, child in self.children.items()]
        return sorted(visits, key=lambda x: x[1], reverse=True)
        
    def get_best_move(self, temperature=0.0):
        """Select a move based on visit counts and temperature.
        
        Args:
            temperature: Controls randomness in selection
                       0.0 = select most visited move
                       1.0 = sample proportionally to visit counts
                       
        Returns:
            Selected chess.Move
        """
        visits = self.get_visit_counts()
        if not visits:
            return None
            
        if temperature == 0:
            # Select most visited move
            return visits[0][0]
            
        # Convert visit counts to probabilities
        counts = np.array([count for _, count in visits])
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        
        # Sample move according to probabilities
        move_idx = np.random.choice(len(visits), p=probs)
        return visits[move_idx][0] 