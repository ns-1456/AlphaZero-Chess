import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import math
import time
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# ====== MODEL CODE ======
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=5, channels=128):  # Smaller for faster training
        super(ChessNet, self).__init__()
        
        # Input: 19 * 8 * 8
        # Initial convolution
        self.conv_input = nn.Conv2d(19, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy head (output: 1968 moves)
        self.conv_policy = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, 1968)
        
        # Value head (output: 1 value)
        self.conv_value = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 64)
        self.fc_value2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.fc_policy(policy)
        
        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(-1, 1 * 8 * 8)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return policy, value

def encode_board(board):
    """Encode chess board into tensor for neural network input."""
    # 12 piece planes for each piece type and color (6 * 2)
    # 4 castling rights, 1 side to move, 1 move count, 1 no-progress count
    pieces_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Initialize 19 planes of 8x8 board
    state = torch.zeros(19, 8, 8, dtype=torch.float)
    
    # Fill piece planes (12 channels)
    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        piece_idx = pieces_map[piece.symbol()]
        state[piece_idx][rank][file] = 1
    
    # Fill auxiliary planes
    # Castling rights (4 channels)
    state[12][0][0] = float(board.has_kingside_castling_rights(chess.WHITE))
    state[13][0][0] = float(board.has_queenside_castling_rights(chess.WHITE))
    state[14][0][0] = float(board.has_kingside_castling_rights(chess.BLACK))
    state[15][0][0] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Side to move (1 channel)
    if board.turn == chess.WHITE:
        state[16].fill_(1)
    
    # Move count (1 channel)
    state[17].fill_(min(board.fullmove_number / 50.0, 1.0))
    
    # No-progress count (1 channel)
    state[18].fill_(min(board.halfmove_clock / 100.0, 1.0))
    
    return state

# ====== MOVE ENCODING ======
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
    return 0

# Function for multiprocessing - must be at module level to be picklable
def run_game(args):
    model_state, num_simulations, device_str, game_id = args
    
    # Create a new model and load state dict
    device = torch.device(device_str)
    game_model = ChessNet().to(device)
    game_model.load_state_dict(model_state)
    
    return self_play_game(game_model, num_simulations, device_str, game_id)

# ====== MCTS CODE ======
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
    def __init__(self, model, num_simulations=100, device='cpu'):
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

# ====== TRAINING CODE ======
class ChessDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play_game(model, num_simulations, device_str, game_id):
    """Play a single game and return training data."""
    device = torch.device(device_str)
    if torch.cuda.is_available() and device_str == 'cuda':
        # Create a new model instance for this process
        game_model = ChessNet().to(device)
        game_model.load_state_dict(model.state_dict())
        mcts = MCTS(game_model, num_simulations=num_simulations, device=device)
    else:
        # On CPU, just use the provided model
        mcts = MCTS(model, num_simulations=num_simulations, device=device)

    board = chess.Board()
    states, policies, values = [], [], []
    
    while not board.is_game_over():
        # Get current state
        state_tensor = encode_board(board).to(device)
        states.append(state_tensor.cpu())  # Store on CPU
        
        # Get MCTS policy
        policy = mcts.search(board)
        policies.append(policy.cpu())  # Store on CPU
        
        # Select move
        if len(board.move_stack) < 30:  # Temperature = 1
            probs = policy.cpu().numpy()
            move_idx = np.random.choice(len(probs), p=probs)
        else:  # Temperature = 0
            move_idx = policy.argmax().item()
        
        # Convert move index back to chess move
        moved = False
        for move in board.legal_moves:
            if move_to_index(move) == move_idx:
                board.push(move)
                moved = True
                break
        
        # If we couldn't find the exact move, just pick a random one
        if not moved and not board.is_game_over():
            moves = list(board.legal_moves)
            if moves:
                board.push(np.random.choice(moves))
    
    # Get game result
    outcome = board.outcome()
    if outcome is None:
        game_value = 0
    else:
        game_value = 1 if outcome.winner else -1
    
    # Assign values to all states
    current_value = game_value
    for _ in range(len(states)):
        values.append(current_value)
        current_value = -current_value  # Value flips between positions
    
    print(f"Game {game_id+1} completed with {len(states)} positions")
    return states, policies, values

def parallel_self_play(model, num_games, num_simulations, device_str):
    """Generate self-play games in parallel"""
    start_time = time.time()
    print(f"Starting {num_games} parallel self-play games...")
    
    # Determine number of processes
    num_workers = min(multiprocessing.cpu_count(), num_games)
    print(f"Using {num_workers} workers")
    
    # Make model's state dict serializable for multiprocessing
    model_state = model.state_dict()
    
    # Prepare arguments for each game
    args = [(model_state, num_simulations, device_str, i) for i in range(num_games)]
    
    # Run games in parallel
    game_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        game_results = list(executor.map(run_game, args))
    
    # Combine results
    all_states = []
    all_policies = []
    all_values = []
    
    for states, policies, values in game_results:
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
    
    elapsed = time.time() - start_time
    print(f"Self-play completed in {elapsed:.1f} seconds. Collected {len(all_states)} positions.")
    
    return all_states, all_policies, all_values

def train_epoch(model, optimizer, dataloader, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    policy_loss_total = 0
    value_loss_total = 0
    num_batches = 0
    
    for states, policies, values in dataloader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        policy_pred, value_pred = model(states)
        
        # Calculate loss
        policy_loss = -torch.sum(policies * torch.log_softmax(policy_pred, dim=1)) / policies.shape[0]
        value_loss = torch.mean((value_pred.squeeze() - values) ** 2)
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, policy_loss_total / num_batches, value_loss_total / num_batches

def train_model(num_iterations=2, games_per_iteration=5, num_epochs=3, batch_size=64, num_simulations=100):
    """Main training loop with parallel self-play."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Self-play phase
        all_states, all_policies, all_values = parallel_self_play(
            model, games_per_iteration, num_simulations, device.type)
        
        # Convert to tensors
        states_tensor = torch.stack(all_states)
        policies_tensor = torch.stack(all_policies)
        values_tensor = torch.tensor(all_values)
        
        # Create dataset and dataloader
        dataset = ChessDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training phase
        print("Training on collected data...")
        for epoch in range(num_epochs):
            total_loss, policy_loss, value_loss = train_epoch(model, optimizer, dataloader, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Loss: {total_loss:.4f} "
                  f"(Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
        
        # Save model checkpoint
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_checkpoint_{iteration + 1}.pt')
        
        print(f"Saved model checkpoint {iteration + 1}")

if __name__ == "__main__":
    # Parse arguments and run training
    import argparse
    parser = argparse.ArgumentParser(description='Train AlphaZero Chess model with parallelization')
    parser.add_argument('--iterations', type=int, default=2, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=5, help='Number of self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=100, help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()
    
    print(f"Training with config:")
    print(f"- Iterations: {args.iterations}")
    print(f"- Games per iteration: {args.games}")
    print(f"- MCTS simulations per move: {args.simulations}")
    print(f"- Epochs per iteration: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    
    train_model(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
