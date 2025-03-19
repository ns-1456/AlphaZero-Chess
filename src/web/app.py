from flask import Flask, render_template, request, jsonify
import chess
import torch
from ..model.chess_model import ChessModel
from ..model.move_encoding import MoveEncoder
from ..mcts.search import MCTS
import os

app = Flask(__name__)

# Initialize model and components
model = ChessModel()
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
else:
    print("Warning: No trained model found. Using untrained model.")
model.eval()

move_encoder = MoveEncoder()
mcts = MCTS(model, move_encoder, num_simulations=800)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    """Get AI's move for the current position."""
    # Get FEN string from request
    fen = request.json.get('fen')
    if not fen:
        return jsonify({'error': 'No position provided'}), 400
        
    # Create board from FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid position'}), 400
        
    # Check if game is over
    if board.is_game_over():
        return jsonify({
            'game_over': True,
            'result': board.result()
        })
        
    # Get AI's move
    root = mcts.search(board)
    move = root.get_best_move(temperature=0.0)  # Use temperature 0 for strongest play
    
    if move is None:
        return jsonify({'error': 'No legal moves'}), 400
        
    # Make the move
    board.push(move)
    
    return jsonify({
        'move': move.uci(),
        'fen': board.fen(),
        'game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else None
    })

if __name__ == '__main__':
    app.run(debug=True) 