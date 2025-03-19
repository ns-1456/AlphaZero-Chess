# AlphaZero-like Chess Implementation

A Python implementation of a chess AI inspired by DeepMind's AlphaZero. This project uses deep learning and Monte Carlo Tree Search (MCTS) to learn and play chess through self-play.

## Features

- **Neural Network Architecture**
  - Dual-headed network (policy and value heads)
  - Convolutional layers for board processing
  - Policy output for move prediction
  - Value output for position evaluation

- **Monte Carlo Tree Search**
  - Efficient tree search implementation
  - UCB1 exploration formula
  - Parallel game simulation
  - Visit count-based move selection

- **Training Pipeline**
  - Self-play game generation
  - Experience replay buffer
  - Policy and value loss optimization
  - Model checkpointing

- **Web Interface**
  - Interactive chess board
  - Play against AI
  - Move validation
  - Game state visualization

## Project Structure

```
.
├── src/
│   ├── environment/      # Chess game environment
│   ├── model/           # Neural network architecture
│   ├── mcts/            # Monte Carlo Tree Search
│   ├── training/        # Training and self-play
│   └── web/             # Web interface
├── notebooks/           # Jupyter notebooks
├── tests/              # Unit tests
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ns-1456/AlphaZero-like-Chess.git
cd AlphaZero-like-Chess
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

You can train the model in two ways:

### Local Training
```bash
python train.py
```

### Google Colab (Recommended)
1. Open `AlphaZero_Chess.ipynb` in Google Colab
2. Select GPU runtime
3. Run all cells

## Playing Against the AI

1. Start the web server:
```bash
python run_server.py
```

2. Open your browser and visit `http://localhost:8080`

## Training Parameters

- Self-play games: 100
- MCTS simulations per move: 800
- Training epochs: 10
- Batch size: 256
- Temperature:
  - First 30 moves: τ = 1.0 (exploration)
  - After 30 moves: τ = 0.0 (exploitation)

## Dependencies

- Python 3.8+
- PyTorch
- python-chess
- Flask
- tqdm

## License

MIT License

## Acknowledgments

This project is inspired by:
- [DeepMind's AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [Leela Chess Zero](https://lczero.org/) 