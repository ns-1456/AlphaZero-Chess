# AlphaZero-like Chess Model

A PyTorch implementation of an AlphaZero-like model for playing chess, featuring:
- Deep neural network with policy and value heads
- Monte Carlo Tree Search (MCTS)
- Self-play training pipeline

## Requirements
```
numpy==1.24.3
torch==2.1.0
python-chess==1.10.0
tqdm==4.66.1
```

## Project Structure
- `model.py`: Neural network architecture (ResNet with policy and value heads)
- `mcts.py`: Monte Carlo Tree Search implementation
- `train.py`: Training script with self-play and learning pipeline

## Training on Google Colab
1. Upload these files to your Google Drive
2. Create a new Colab notebook
3. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Start training:
```python
!python train.py
```

## Model Architecture
- Input: 8x8x12 board representation (6 piece types × 2 colors)
- ResNet backbone with 19 residual blocks
- Policy head: Predicts move probabilities (1968 possible moves)
- Value head: Evaluates position (-1 to 1)

## Training Process
1. Self-play games using MCTS and current model
2. Collection of (state, policy, value) training data
3. Model training on collected data
4. Repeat with improved model

## Checkpoints
Model checkpoints are saved after each iteration in format: `model_checkpoint_N.pt`
