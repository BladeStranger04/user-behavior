import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 24
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 40
BATCH_SIZE = 32