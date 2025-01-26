import os
import torch

# General Configurations
MODEL_NAME = "CustomResNet18"  # Name of the model (CustomResNet18 / CNN)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU
DATA_DIR = "./data"  # Directory to store data
CHECKPOINT_DIR = "./checkpoints"  # Directory to store model checkpoints
LOG_DIR = "./logs"  # Directory to store logs
NUM_WORKERS = os.cpu_count() - 1  # Use all but one core for data loading
USE_GUI = False  # Set to True to launch the GUI after training

# Data Configurations
BATCH_SIZE = 512  # Number of samples per batch
IMAGE_SIZE = (32, 32)  # Size of input images
MEAN = (0.5, 0.5, 0.5)  # Mean for normalization
STD = (0.5, 0.5, 0.5)  # Standard deviation for normalization
AUGMENTATION_PADDING = 4  # Padding for data augmentation

# Training Configurations
NUM_EPOCHS = 150  # Number of training epochs
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate
ETA_MIN=1e-5  # Minimum learning rate for CosineAnnealingLR
PATIENCE = 30  # Patience for early stopping
SAVE_INTERVAL = 10  # Interval for saving checkpoints
DROPOUT_RATE = 0.3  # Dropout rate for regularization
WEIGHT_DECAY = 1e-3  # Weight decay for optimizer
GRAD_CLIP = 2.0  # Gradient clipping value (can try 1 later)
LABEL_SMOOTHING = 0.1  # Label smoothing factor