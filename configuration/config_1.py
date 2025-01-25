import os
import torch

# General Configurations
MODEL_NAME = "CustomResNet18"  # Name of the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU
DATA_DIR = "./data"  # Directory to store data
CHECKPOINT_DIR = "./checkpoints"  # Directory to store model checkpoints
LOG_DIR = "./logs"  # Directory to store logs
NUM_WORKERS = os.cpu_count() - 1  # Use all but one core for data loading
USE_GUI = False  # Set to True to launch the GUI after training

# Data Configurations
BATCH_SIZE = 512  # Number of samples per batch
TRAIN_SPLIT = 0.8  # Proportion of data to use for training
IMAGE_SIZE = (32, 32)  # Size of input images
MEAN = (0.5, 0.5, 0.5)  # Mean for normalization
STD = (0.5, 0.5, 0.5)  # Standard deviation for normalization
AUGMENTATION_PADDING = 4  # Padding for data augmentation

# Training Configurations
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 0.001  # Initial learning rate
SCHEDULER_STEP_SIZE = 20  # Step size for learning rate scheduler
SCHEDULER_GAMMA = 0.1  # Gamma for learning rate scheduler
PATIENCE = 30  # Patience for early stopping
SAVE_INTERVAL = 10  # Interval for saving checkpoints
DROPOUT_RATE = 0.6  # Dropout rate for regularization
WEIGHT_DECAY = 1e-4  # Weight decay for optimizer