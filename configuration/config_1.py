import os
import torch

# General Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
NUM_WORKERS = os.cpu_count() - 1  # Use all but one core

# Data Configurations
BATCH_SIZE = 256
TRAIN_SPLIT = 0.8
IMAGE_SIZE = (32, 32)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
AUGMENTATION_PADDING = 4

# Training Configurations
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5
PATIENCE = 20
SAVE_INTERVAL = 10
DROPOUT_RATE = 0.6
WEIGHT_DECAY = 1e-4