from configuration.config_1 import *

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after the last improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
    """
    def __init__(self):
        self.patience = PATIENCE  # Maximum number of epochs to wait for improvement
        self.delta = 0.0  # Minimum improvement in validation loss
        self.best_loss = None  # Stores the best validation loss observed so far
        self.counter = 0  # Counts epochs without improvement
        self.early_stop = False  # Indicates whether to stop training

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss  # Update the best loss
            self.counter = 0  # Reset the counter if improvement is observed
        else:
            self.counter += 1  # Increment the counter if no improvement
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping if patience is exceeded