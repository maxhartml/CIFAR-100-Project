from configuration.config_1 import *

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.

    Args:
        patience (int): How many epochs to wait after the last improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
    """
    def __init__(self, patience = 5, delta=0.0):
        self.patience = PATIENCE
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True