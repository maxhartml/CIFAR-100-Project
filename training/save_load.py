import os
import torch
import warnings
from glob import glob
from configuration.config_1 import *

def ensure_checkpoint_dir_exists(dir_path):
    """
    Ensure the `checkpoints` folder exists in the project directory.
    If the directory does not exist, it will be created.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[INFO] Created directory: {dir_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Save the model, optimizer, scheduler state, and current epoch to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.
        epoch (int): The current epoch to save.
        path (str): The file path to save the checkpoint.

    Returns:
        None
    """
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,  # Last completed epoch
        'model_state_dict': model.state_dict(),  # Model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
        'scheduler_state_dict': scheduler.state_dict()  # Scheduler state
    }

    # Save checkpoint to the specified path
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved successfully at: {path}")

def load_latest_checkpoint(model, optimizer, scheduler):
    """
    Load the latest checkpoint from the `checkpoints` directory.
    If no checkpoints are found, return 0 to start training from scratch.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load state into.
        checkpoint_dir (str): Path to the directory containing checkpoint files.
        device (torch.device): The device to map the loaded model and optimizer states.

    Returns:
        int: The epoch value from the loaded checkpoint, or 0 if no checkpoint is found.
    """
    # Find all checkpoint files matching the naming pattern
    checkpoint_files = glob(os.path.join(CHECKPOINT_DIR, "cifar100-checkpoint-*.pth"))
    if not checkpoint_files:
        print(f"[INFO] No checkpoint files found in {CHECKPOINT_DIR}. Starting from scratch.")
        return 0

    # Sort files by epoch number (extracted from the filename) in descending order
    checkpoint_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]), reverse=True)
    latest_checkpoint = checkpoint_files[0]  # Get the latest checkpoint file

    try:
        # Load the checkpoint file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore specific warnings related to `torch.load`
            checkpoint = torch.load(latest_checkpoint, map_location=torch.device(DEVICE))

        # Restore model, optimizer, and scheduler states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']  # Retrieve the last saved epoch

        print(f"[INFO] Checkpoint loaded successfully from: {latest_checkpoint}")
        print(f"[INFO] Resuming from epoch: {epoch}")
        return epoch

    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {latest_checkpoint}")
        print(f"[ERROR] {e}")
        return 0