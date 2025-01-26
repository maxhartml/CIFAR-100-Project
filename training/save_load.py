import os
import torch
import warnings
from glob import glob
from configuration.config_1 import *
import json
import time


def ensure_checkpoint_dir_exists(dir_path):
    """Ensure the directory exists; create if not."""
    
    os.makedirs(dir_path, exist_ok=True)
    print(f"[INFO] Directory verified/created: {dir_path}")


def create_training_directory(base_dir, model_name, config):
    """Create a subdirectory for the current training run."""

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    dir_name = f"{model_name}_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)

    # Save configuration to JSON
    config_path = os.path.join(full_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Training directory created: {full_path}")
    print(f"[INFO] Configuration saved to: {config_path}")
    return full_path


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save the model, optimizer, and scheduler states."""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved at: {path}")


def load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """Load the latest checkpoint from the given directory."""

    checkpoint_files = glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*.pth"))
    if not checkpoint_files:
        print(f"[INFO] No checkpoint found in {checkpoint_dir}. Starting from scratch.")
        return 0

    checkpoint_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]), reverse=True)
    latest_checkpoint = checkpoint_files[0]

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore specific warnings related to `torch.load`
            checkpoint = torch.load(latest_checkpoint, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"[INFO] Loaded checkpoint from: {latest_checkpoint} (Epoch {epoch})")
        return epoch
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return 0
