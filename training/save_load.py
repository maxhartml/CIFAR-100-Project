import torch
import warnings

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch to save.
        path (str): The file path to save the checkpoint.

    Returns:
        None
    """
    checkpoint = {
        'epoch': epoch,  # Last completed epoch
        'model_state_dict': model.state_dict(),  # Model weights
        'optimizer_state_dict': optimizer.state_dict()  # Optimizer state
    }
    torch.save(checkpoint, path)  # Save checkpoint to the specified path
    print(f"[INFO] Checkpoint saved successfully at: {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        path (str): The file path to load the checkpoint from.
        device (torch.device): The device to map the loaded model and optimizer states.

    Returns:
        int: The epoch value from the loaded checkpoint, or 0 if no checkpoint is found.
    """
    try:
        # Suppress specific warnings related to `torch.load` and weights_only=False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(path, map_location=torch.device(device))

        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Retrieve the epoch from the checkpoint
        epoch = checkpoint['epoch']
        print(f"[INFO] Checkpoint loaded successfully from: {path}")
        print(f"[INFO] Resuming from epoch: {epoch}")
        return epoch

    except FileNotFoundError:
        print(f"[ERROR] Checkpoint file not found at: {path}. Starting from scratch.")
        return 0