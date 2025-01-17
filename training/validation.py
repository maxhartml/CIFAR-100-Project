import torch

def compute_validation_loss(model, valloader, criterion, device):
    """
    Compute the validation loss for the model on the validation set.

    Args:
        model (torch.nn.Module): The trained model.
        valloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): Device to perform computation on.

    Returns:
        float: Average validation loss over the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(valloader)
    return avg_loss