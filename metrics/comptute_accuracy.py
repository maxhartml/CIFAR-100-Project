import torch
from torchmetrics.classification import MulticlassAccuracy
from configuration.config_1 import DEVICE

def compute_accuracy(model, dataloader, num_classes=100):
    """
    Compute accuracy for the model on the given dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset (train/validation/test).
        num_classes (int): The number of classes in the dataset.

    Returns:
        float: Accuracy in percentage.
    """
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)
    accuracy = accuracy_metric.compute().item() * 100
    accuracy_metric.reset()
    return accuracy