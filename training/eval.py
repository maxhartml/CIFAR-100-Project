import torch
from torchmetrics.classification import MulticlassAccuracy
from configuration.config_1 import DEVICE

def evaluate_model(model, testloader, classes, writer, epoch):
    """
    Evaluate the trained model on the test dataset and return accuracy metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        classes (list): List of class names corresponding to the dataset.

    Returns:
        dict: Dictionary containing overall accuracy and per-class accuracy.
    """
    # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()

    # Metrics for overall and per-class accuracy
    overall_metric = MulticlassAccuracy(num_classes=len(classes)).to(DEVICE)
    per_class_metric = MulticlassAccuracy(num_classes=len(classes), average=None).to(DEVICE)

    all_predictions = []
    all_labels = []

    # Disable gradient computation during evaluation for efficiency
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass: compute predictions
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)  # Get the predicted class with the highest score

            # Accumulate predictions and labels
            all_predictions.append(predictions)
            all_labels.append(labels)

    # Concatenate all predictions and labels for metric computation
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    overall_accuracy = overall_metric(all_predictions, all_labels).item() * 100  # Overall accuracy in percentage
    per_class_accuracy = per_class_metric(all_predictions, all_labels) * 100  # Per-class accuracy as percentages

    # Print results
    print(f"[INFO] Overall accuracy on the test dataset: {overall_accuracy:.2f}%")
    writer.add_scalar("Accuracy/Overall", overall_accuracy, epoch)
    print("[INFO] Per-class accuracy:")
    for classname, accuracy in zip(classes, per_class_accuracy.tolist()):
        print(f"  - {classname:15s}: {accuracy:.2f}%")
        writer.add_scalar(f"Accuracy/{classname}", accuracy, epoch)

    # Return metrics as a dictionary
    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": dict(zip(classes, per_class_accuracy.tolist()))
    }