import torch

def evaluate_model(model, testloader, device, classes):
    """
    Evaluate the trained model on the test dataset and print accuracy metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device on which computations are performed (CPU or GPU).
        classes (list): List of class names corresponding to the dataset.

    Returns:
        None
    """
    # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()

    # Initialize counters for overall and per-class accuracy
    correct = 0
    total = 0
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    # Disable gradient computation during evaluation for efficiency
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: compute predictions
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)  # Get the predicted class with the highest score

            # Update overall accuracy counters
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Update per-class accuracy counters
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    class_correct[classes[label]] += 1
                class_total[classes[label]] += 1
    
    # Calculate and print overall accuracy
    accuracy = 100 * correct / total
    print(f"[INFO] Overall accuracy on the test dataset: {accuracy:.2f}%")

    # Calculate and print accuracy for each class
    print("[INFO] Per-class accuracy:")
    for classname, correct_count in class_correct.items():
        if class_total[classname] > 0:  # Avoid division by zero
            class_accuracy = 100 * float(correct_count) / class_total[classname]
            print(f"  - {classname:15s}: {class_accuracy:.2f}%")
        else:
            print(f"  - {classname:15s}: No samples in test set")