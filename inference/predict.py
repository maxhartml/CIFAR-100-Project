import torch
from inference.preprocess import preprocess_image
from configuration.config_1 import DEVICE

def predict_image(model, image_path, classes):
    """
    Predict the class label for a given image using the model.

    Args:
        model (torch.nn.Module): The trained model for inference.
        image_path (str): Path to the input image.
        device (torch.device): Device to perform inference on (CPU or GPU).
        classes (list): List of class labels corresponding to the model's output.

    Returns:
        str: The predicted class label for the input image.
    """
    # Preprocess the input image
    input_image = preprocess_image(image_path).to(DEVICE)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference without tracking gradients
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)  # Get the class index with the highest score
        predicted_label = classes[predicted.item()]  # Map index to class label

    return predicted_label