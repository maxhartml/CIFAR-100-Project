import torch
import torchvision.transforms as transforms
from PIL import Image
from configuration.config_1 import *

def predict_image(model, image_path, classes):
    """
    Predict the class label for a given image using the model.

    Args:
        model (torch.nn.Module): The trained model for inference.
        image_path (str): Path to the input image.
        classes (list): List of class labels corresponding to the model's output.

    Returns:
        str: The predicted class label for the input image.
    """

     # Define the transformations: resize, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # Resize image to 32x32 pixels
        transforms.ToTensor(),       # Convert image to PyTorch tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize to CIFAR-like range
    ])

    # Open the image, ensure it's RGB, and apply transformations
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

    

    # Set the model to evaluation mode
    model.eval()

    # Perform inference without tracking gradients
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)  # Get the class index with the highest score
        predicted_label = classes[predicted.item()]  # Map index to class label

    return predicted_label