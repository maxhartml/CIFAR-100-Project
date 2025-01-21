import torchvision
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from configuration.config_1 import *
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def get_cifar100_loaders():
    """
    Prepare DataLoaders for the CIFAR-100 dataset with transformations.

    Args:
        batch_size (int): Number of samples per batch for the DataLoader.

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for the CIFAR-100 training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the CIFAR-100 test dataset.
        classes (list): List of class names for the CIFAR-100 dataset.
    """
    # ---------------------------------------------------
    # Define Transformations
    # ---------------------------------------------------
    # Data augmentation and normalization for training and test datasets.
    # Training: Adds random cropping and horizontal flipping for data augmentation.
    # Test: Only normalization, no data augmentation.
  
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=AUGMENTATION_PADDING), # Add random cropping
        transforms.RandomHorizontalFlip(), # Add random horizontal flipping
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # simulate lighting changes
        transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
        transforms.Normalize(MEAN, STD), # Normalize to range [-1, 1]
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3,3.3)),     # Add random erasing
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Get number of CPU cores
    num_workers =  NUM_WORKERS
    
    # ---------------------------------------------------
    # Load CIFAR-100 Dataset
    # ---------------------------------------------------
    # Load full training dataset
    full_trainset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)

    # Split into training and validation sets
    train_size = int(TRAIN_SPLIT * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # Apply Validation Transform to the Validation Set
    valset.dataset.transform = test_transform

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    testset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    # ---------------------------------------------------
    # Retrieve Class Labels
    # ---------------------------------------------------
    # CIFAR-100 contains 100 classes, which can be retrieved as a list of strings.
    classes = full_trainset.classes

    sample_images_from_loader(trainloader, output_dir="train_sample_images", num_samples=5, classes=classes)
    sample_images_from_loader(valloader, output_dir="val_sample_images", num_samples=5, classes=classes)

    # ---------------------------------------------------
    # Return the DataLoaders and Class Labels
    # ---------------------------------------------------
    return trainloader, valloader, testloader, classes



def sample_images_from_loader(loader, output_dir="sample_images", num_samples=5, classes=None):
    """
    Sample images from the DataLoader and save them to a specified directory.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader to sample images from.
        output_dir (str): Directory to save the sampled images.
        num_samples (int): Number of images to sample.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    data_iter = iter(loader)
    images, labels = next(data_iter)  # Get a batch of images and labels

    for i in range(num_samples):
        image = images[i]  # Get the i-th image
        label = labels[i].item()  # Get the corresponding label

        # Save the image with its class label in the filename
        save_image(image, os.path.join(output_dir, f"image_{i}_label_{classes[label]}.png"))

    print(f"[INFO] Saved {num_samples} images to '{output_dir}'.")