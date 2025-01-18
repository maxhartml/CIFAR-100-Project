import torchvision
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from configuration.config_1 import *

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
        transforms.RandomRotation(15),  # Add random rotation
        transforms.RandomErasing(),     # Add random erasing
        transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
        transforms.Normalize(MEAN, STD) # Normalize to range [-1, 1]
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
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    testset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # ---------------------------------------------------
    # Retrieve Class Labels
    # ---------------------------------------------------
    # CIFAR-100 contains 100 classes, which can be retrieved as a list of strings.
    classes = full_trainset.classes

    # ---------------------------------------------------
    # Return the DataLoaders and Class Labels
    # ---------------------------------------------------
    return trainloader, valloader, testloader, classes