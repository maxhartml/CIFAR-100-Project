import torchvision
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_cifar100_loaders(batch_size):
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
        transforms.RandomCrop(32, padding=4),  # Randomly crop with padding for better generalization
        transforms.RandomHorizontalFlip(),    # Random horizontal flipping for data augmentation
        transforms.ToTensor(),                # Convert PIL images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ---------------------------------------------------
    # Load CIFAR-100 Dataset
    # ---------------------------------------------------
    # Load full training dataset
    full_trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # Apply Validation Transform to the Validation Set
    valset.dataset.transform = test_transform

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---------------------------------------------------
    # Retrieve Class Labels
    # ---------------------------------------------------
    # CIFAR-100 contains 100 classes, which can be retrieved as a list of strings.
    classes = full_trainset.classes

    # ---------------------------------------------------
    # Return the DataLoaders and Class Labels
    # ---------------------------------------------------
    return trainloader, valloader, testloader, classes