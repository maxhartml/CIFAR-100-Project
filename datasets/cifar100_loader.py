from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from configuration.config_1 import *
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os

def get_cifar100_loaders():
    """
    Prepare DataLoaders for the CIFAR-100 dataset with transformations.

    Args:
        batch_size (int): Number of samples per batch for the DataLoader.

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for the CIFAR-100 training dataset.
        valloader (torch.utils.data.DataLoader): DataLoader for the CIFAR-100 validation dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the CIFAR-100 test dataset.
        classes (list): List of class names for the CIFAR-100 dataset.
    """
    # ---------------------------------------------------
    # Define Transformations
    # ---------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=AUGMENTATION_PADDING),  # Add random cropping
        transforms.RandomHorizontalFlip(),  # Add random horizontal flipping
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Simulate lighting changes
        transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
        transforms.Normalize(MEAN, STD),  # Normalize to range [-1, 1]
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # ---------------------------------------------------
    # Load CIFAR-100 Dataset
    # ---------------------------------------------------
    # Load full training dataset
    full_trainset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)

    # Load the full test dataset (will split into validation and test sets)
    full_testset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)

    # Split the test dataset into validation and test sets
    val_size = len(full_testset) // 2  # 5,000 images for validation
    test_size = len(full_testset) - val_size  # 5,000 images for test
    valset, testset = random_split(full_testset, [val_size, test_size])

    # Print dataset sizes
    print("[INFO] Dataset Sizes:")
    print(f"    Training Set: {len(full_trainset)} images")
    print(f"    Validation Set: {len(valset)} images")
    print(f"    Test Set: {len(testset)} images")

    # Create DataLoaders
    num_workers = NUM_WORKERS
    trainloader = DataLoader(full_trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    # ---------------------------------------------------
    # Retrieve Class Labels
    # ---------------------------------------------------
    classes = full_trainset.classes

    # Sample images for verification (optional)
    sample_images_from_loader(trainloader, output_dir="images/train_sample", num_samples=5, classes=classes)
    sample_images_from_loader(valloader, output_dir="images/val_sample", num_samples=5, classes=classes)
    sample_images_from_loader(testloader, output_dir="images/test_sample", num_samples=5, classes=classes)

    # ---------------------------------------------------
    # Return the DataLoaders and Class Labels
    # ---------------------------------------------------
    return trainloader, valloader, testloader, classes


def sample_images_from_loader(loader, output_dir, num_samples, classes):
    """
    Sample images from the DataLoader and save them to a specified directory.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader to sample images from.
        output_dir (str): Directory to save the sampled images.
        num_samples (int): Number of images to sample.
        classes (list): List of class names for CIFAR-100.
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