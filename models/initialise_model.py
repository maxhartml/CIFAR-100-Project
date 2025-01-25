def initialize_model(model_name, device, num_classes=100):
    """
    Dynamically initialize a model based on the provided model name.

    Args:
        model_name (str): Name of the model to initialize ("CNN", "CustomResNet18", "PretrainedResNet18").
        device (torch.device): Device to move the model to (e.g., "cpu" or "cuda").
        num_classes (int): Number of output classes (default: 100 for CIFAR-100).

    Returns:
        torch.nn.Module: Initialized model.
    """
    if model_name == "CNN":
        from models.custom_CNN import CNN
        model = CNN().to(device)
    elif model_name == "CustomResNet18":
        from models.custom_ResNet import CustomResNet18
        model = CustomResNet18().to(device)
    elif model_name == "PretrainedResNet18":
        from torchvision.models import resnet18, ResNet18_Weights
        import torch.nn as nn
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Log total parameters and model name
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Using Model: {model.__class__.__name__}")
    print(f"[INFO] Model initialized with {total_params:,} parameters.")

    return model