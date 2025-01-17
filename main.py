import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import tkinter as tk

from models.cifar100_net import CIFAR100Net
from datasets.cifar100_loader import get_cifar100_loaders
from training.train import train_model
from training.eval import evaluate_model
from training.save_load import load_latest_checkpoint, ensure_checkpoint_dir_exists
from gui.image_classifier_gui import ImageClassifierGUI


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def print_separator():
    """Prints a separator for better console readability."""
    print("=" * 50)

# ---------------------------------------------------
# Main Entry Point
# ---------------------------------------------------
if __name__ == "__main__":
    # Step 1: Program Start
    print_separator()
    print("🔥 CIFAR-100 Image Classification - Start 🔥")
    print_separator()

    # Step 2: Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Step 3: Load the CIFAR-100 dataset
    print_separator()
    print("[INFO] Preparing CIFAR-100 dataset...")
    trainloader, valloader, testloader, classes = get_cifar100_loaders(batch_size=64)
    print(f"[INFO] Dataset loaded successfully. Number of classes: {len(classes)}")

    # Step 4: Initialize the model
    print_separator()
    print("[INFO] Initializing model...")
    model = CIFAR100Net().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model initialized with {total_params:,} parameters.")

    # Step 5: Define loss function, optimizer, and learning rate scheduler
    print_separator()
    print("[INFO] Setting up training configuration...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    print("[INFO] Training configuration complete.")

    # Step 6: Checkpoint management
    checkpoint_dir = ensure_checkpoint_dir_exists()
    print_separator()
    print("[INFO] Checking for existing checkpoints...")
    epoch = load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir, device)

    # Step 7: Training
    print_separator()
    num_epochs = 1
    print(f"[INFO] Starting training from epoch {epoch + 1} for {num_epochs} epochs...")
    epoch = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        start_epoch=epoch,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        save_interval=10
    )

    # Step 9: Evaluate the model
    print_separator()
    print("[INFO] Evaluating model on test data...")
    evaluate_model(model, testloader, device, classes)
    print("[INFO] Evaluation complete.")

    # Step 10: Launch the GUI
    print_separator()
    print("[INFO] Launching the GUI...")
    root = tk.Tk()
    app = ImageClassifierGUI(root, model, device, classes)
    root.mainloop()

    # Step 11: Program End
    print("[INFO] GUI session ended. Program complete.")
    print_separator()