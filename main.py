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
from configuration.config_1 import *
from torch.utils.tensorboard import SummaryWriter


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

    writer = SummaryWriter(log_dir=LOG_DIR)

    # Step 1: Program Start
    print_separator()
    print("🔥 CIFAR-100 Image Classification - Start 🔥")
    print_separator()

    # Step 2: Initialize the device
    print(f"[INFO] Using device: {DEVICE}")

    # Step 3: Load the CIFAR-100 dataset
    print_separator()
    print("[INFO] Preparing CIFAR-100 dataset...")
    trainloader, valloader, testloader, classes = get_cifar100_loaders()
    print(f"[INFO] Dataset loaded successfully. Number of classes: {len(classes)}")

    # Step 4: Initialize the model
    print_separator()
    print("[INFO] Initializing model...")
    model = CIFAR100Net().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model initialized with {total_params:,} parameters.")

    # Step 5: Define loss function, optimizer, and learning rate scheduler
    print_separator()
    print("[INFO] Setting up training configuration...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    print("[INFO] Training configuration complete.")

    # Step 6: Checkpoint management
    ensure_checkpoint_dir_exists()
    print_separator()
    print("[INFO] Checking for existing checkpoints...")
    epoch = load_latest_checkpoint(model, optimizer, scheduler)

    # Step 7: Training
    print_separator()
    print(f"[INFO] Starting training from epoch {epoch + 1} for {NUM_EPOCHS} epochs...")
    epoch = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        start_epoch=epoch,
        writer=writer
    )

    # Step 9: Evaluate the model
    print_separator()
    print("[INFO] Evaluating model on test data...")
    evaluate_model(model, testloader, classes, writer, epoch)
    print("[INFO] Evaluation complete.")

    if DEVICE == "cpu":
        # Step 10: Launch the GUI
        print_separator()
        print("[INFO] Launching the GUI...")
        root = tk.Tk()
        app = ImageClassifierGUI(root, model, classes)
        root.mainloop()

    # Step 11: Program End
    print("[INFO] GUI session ended. Program complete.")
    print_separator()