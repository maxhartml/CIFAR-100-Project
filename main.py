import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import tkinter as tk
from datasets.cifar100_loader import get_cifar100_loaders
from models.initialise_model import initialize_model
from training.train import train_model
from metrics.comptute_accuracy import compute_accuracy
from training.save_load import load_latest_checkpoint, ensure_checkpoint_dir_exists
from gui.image_classifier_gui import ImageClassifierGUI
from configuration.config_1 import *
from torch.utils.tensorboard import SummaryWriter
import time

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

    run_name = f"{MODEL_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    # Step 1: Program Start
    print_separator()
    print("🔥 CIFAR-100 Image Classification - Start 🔥")
    print_separator()

    # Step 2: Initialize the device
    print(f"[INFO] Using device: {DEVICE}")

    # step 3: Load the CIFAR-100 dataset
    print_separator()
    print("[INFO] Preparing CIFAR-100 dataset...")
    trainloader, valloader, testloader, classes = get_cifar100_loaders()
    print(f"[INFO] Dataset loaded successfully. Number of classes: {len(classes)}")

    # step 4: Initialize the model
    print_separator()
    print("[INFO] Initializing model...")
    model = initialize_model(MODEL_NAME, DEVICE)

    # Step 5: Define loss function, optimizer, and learning rate scheduler
    print_separator()
    print("[INFO] Setting up training configuration...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    print("[INFO] Training configuration complete.")

    # Step 6: Checkpoint management
    ensure_checkpoint_dir_exists(dir_path=CHECKPOINT_DIR)
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

    # Step 8: Evaluate the model
    print_separator()
    print("[INFO] Evaluating model on test data...")
    test_accuracy = compute_accuracy(model, testloader)
    print("[INFO] Test Accuracy: {:.2f}%".format(test_accuracy))
    print("[INFO] Evaluation complete.")

     # Step 9: Save the final model
    print_separator()
    print("[INFO] Saving the final model...")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{run_name}.pth"))
    print("[INFO] Model saved successfully.")
    writer.close()
    print("[INFO] Tensorboard writer closed.")
    
    # Step 10: Launch the GUI (optional based on configuration)
    if USE_GUI:
        print_separator()
        print("[INFO] Launching the GUI...")
        root = tk.Tk()
        app = ImageClassifierGUI(root, model, classes)
        root.mainloop()

    # Step 11: Program End
    print_separator()
    print("[INFO] Program complete.")
    print_separator()