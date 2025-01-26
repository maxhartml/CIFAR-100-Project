import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import tkinter as tk
from datasets.cifar100_loader import get_cifar100_loaders
from models.initialise_model import initialize_model
from training.train import train_model
from metrics.comptute_accuracy import compute_accuracy
from training.save_load import load_latest_checkpoint, ensure_checkpoint_dir_exists, create_training_directory
from gui.image_classifier_gui import ImageClassifierGUI
from configuration.config_1 import *
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------
# Main Entry Point
# ---------------------------------------------------
if __name__ == "__main__":

    # Configuration setup
    training_config = {
        "MODEL_NAME": MODEL_NAME,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "INITIAL_LEARNING_RATE": INITIAL_LEARNING_RATE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "DROPOUT_RATE": DROPOUT_RATE,
        "GRAD_CLIP": GRAD_CLIP,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "PATIENCE": PATIENCE,
    }

    # Create a dedicated subdirectory for this training session
    ensure_checkpoint_dir_exists(dir_path=CHECKPOINT_DIR)
    training_dir = create_training_directory(CHECKPOINT_DIR, MODEL_NAME, training_config)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, os.path.basename(training_dir)))

     # Step 1: Program Start
    print("=" * 50)
    print("🔥 CIFAR-100 Image Classification - Start 🔥")
    print("=" * 50)

    # Step 2: Initialize the device
    print(f"[INFO] Using device: {DEVICE}")

    # step 3: Load the CIFAR-100 dataset
    print("[INFO] Preparing CIFAR-100 dataset...")
    trainloader, valloader, testloader, classes = get_cifar100_loaders()
    print(f"[INFO] Dataset loaded successfully. Number of classes: {len(classes)}")

    # step 4: Initialize the model
    print("[INFO] Initializing model...")
    model = initialize_model(MODEL_NAME, DEVICE)

    # Step 5: Define loss function, optimizer, and learning rate scheduler
    print("[INFO] Setting up training configuration...")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(trainloader)  # Total number of steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=ETA_MIN)
    print("[INFO] CosineAnnealingLR scheduler initialized.")
    print("[INFO] Training configuration complete.")

    # Step 6: Checkpoint management
    print("[INFO] Checking for existing checkpoints...")
    epoch = load_latest_checkpoint(model, optimizer, scheduler, training_dir)

    # Step 7: Training the model
    print(f"[INFO] Starting training from epoch {epoch + 1} for {NUM_EPOCHS} epochs...")
    epoch = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        start_epoch=epoch,
        writer=writer,
        save_dir=training_dir,
    )

    # Step 8: Evaluate the model on the test data
    print("[INFO] Evaluating model on test data...")
    test_accuracy = compute_accuracy(model, testloader)
    print("[INFO] Test Accuracy: {:.2f}%".format(test_accuracy))
    print("[INFO] Evaluation complete.")

     # Step 9: Save the final model
    final_model_path = os.path.join(training_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Final model saved at: {final_model_path}")
    writer.close()
    print("[INFO] TensorBoard writer closed.")
    
    # Step 10: Launch the GUI (optional based on configuration)
    if USE_GUI:
        print("[INFO] Launching the GUI...")
        root = tk.Tk()
        app = ImageClassifierGUI(root, model, classes)
        root.mainloop()

    # Step 10: Program End
    print("=" * 50)
    print("[INFO] Program complete.")
    print("=" * 50)