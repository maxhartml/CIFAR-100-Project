import os
import torch
import time
from training.save_load import save_checkpoint
from metrics.compute_validation_loss import compute_validation_loss
from metrics.comptute_accuracy import compute_accuracy
from training.early_stopping import EarlyStopping
from configuration.config_1 import *


def train_model(model, trainloader, valloader, optimizer, criterion, scheduler, start_epoch, writer, save_dir):
    """
    Train the model on the given dataset for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        trainloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        valloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.
        criterion (torch.nn.Module): The loss function to minimize.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        start_epoch (int): The starting epoch number (useful for resuming training).
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
        save_dir (str): The directory where checkpoints and the final model will be saved.

    Returns:
        int: The epoch number after completing the training.
    """
    early_stopping = EarlyStopping()

    # Total number of batches in an epoch
    num_batches = len(trainloader)
    print_every = max(1, num_batches // 2)  # Print progress 2 times per epoch

    # Training loop
    step = 0  # Global step counter
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        print(f"[INFO] Starting Epoch {epoch + 1}/{start_epoch + NUM_EPOCHS}...")
        running_loss = 0.0
        model.train()  # Ensure the model is in training mode

        # Track start time for calculating images per second
        start_time = time.time()
        total_images = 0
        val_loss = None  # Variable to store validation loss

        # Iterate over batches
        for i, data in enumerate(trainloader):
            # Get inputs and labels, and move them to the device
            inputs, labels = data
            batch_size = inputs.size(0)  # Get the batch size
            total_images += batch_size  # Track the total images processed
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Compute predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP, norm_type=2.0)

            # Optimisation step
            optimizer.step()

            # Update the learning rate scheduler
            scheduler.step()  # CosineAnnealingLR needs to step after every batch
            step += 1  # Increment the global step

            # Accumulate loss for reporting
            running_loss += loss.item()

            # Print progress periodically
            if (i + 1) % print_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_train_loss = running_loss / print_every
                val_loss = compute_validation_loss(model, valloader, criterion)  # Compute validation loss once
                model.train()  # Reset to training mode after validation

                # Calculate elapsed time and images per second
                elapsed_time = time.time() - start_time
                images_per_second = total_images / elapsed_time

                print(f"[Epoch {epoch + 1}, Batch {i + 1}/{num_batches}] "
                      f"LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Images/sec: {images_per_second:.2f}")

                # Log metrics to TensorBoard
                writer.add_scalar("Loss/train", avg_train_loss, step)
                writer.add_scalar("Loss/validation", val_loss, step)
                writer.add_scalar("Learning Rate", current_lr, step)
                writer.add_scalar("Performance/images_per_sec", images_per_second, step)

                running_loss = 0.0

        # Compute train and validation accuracy
        train_accuracy = compute_accuracy(model, trainloader, num_classes=100)
        val_accuracy = compute_accuracy(model, valloader, num_classes=100)

        # Log accuracy metrics to console
        print(f"[INFO] Epoch {epoch + 1} - "
              f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

        # Log accuracy metrics to TensorBoard
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

        # Save a checkpoint
        if save_dir and (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint-epoch-{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)

        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            writer.add_text("Early Stopping", f"Triggered at epoch {epoch + 1}", epoch + 1)
            print(f"[INFO] Early stopping triggered at epoch {epoch + 1}")
            break

    print("[INFO] Training complete.")
    return start_epoch + NUM_EPOCHS