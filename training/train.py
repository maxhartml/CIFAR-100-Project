import torch

def train_model(model, trainloader, optimizer, criterion, scheduler, device, start_epoch, num_epochs):
    """
    Train the model on the given dataset for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        trainloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters.
        criterion (torch.nn.Module): The loss function to minimize.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to train the model on (CPU or GPU).
        start_epoch (int): The starting epoch number (useful for resuming training).
        num_epochs (int): The number of epochs to train for.

    Returns:
        None
    """
    # Total number of batches in an epoch
    num_batches = len(trainloader)
    print_every = max(1, num_batches // 10)  # Print progress 10 times per epoch

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"[INFO] Starting Epoch {epoch + 1}/{start_epoch + num_epochs}...")
        running_loss = 0.0
        model.train()  # Ensure the model is in training mode

        # Iterate over batches
        for i, data in enumerate(trainloader):
            # Get inputs and labels, and move them to the device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Compute predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()

            # Print progress periodically
            if (i + 1) % print_every == 0:
                current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate
                avg_loss = running_loss / print_every
                print(f"[Epoch {epoch + 1}, Batch {i + 1}/{num_batches}] "
                      f"LR: {current_lr:.6f}, Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Step the learning rate scheduler
        scheduler.step()

        # Report the completion of the epoch
        print(f"[INFO] Epoch {epoch + 1} completed.")

    print("[INFO] Training complete.")
    
    return start_epoch + num_epochs