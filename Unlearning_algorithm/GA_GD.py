import math
import torch
import random

def Gradient_ascent_descent(model, tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform a mixed gradient ascent (forget set) and gradient descent (retain set) 
    optimization process for fine-tuning the model.

    Args:
        model: The model to be optimized
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        retain_set: Dataset to retain (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs
        chunk_size: Batch size for training
        gamma: (not implemented here) typically a weighting factor for the retain loss, default is 1.0

    Returns:
        None
    """
    data_length_forget = len(forget_data)
    random.shuffle(retain_set)
    # Merge multiple strings from retain_set into one large retain_data
    retain_data = " ".join(retain_set)
    data_length_retain = len(retain_data)

    # Calculate number of batches for forget data
    num_chunks_forget = math.ceil(data_length_forget / chunk_size)

    for epoch in range(epochs):
        print(f"\n[FineTuning] Epoch {epoch + 1} ...")
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        total_loss = 0.0

        # Process forget_data (gradient ascent)
        for chunk_idx in range(num_chunks_forget):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            forget_chunk_text = forget_data[start_idx:end_idx]

            # Tokenize forget batch
            forget_inputs = tokenizer(
                forget_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            forget_input_ids = forget_inputs["input_ids"]
            forget_attention_mask = forget_inputs["attention_mask"]
            forget_labels = forget_input_ids.masked_fill(forget_attention_mask == 0, -100)

            # Forward pass on forget data
            forget_outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                labels=forget_labels
            )
            loss_forget = forget_outputs.loss
            loss_forget = -loss_forget  # Negate loss for gradient ascent (forgetting)
            
            # Tokenize retain batch
            retain_chunk_text = retain_data[start_idx:end_idx]
            retain_inputs = tokenizer(
                retain_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            retain_input_ids = retain_inputs["input_ids"]
            retain_attention_mask = retain_inputs["attention_mask"]
            retain_labels = retain_input_ids.masked_fill(retain_attention_mask == 0, -100)

            # Forward pass on retain data
            retain_outputs = model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
                labels=retain_labels
            )
            loss_retain = retain_outputs.loss

            # Combined loss: ascent on forget + descent on retain
            loss = loss_forget + loss_retain  

            # Track both losses
            total_loss += loss_retain.item()
            total_loss += loss_forget.item()

            # Backpropagation and update
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")
