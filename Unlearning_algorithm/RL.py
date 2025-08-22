import math
import torch
import torch.nn as nn

def Random_label(model, tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform a mixed gradient ascent and gradient descent optimization process 
    for fine-tuning the model.

    Args:
        model: The model to be optimized
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs
        chunk_size: Batch size for training

    Returns:
        None
    """
    data_length_forget = len(forget_data)
    loss_fn = nn.CrossEntropyLoss()
    # Calculate the number of batches
    num_chunks_forget = math.ceil(data_length_forget / chunk_size)

    for epoch in range(epochs):
        print(f"\n[FineTuning] Epoch {epoch + 1} ...")
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        total_loss = 0.0

        # Process forget_data (gradient ascent with randomized labels)
        for chunk_idx in range(num_chunks_forget):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            forget_chunk_text = forget_data[start_idx:end_idx]
            
            # Tokenize inputs
            infer_forget_inputs = tokenizer(
                forget_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            infer_forget_input_ids = infer_forget_inputs["input_ids"]
            infer_forget_attention_mask = infer_forget_inputs["attention_mask"]

            # Mask the padding tokens to be ignored in loss calculation
            infer_forget_labels = infer_forget_input_ids.masked_fill(infer_forget_attention_mask == 0, -100)

            # Get vocabulary size
            num_classes = tokenizer.vocab_size

            # Randomize labels: randomly generate labels with the same shape as original
            random_labels = torch.randint(0, num_classes, infer_forget_labels.shape).to(device)

            # Ensure padding tokens are ignored in loss computation
            random_labels = random_labels.masked_fill(infer_forget_attention_mask == 0, -100)

            # Forward pass
            forget_outputs = model(
                input_ids=infer_forget_input_ids,
                attention_mask=infer_forget_attention_mask,
                labels=random_labels
            )

            # Compute loss
            loss = forget_outputs.loss
            total_loss += loss.item()
            
            # Zero gradients before backward pass
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Parameter update
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")
