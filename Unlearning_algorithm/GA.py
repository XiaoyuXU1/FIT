import math

def Gradient_ascent(model, tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform gradient ascent optimization for fine-tuning the model.

    Args:
        model: The model to be optimized
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs (default: 3)
        chunk_size: Batch size (default: 16)

    Returns:
        None
    """
    data_length = len(forget_data)
    num_chunks = math.ceil(data_length / chunk_size)
    
    for epoch in range(epochs):
        print(f"\n[FineTuning] Epoch {epoch + 1} ...")
        optimizer.zero_grad()
        total_loss = 0.0
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            chunk_text = forget_data[start_idx:end_idx]

            # Tokenize inputs
            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            weighted_loss = -loss  # Apply negative loss for gradient ascent

            # Accumulate total loss and backpropagate
            total_loss += weighted_loss.item()
            weighted_loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / num_chunks
        print(f"[Epoch {epoch + 1}] weighted_loss={avg_loss:.4f}")
