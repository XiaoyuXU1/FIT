import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum of losses for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss

def kl_loss(prob_p, prob_q):
    return -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
    
def NPO(model, infer_model, tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform the mixed gradient ascent and gradient descent optimization process 
    for fine-tuning the model.

    Args:
        model: The model to be optimized
        infer_model: A reference (oracle) model used for comparison
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
    # Calculate the number of batches
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

            # Tokenize inputs
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

            # Forward pass
            forget_outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                labels=forget_labels
            )
            loss_forget_current = get_batch_loss(forget_outputs.logits, forget_labels) 

            # Oracle model forward pass (no gradient)
            with torch.no_grad():
                infer_forget_outputs = infer_model(
                    input_ids=forget_input_ids,
                    attention_mask=forget_attention_mask,
                    labels=forget_labels
                )

            loss_forget_oracle = get_batch_loss(infer_forget_outputs.logits, forget_labels)
            neg_log_ratios = loss_forget_current - loss_forget_oracle

            # Negative preference optimization loss
            loss_forget = -F.logsigmoid(0.1 * neg_log_ratios).mean() * 2 / 0.1 

            loss = loss_forget

            total_loss += loss_forget.item()
            # Backward pass (we accumulate the gradients)
            loss.backward()
            # Update parameters
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")
