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
    
def NPO_KL(model, infer_model, tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform a mixed gradient ascent and gradient descent optimization process 
    for fine-tuning the model.

    Args:
        model: The model to be optimized
        infer_model: Reference model (oracle) used for comparison
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        retain_set: Dataset to retain (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs
        chunk_size: Batch size for training

    Returns:
        None
    """
    data_length_forget = len(forget_data)
    random.shuffle(retain_set)
    # Merge multiple strings in retain_set into one large string
    retain_data = " ".join(retain_set)
    data_length_retain = len(retain_data)

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

            # Tokenize forget data
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
            loss_forget_current = get_batch_loss(forget_outputs.logits, forget_labels) 

            # Tokenize retain data
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
            with torch.no_grad():
                infer_retain_outputs = infer_model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                    labels=retain_labels
                )
                infer_forget_outputs = infer_model(
                    input_ids=forget_input_ids,
                    attention_mask=forget_attention_mask,
                    labels=forget_labels
                )

            prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
            prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

            loss_forget_oracle = get_batch_loss(infer_forget_outputs.logits, forget_labels)
            neg_log_ratios = loss_forget_current - loss_forget_oracle
            loss_forget = -F.logsigmoid(0.1 * neg_log_ratios).mean() * 2 / 0.1 

            # Retain KL-divergence loss
            loss_retain = kl_loss(prob_retain_p, prob_retain_q)

            # Total loss combines forget and retain terms
            loss = loss_forget + loss_retain

            # Add to total loss
            total_loss += loss_retain.item()
            total_loss += loss_forget.item()

            # Backward pass (accumulate gradients)
            loss.backward()
            # Update parameters
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")
