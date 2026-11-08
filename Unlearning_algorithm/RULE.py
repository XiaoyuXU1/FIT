import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sequence_nll(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)


def _kl_loss(prob_p, prob_q):
    return -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()


def RULE(
    model,
    infer_model,
    tokenizer,
    forget_data,
    retain_set,
    optimizer,
    scheduler,
    epochs,
    chunk_size,
    device,
    beta=0.1,
    retain_weight=1.0,
):
    """Lightweight RULE-style RL pilot primitive.

    This is a FIT pilot-compatible implementation, not the full official RULE
    code. It uses a policy-ratio style forget objective against the reference
    model and a retain-side KL term, so the primitive exposes the same
    per-request update interface as the existing FIT primitives.
    """
    random.shuffle(retain_set)
    retain_data = " ".join(retain_set)
    num_chunks_forget = math.ceil(len(forget_data) / chunk_size)

    for epoch in range(epochs):
        print(f"\n[RULE pilot] Epoch {epoch + 1} ...")
        model.train()
        total_loss = 0.0

        for chunk_idx in range(num_chunks_forget):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            forget_chunk_text = forget_data[start_idx:end_idx]
            retain_chunk_text = retain_data[start_idx:end_idx]

            forget_inputs = tokenizer(
                forget_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True,
            ).to(device)
            forget_labels = forget_inputs["input_ids"].masked_fill(
                forget_inputs["attention_mask"] == 0, -100
            )

            retain_inputs = tokenizer(
                retain_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True,
            ).to(device)
            retain_labels = retain_inputs["input_ids"].masked_fill(
                retain_inputs["attention_mask"] == 0, -100
            )

            forget_outputs = model(
                input_ids=forget_inputs["input_ids"],
                attention_mask=forget_inputs["attention_mask"],
                labels=forget_labels,
            )
            retain_outputs = model(
                input_ids=retain_inputs["input_ids"],
                attention_mask=retain_inputs["attention_mask"],
                labels=retain_labels,
            )
            with torch.no_grad():
                ref_forget_outputs = infer_model(
                    input_ids=forget_inputs["input_ids"],
                    attention_mask=forget_inputs["attention_mask"],
                    labels=forget_labels,
                )
                ref_retain_outputs = infer_model(
                    input_ids=retain_inputs["input_ids"],
                    attention_mask=retain_inputs["attention_mask"],
                    labels=retain_labels,
                )

            current_forget_nll = _sequence_nll(forget_outputs.logits, forget_labels)
            ref_forget_nll = _sequence_nll(ref_forget_outputs.logits, forget_labels)

            # A compact policy-ratio style objective: increase relative forget
            # loss while clipping extreme rewards for pilot stability.
            relative_forget = torch.clamp(current_forget_nll - ref_forget_nll, -20.0, 20.0)
            forget_loss = -F.logsigmoid(beta * relative_forget).mean() * (2.0 / beta)

            retain_prob = torch.softmax(retain_outputs.logits, dim=-1)
            ref_retain_prob = torch.softmax(ref_retain_outputs.logits, dim=-1)
            retain_loss = _kl_loss(retain_prob, ref_retain_prob)

            loss = forget_loss + retain_weight * retain_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"[RULE pilot] avg_loss={total_loss / max(1, num_chunks_forget):.4f}")
