import math
import random

import torch
import torch.nn.functional as F


def _kl_loss_from_logits(student_logits, teacher_probs):
    log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_probs, teacher_probs, reduction="batchmean")


def UNDIAL(
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
    forget_weight=0.5,
    retain_weight=1.0,
    temperature=2.0,
):
    """Lightweight UNDIAL-style self-distillation pilot primitive.

    This is a FIT pilot-compatible implementation, not the full official UNDIAL
    code. It uses teacher-guided retain logits and adjusted high-entropy forget
    targets, which provides a distillation-style per-request update primitive
    for the rebuttal pilot.
    """
    random.shuffle(retain_set)
    retain_data = " ".join(retain_set)
    num_chunks_forget = math.ceil(len(forget_data) / chunk_size)

    for epoch in range(epochs):
        print(f"\n[UNDIAL pilot] Epoch {epoch + 1} ...")
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
            retain_inputs = tokenizer(
                retain_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True,
            ).to(device)

            forget_outputs = model(
                input_ids=forget_inputs["input_ids"],
                attention_mask=forget_inputs["attention_mask"],
            )
            retain_outputs = model(
                input_ids=retain_inputs["input_ids"],
                attention_mask=retain_inputs["attention_mask"],
            )

            with torch.no_grad():
                teacher_retain_outputs = infer_model(
                    input_ids=retain_inputs["input_ids"],
                    attention_mask=retain_inputs["attention_mask"],
                )

            # Retain: match teacher/reference softened logits.
            teacher_retain_probs = torch.softmax(
                teacher_retain_outputs.logits / temperature, dim=-1
            )
            retain_loss = _kl_loss_from_logits(
                retain_outputs.logits / temperature, teacher_retain_probs
            ) * (temperature ** 2)

            # Forget: distill toward a high-entropy adjusted target rather than
            # the original answer distribution.
            vocab_size = forget_outputs.logits.size(-1)
            uniform_probs = torch.full_like(
                forget_outputs.logits,
                fill_value=1.0 / vocab_size,
            )
            forget_loss = _kl_loss_from_logits(forget_outputs.logits, uniform_probs)

            loss = retain_weight * retain_loss + forget_weight * forget_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"[UNDIAL pilot] avg_loss={total_loss / max(1, num_chunks_forget):.4f}")
