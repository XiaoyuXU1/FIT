"""Legacy ALKN variant, missing task-vector negation.

For the PCH paper reimplementation use experiments/alkn.py.
"""
import torch
import math
import tqdm
import numpy as np

def tokenize_texts(tokenizer, texts, max_length=512):
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
def ALKN(
    model, tokenizer,
    forget_texts, retain_texts,
    optimizer, scheduler,
    epochs, chunk_size, device,
    forget_loss="CTV",
    unmask_ratio_start=1.0, unmask_ratio_end=0.5,
    mask_p_k=1.0, init_prob_k=20.0,
    threshold_update_step=10, low_lr_ratio=0.1,
    prev_m_vector=None, prev_mask_thr=None
):
    """
    ALKN with paper-level ESL, DGS, and APM.
    Returns updated m_vector and mask_threshold for continual unlearning.
    """
    model.train()

    # 1. Retain gradient (chunked)
    retain_data = " ".join(retain_texts)
    total_len_r = len(retain_data)
    num_chunks_r = math.ceil(total_len_r / chunk_size)
    retain_grad = {}
    m_vector = prev_m_vector if prev_m_vector is not None else {}

    for idx in tqdm.tqdm(range(num_chunks_r), desc="Retain Gradient Extraction"):
        text = retain_data[idx*chunk_size:(idx+1)*chunk_size]
        enc = tokenizer(text, truncation=True, max_length=chunk_size,
                        padding='max_length', return_tensors='pt').to(device)
        outputs = model(**enc, labels=enc.input_ids)
        outputs.loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None: continue
            g = p.grad.detach().clone()
            retain_grad[name] = retain_grad.get(name, torch.zeros_like(g)) + g
        model.zero_grad()

    # average and init m_vector
    for name, g in retain_grad.items():
        retain_grad[name] = g / num_chunks_r
        if name not in m_vector:
            m_vector[name] = torch.zeros_like(retain_grad[name])

    # 2. Compute forget initial probs (CPU)
    forget_data = " ".join(forget_texts)
    total_len_f = len(forget_data)
    num_chunks_f = math.ceil(total_len_f / chunk_size)
    forget_init = []
    model.eval()
    with torch.no_grad():
        for idx in tqdm.tqdm(range(num_chunks_f), desc="Compute Forget Init Probs"):
            text = forget_data[idx*chunk_size:(idx+1)*chunk_size]
            enc = tokenizer(text, truncation=True, max_length=chunk_size,
                            padding='max_length', return_tensors='pt').to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            labels = enc.input_ids.clone().masked_fill(enc.input_ids==tokenizer.pad_token_id, -100)
            sl = labels[...,1:].contiguous()
            safe = sl.masked_fill(sl==-100,0).unsqueeze(-1)
            sel = torch.gather(probs[...,:-1,:], -1, safe).squeeze(-1)
            sel[sl==-100] = 0
            forget_init.append(sel.cpu())
    forget_init = torch.cat(forget_init,0)
    model.train()
    torch.cuda.empty_cache()

    # 3. Main training with ESL + DGS + APM
    mask_thr = prev_mask_thr if prev_mask_thr is not None else {}
    mask_prev = {}  # init previous mask for APM
    max_steps = epochs * num_chunks_f
    global_step = 0

    for epoch in range(epochs):
        for idx in tqdm.tqdm(range(num_chunks_f), desc=f"Training Epoch {epoch+1}", leave=False):
            text = forget_data[idx*chunk_size:(idx+1)*chunk_size]
            enc = tokenizer(text, truncation=True, max_length=chunk_size,
                            padding='max_length', return_tensors='pt').to(device)
            logits = model(**enc).logits
            pred = logits[..., :-1, :].log_softmax(dim=-1)

            # ESL soft labels using forget_init
            pi = forget_init[idx].to(device)
            y_tilde = torch.sigmoid(init_prob_k * (pi - 1.0))
            y_tilde = y_tilde.clamp(max=1.0)
            y_tilde = y_tilde.unsqueeze(0).unsqueeze(-1)

            labels = enc.input_ids.clone().masked_fill(enc.input_ids==tokenizer.pad_token_id, -100)
            sl = labels[...,1:].contiguous()
            safe = sl.masked_fill(sl==-100,0).unsqueeze(-1)
            true_dist = torch.zeros_like(pred).scatter_(2, safe, y_tilde)

            # CTV loss
            loss_mat = -true_dist * pred
            mask = (sl != -100).float()
            loss = (loss_mat.sum(-1) * mask).sum() / mask.sum()
            loss.backward()

            if forget_loss=="CTV":
                # dynamic gradient sparsity
                for name, p in model.named_parameters():
                    if p.grad is None: continue
                    mul = p.grad * retain_grad[name]
                    m_vector[name] += mul - mul.std() * mask_p_k

                # threshold update
                if (global_step+1)%threshold_update_step==0:
                    for name, vec in m_vector.items():
                        flat = vec.flatten()
                        k = max(5, int(flat.numel()*0.15))
                        topk = flat.topk(k).values
                        maxv, minv = topk[-1], flat.topk(k, largest=False).values[-1]
                        ratio = unmask_ratio_start - (unmask_ratio_start-unmask_ratio_end)*(global_step/max_steps)
                        adj = (ratio*flat.numel()-k+1)/(flat.numel()-2*k+2)
                        mask_thr[name] = maxv - (maxv-minv)*adj
                    # mask previous for APM
                    mask_prev = {n:(vec>mask_thr[n]) for n,vec in m_vector.items()}

                # apply mask & APM via gradient scaling
                for name, p in model.named_parameters():
                    if p.grad is None: continue
                    p.grad *= (m_vector[name]>mask_thr.get(name,-1e9)).float().to(p.grad.device)
                    prev_mask_tensor = mask_prev.get(name)
                    if isinstance(prev_mask_tensor, torch.Tensor) and prev_mask_tensor.any():
                        p.grad *= low_lr_ratio

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step +=1

    return m_vector, mask_thr
