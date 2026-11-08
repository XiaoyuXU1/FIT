# -*- coding: utf-8 -*-
"""
PISCES (Precise In-Parameter Concept Erasure) — single-file, Llama/Yi compatible.

Key points:
- Apply across **all layers by default** (as in the paper): layer_ids = range(L)
- Edit **column-space** of FFN output matrix: down_proj.weight [hidden, inter] -> transpose to [inter, hidden]
- SAE acts on hidden-dim vectors (aligned with lm_head width)
- Fully dtype-safe: supports bf16/float16/float32 mixed environments

Entry:
    Pisces(model, tokenizer, forget_texts, epochs, chunk_size, device, ... ) -> model
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Set
import math
from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================= SAE container =======================
@dataclass
class SAEPair:
    # PyTorch Linear(weight): [out_features, in_features]
    encoder: nn.Linear  # Linear(d_hidden -> k), weight: [k, d_hidden]
    decoder: nn.Linear  # Linear(k -> d_hidden), weight: [d_hidden, k]


# ======================= helpers =======================
@contextmanager
def no_autocast():
    old = torch.is_autocast_enabled()
    try:
        if old:
            torch.set_autocast_enabled(False)
        yield
    finally:
        if old:
            torch.set_autocast_enabled(True)


def _chunk_text_ids(tokenizer, text: str, max_tokens: int, min_freq: int,
                    chunk_size_chars: int, epochs: int) -> List[int]:
    """Tokenize long forget_texts by char-chunks to build a concept token-id set."""
    ids: List[int] = []
    text = text if isinstance(text, str) else " ".join(text)
    stride = max(1, len(text) // max(1, math.ceil(len(text) / max(1, chunk_size_chars * 16))))
    for _ in range(max(1, epochs)):
        cur = 0
        while cur < len(text):
            seg = text[cur: cur + stride]; cur += stride
            toks = tokenizer.encode(seg, add_special_tokens=False)
            if toks:
                ids.extend(toks)
            if len(ids) >= max_tokens:
                break
        if len(ids) >= max_tokens:
            break
    cnt = Counter(ids)
    kept = [tid for tid, f in cnt.most_common(max_tokens) if f >= min_freq]
    return kept


def sae_encode(sae: SAEPair, V: torch.Tensor) -> torch.Tensor:
    """
    Encode hidden-dim parameter rows into SAE features.
    encoder.weight: [k, d_hidden] => V[* , d_hidden] @ encoder.weight.T -> [* , k]
    Cast to V.dtype to avoid dtype mismatch (bf16/float32).
    """
    enc_w_T = sae.encoder.weight.t().to(device=V.device, dtype=V.dtype)  # [d_hidden, k]
    return torch.relu(V @ enc_w_T)


def sae_decode(sae: SAEPair, M: torch.Tensor) -> torch.Tensor:
    """
    Decode SAE features back to hidden-dim vectors.
    decoder.weight: [d_hidden, k] => M[* , k] @ decoder.weight.T -> [* , d_hidden]
    """
    dec_w_T = sae.decoder.weight.t().to(device=M.device, dtype=M.dtype)  # [k, d_hidden]
    return M @ dec_w_T


@torch.no_grad()
def vocab_projection(
    E: torch.Tensor,        # [V, d_hidden]  (lm_head weight)
    W_dec_T: torch.Tensor,  # [k, d_hidden]  (decoder.weight.T)
    use_cosine: bool = False,
    topk: int = 50,
    bottomk: int = 50,
    batch: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature f (row in W_dec_T), rank tokens via E @ w_f (or cosine).
    DTYPE-SAFE:
      - raw path uses E.dtype
      - cosine path computes in float32 for stability
    Returns top_idx[k, topk], bot_idx[k, bottomk].
    """
    W_dec_T = W_dec_T.to(device=E.device)

    if not use_cosine:
        W_dec_T = W_dec_T.to(dtype=E.dtype)
        E_used = E
        tops, bots = [], []
        for s in range(0, W_dec_T.size(0), batch):
            wf = W_dec_T[s:s+batch]                         # [b, d_hidden]
            logits = E_used @ wf.t()                        # [V, b]
            top_idx = torch.topk(logits, topk, dim=0).indices.T
            bot_idx = torch.topk(-logits, bottomk, dim=0).indices.T
            tops.append(top_idx); bots.append(bot_idx)
        return torch.cat(tops, 0), torch.cat(bots, 0)

    # cosine path in float32
    E_f32 = E.to(dtype=torch.float32)
    W_f32 = W_dec_T.to(dtype=torch.float32)
    E_norm = F.normalize(E_f32, dim=-1)
    tops, bots = [], []
    for s in range(0, W_f32.size(0), batch):
        wf = W_f32[s:s+batch]
        logits = E_norm @ F.normalize(wf, dim=-1).T         # [V, b]
        top_idx = torch.topk(logits, topk, dim=0).indices.T
        bot_idx = torch.topk(-logits, bottomk, dim=0).indices.T
        tops.append(top_idx); bots.append(bot_idx)
    del E_f32, W_f32, E_norm
    return torch.cat(tops, 0), torch.cat(bots, 0)


# ======================= Llama / Yi adapters =======================
_LLAMA_YI_WOUT_PATHS = [
    "model.layers.{i}.mlp.down_proj.weight",  # common: [hidden, inter]
    "model.layers.{i}.mlp.o_proj.weight",     # rare alt
]

def _has_attr(obj, path: str) -> bool:
    cur = obj
    for p in path.split('.'):
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    return True

def _get_attr(obj, path: str):
    cur = obj
    for p in path.split('.'):
        cur = getattr(cur, p)
    return cur

def _set_tensor_(param: torch.Tensor, value: torch.Tensor) -> None:
    with torch.no_grad():
        param.data.copy_(value.to(param.data.dtype).to(param.data.device))

def llama_yi_get_W_out(model, layer_id: int) -> torch.Tensor:
    for pat in _LLAMA_YI_WOUT_PATHS:
        path = pat.format(i=layer_id)
        if _has_attr(model, path):
            return _get_attr(model, path).detach().clone()
    raise ValueError(f"[get_W_out] Cannot locate FFN output weight for layer {layer_id}.")

def llama_yi_set_W_out_(model, layer_id: int, W: torch.Tensor) -> None:
    for pat in _LLAMA_YI_WOUT_PATHS:
        path = pat.format(i=layer_id)
        if _has_attr(model, path):
            _set_tensor_(_get_attr(model, path), W)
            return
    raise ValueError(f"[set_W_out] Cannot write FFN output weight for layer {layer_id}.")


# ======================= Main: Pisces =======================
@torch.no_grad()
def Pisces(
    model,
    tokenizer,
    forget_texts: str,          # target concept raw text (str)
    epochs: int,
    chunk_size: int,            # used for char-chunking & batch sizing
    device: str | torch.device,
    layer_ids: Optional[Iterable[int]] = None,  # default: all layers
    # ---- optional / advanced ----
    tau: float = 0.4,           # row selection threshold
    mu: float = 1.2,            # erasure magnitude
    topk: int = 50,
    bottomk: int = 50,
    density_threshold: float = 0.15,     # feature selection: concept density
    exclude_noise_ratio: float = 0.95,   # feature selection: noise tolerance
    max_forget_tokens: int = 4096,
    min_token_freq: int = 1,
    use_cosine_vproj: bool = False,      # cosine for robustness; raw for sharper selection
    edit_in_place: bool = True,          # in-place writeback
) -> nn.Module:
    """
    PISCES-style concept erasure (column-space editing).
      - By default, applies across **all transformer layers**.
      - Selective per-vector editing (avoid editing all vectors to limit recon error).
    Returns the edited `model`.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device).eval()

    # 0) Unembedding (E) and hidden size
    if not hasattr(model, "lm_head") or not hasattr(model.lm_head, "weight"):
        raise ValueError("model.lm_head.weight not found; cannot perform VocabProj.")
    E = model.lm_head.weight.detach().to(device)     # [V, d_hidden] (bf16/float16/float32)
    d_hidden = E.size(1)

    # 1) Target layers (default: all)
    try:
        L = len(model.model.layers)
    except Exception as e:
        raise ValueError("Cannot infer number of transformer layers.") from e
    layer_ids = list(layer_ids) if layer_ids is not None else list(range(L))

    # 2) Probe a down_proj to get (hidden, inter)
    sample_w = llama_yi_get_W_out(model, layer_ids[0]).to(device)  # [hidden, inter]
    hidden_from_down, inter_from_down = sample_w.shape
    if hidden_from_down != d_hidden:
        raise ValueError(
            f"Hidden size mismatch: lm_head width={d_hidden}, but down_proj OUT={hidden_from_down}."
        )
    print(f"[Pisces] E={tuple(E.shape)} dtype={E.dtype}  "
          f"down_proj={tuple(sample_w.shape)} dtype={sample_w.dtype}  "
          f"W_cols={tuple(sample_w.t().shape)}")

    # 3) Create SAE pairs on hidden dim (use lm_head dtype)
    k = max(256, min(1024, d_hidden // 4))
    sae_pairs: Dict[int, SAEPair] = {
        lid: SAEPair(
            encoder=nn.Linear(d_hidden, k, bias=False).to(device).to(dtype=E.dtype).eval(),
            decoder=nn.Linear(k, d_hidden, bias=False).to(device).to(dtype=E.dtype).eval(),
        )
        for lid in layer_ids
    }

    # 4) Build concept tokens
    concept_token_ids = _chunk_text_ids(
        tokenizer, forget_texts,
        max_tokens=max_forget_tokens, min_freq=min_token_freq,
        chunk_size_chars=chunk_size, epochs=epochs
    )
    vocab = set(concept_token_ids)
    print(vocab)
    # 5) Per-layer: identify Fc -> select rows (column-space) -> erase -> decode -> write back
    with no_autocast():
        for lid in layer_ids:
            sae = sae_pairs[lid]

            # ---- Get W_out and move to column-space ----
            W_out = llama_yi_get_W_out(model, lid).to(device)  # [hidden, inter]
            W_cols = W_out.t().contiguous()                    # [inter, hidden]
            n_rows, d_h = W_cols.shape
            if d_h != d_hidden:
                raise ValueError(f"[Layer {lid}] W_cols hidden width={d_h} != {d_hidden}")

            # ---- VocabProj (decoder.T gives [k, hidden]); dtype handled inside
            W_dec_T = sae.decoder.weight.t().detach().to(device)  # [k, hidden]
            top_idx, bot_idx = vocab_projection(
                E=E, W_dec_T=W_dec_T,
                use_cosine=use_cosine_vproj,
                topk=topk, bottomk=bottomk,
                batch=max(512, chunk_size)
            )

            # ---- Identify Fc (concept-related SAE features)
            Fc: Set[int] = set()
            for f in range(top_idx.size(0)):
                tops, bots = set(top_idx[f].tolist()), set(bot_idx[f].tolist())
                top_hit = len([t for t in tops if t in vocab]) / max(1, len(tops))
                bot_hit = len([t for t in bots if t in vocab]) / max(1, len(bots))
                if (top_hit >= density_threshold and (1 - top_hit) <= exclude_noise_ratio) or \
                   (bot_hit >= density_threshold and (1 - bot_hit) <= exclude_noise_ratio):
                    Fc.add(f)
            if not Fc:
                # 无相关特征，跳过该层
                print(f"[Layer {lid}] no concept features selected; skip.")
                continue
            Fc = sorted(Fc)

            # ---- SAE-encode all rows in column-space ----
            row_bs = max(256, chunk_size)
            M_chunks = []
            for s in range(0, n_rows, row_bs):
                v = W_cols[s:s+row_bs]                           # [b, d_hidden] (dtype like model weights)
                m = sae_encode(sae, v)                           # [b, k]
                M_chunks.append(m)
            M = torch.cat(M_chunks, 0)                           # [inter, k]

            # ---- m_hat and row selection (use M.dtype throughout)
            kdim = M.size(1)
            mask_fc = torch.zeros(kdim, dtype=torch.bool, device=M.device); mask_fc[Fc] = True
            m_hat = torch.zeros(kdim, device=M.device, dtype=M.dtype)
            m_hat[mask_fc] = M[:, mask_fc].max(dim=0).values
            if (m_hat[mask_fc] <= torch.zeros(1, device=M.device, dtype=M.dtype)).all():
                print(f"[Layer {lid}] all m_hat <= 0; skip.")
                continue

            thresh = (tau * m_hat).to(dtype=M.dtype).unsqueeze(0) # [1, k]
            comp = torch.zeros_like(M, dtype=torch.bool)
            comp[:, mask_fc] = M[:, mask_fc] >= thresh[:, mask_fc]
            sel = comp.any(dim=1)                                 # [inter]
            idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                print(f"[Layer {lid}] no rows selected; skip.")
                continue

            # ---- erase in feature space & decode back (still column-space)
            M_sel = M[idx].clone()
            M_sel[:, mask_fc] = (-mu * m_hat[mask_fc]).to(M.dtype).unsqueeze(0).expand(M_sel.size(0), -1)
            V_bar = sae_decode(sae, M_sel)                        # [n, d_hidden], dtype=M.dtype
            if V_bar.size(1) != d_hidden:
                raise RuntimeError(f"[Layer {lid}] decoded width {V_bar.size(1)} != hidden {d_hidden}")

            W_cols_new = W_cols.clone()
            W_cols_new[idx] = V_bar

            # ---- transpose back and write into model
            W_new = W_cols_new.t().contiguous()                   # [hidden, inter]
            llama_yi_set_W_out_(model, lid, W_new)

            print(f"[Layer {lid}] edited rows={int(idx.numel())}, |Fc|={len(Fc)}, dtype={W_new.dtype}")

    return model
