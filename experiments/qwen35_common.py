"""Shared data ordering and model loading for the supplementary experiments."""
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PINNED_PCH_REVISIONS = {
    'FITPCH/Llama-3-8B_PCH_finetune': 'b77135a228f20d8cf94174a5f0afe445db4a8930',
    'FITPCH/Llama-3-8B_PCH_retain': '41c9e15c2f922cbcd78615c60f376ffad96716a9',
}


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pch_records(seed=42):
    texts = load_dataset("FITPCH/PCH")["train"]["text"]
    qa = load_dataset("FITPCH/PCH_QA")["train"]
    if not (len(texts) == len(qa) and len(texts) > 300):
        raise ValueError("PCH and QA must align and include a retain set after request 300")
    indices = list(range(300))
    random.Random(seed).shuffle(indices)
    indices.extend(range(300, len(texts)))
    return [dict(index=i, text=texts[i], question=qa[i]["question"], answer=qa[i]["answer"])
            for i in indices]


def manifest(records):
    raw = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    return {"dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "indices": [r["index"] for r in records]}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_model(name, device):
    revision=PINNED_PCH_REVISIONS.get(name)
    config = AutoConfig.from_pretrained(name,revision=revision)
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM
        # Official Transformers supports loading the text backbone from the
        # multimodal checkpoint, without allocating or training the vision tower.
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            name, config=config.text_config, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True, revision=revision,
        )
    else:
        model, loading = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            output_loading_info=True,revision=revision)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Checkpoint did not fully initialize the text model: {loading}")
    model.config.use_cache = False
    # All experiments explicitly use greedy decoding and max_new_tokens=128.
    # Clear incompatible sampling defaults inherited from old saved checkpoints.
    model.generation_config.max_length = 20
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    return model.to(device)


def load_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left",revision=PINNED_PCH_REVISIONS.get(name))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode(tokenizer, text, device, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)
    if inputs.input_ids.shape[1] < 2:
        return None
    return inputs


def decoder(model):
    return model.model


def attention_backend(model):
    result={'full_attention':'sdpa'}
    for layer in decoder(model).layers:
        if hasattr(layer,'linear_attn'):
            block=layer.linear_attn
            result.update(linear_chunk=block.chunk_gated_delta_rule.__module__,
                          linear_recurrent=block.recurrent_gated_delta_rule.__module__,
                          causal_convolution='causal_conv1d' if block.causal_conv1d_fn else 'torch',
                          gated_norm=type(block.norm).__module__)
            break
    return result
