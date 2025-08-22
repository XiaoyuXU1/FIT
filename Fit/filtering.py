import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import hashlib
import math
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import lm_eval
import random
import matplotlib.pyplot as plt
import json
from collections import Counter
import os
from typing import List, Dict, Any, Optional, Callable, Union
import time
from cycler import cycler
TensorLike = Union[torch.Tensor, List[torch.Tensor]]


# ======= 1. Unlearn_request_memory =======
class Unlearn_request_memory:
    """
    Stores unlearning requests; ensures that each record returned by 
    retrieve_unlearning_data() has r["data"] as torch.Tensor([D]), 
    making it convenient for torch.stack later.
    """
    def __init__(self, embed_fn: Optional[Callable[[str], torch.Tensor]] = None):
        """
        Args:
            embed_fn: Optional, a function mapping text -> vector. 
                      If provided, text is converted to embeddings when stored/retrieved.
        """
        self.unlearning_memory: List[Dict[str, Any]] = []
        self.embed_fn = embed_fn

    @staticmethod
    def _to_vec1d(t: torch.Tensor) -> List[torch.Tensor]:
        """
        Normalize any tensor shape into a list of [D]. 
        Returns list[[D]] (input may be [B,D] or higher dimensional).
        """
        if not isinstance(t, torch.Tensor):
            return []
        # Move to CPU/float32 to avoid device/precision conflicts
        t = t.detach().to("cpu", dtype=torch.float32)
        if t.ndim == 1:
            return [t]
        if t.ndim == 2:
            if t.size(0) == 1:
                return [t.squeeze(0)]
            # [B, D] -> split into B vectors [D]
            return [row for row in t]
        # Higher dimensions: flatten into [N, D]
        t = t.view(-1, t.size(-1))
        return [row for row in t]

    def _embed_text(self, text: str) -> List[torch.Tensor]:
        if self.embed_fn is None:
            return []
        emb = self.embed_fn(text)
        return self._to_vec1d(emb)

    def store_unlearning_data(self, user_id: Any, data_to_forget: Any) -> None:
        """
        Accepts Tensor / list[Tensor] / str / list[str]:
        - list: store each item recursively
        - Tensor: normalize into [D] and store each vector as a separate record
        - str: if embed_fn is provided, convert to embedding immediately; 
               otherwise store text directly (to be converted later if possible)
        """
        # list/tuple: expand and store
        if isinstance(data_to_forget, (list, tuple)):
            for item in data_to_forget:
                self.store_unlearning_data(user_id, item)
            return

        records: List[torch.Tensor] = []
        if isinstance(data_to_forget, torch.Tensor):
            records = self._to_vec1d(data_to_forget)
        elif isinstance(data_to_forget, str):
            # Convert to embedding if embed_fn exists; otherwise store raw text
            if self.embed_fn is not None:
                records = self._embed_text(data_to_forget)
            else:
                self.unlearning_memory.append({
                    'user_id': user_id,
                    'data': data_to_forget,   # temporarily store text
                    'is_text': True
                })
                print(f"[Unlearn_request_memory] Stored TEXT for user={user_id}")
                return
        else:
            print(f"[Unlearn_request_memory] Unsupported type: {type(data_to_forget)}; skip.")
            return

        # Store each obtained vector as a separate record
        for vec in records:
            self.unlearning_memory.append({
                'user_id': user_id,
                'data': vec,        # always store Tensor([D]) in the 'data' field
                'is_text': False
            })

    def retrieve_unlearning_data(self) -> List[Dict[str, Any]]:
        """
        Each record returned satisfies: record['data'] is torch.Tensor([D]) (CPU/float32).
        - Normalize and expand any historical mis-stored list/tensor/high-dim inputs
        - For text records: if embed_fn exists, convert on the fly; otherwise skip 
          (to avoid torch.stack errors later)
        """
        out: List[Dict[str, Any]] = []
        for rec in self.unlearning_memory:
            val = rec.get('data', None)
            is_text = rec.get('is_text', False)

            if is_text:
                # Text record: convert to embedding if possible
                if isinstance(val, str) and self.embed_fn is not None:
                    vecs = self._embed_text(val)
                    for v in vecs:
                        out.append({'user_id': rec['user_id'], 'data': v, 'is_text': False})
                else:
                    # Skip text that cannot be converted to avoid errors
                    continue
                continue

            # Non-text: could be Tensor / list / other
            if isinstance(val, torch.Tensor):
                vecs = self._to_vec1d(val)
                for v in vecs:
                    out.append({'user_id': rec['user_id'], 'data': v, 'is_text': False})
            elif isinstance(val, (list, tuple)):
                # In history, list[Tensor] might have been stored in 'data'; expand here
                for item in val:
                    if isinstance(item, torch.Tensor):
                        for v in self._to_vec1d(item):
                            out.append({'user_id': rec['user_id'], 'data': v, 'is_text': False})
                    elif isinstance(item, str) and self.embed_fn is not None:
                        for v in self._embed_text(item):
                            out.append({'user_id': rec['user_id'], 'data': v, 'is_text': False})
                    # ignore other types
            # ignore other types

        return out


# ======= 2. Data_filtering =======
def get_rare_tokens(all_data, tokenizer, rare_token_fraction=0.05):
    """
    Count token frequencies across all texts and return the least frequent tokens.

    Args:
        all_data (str or list[str]): Input text or list of texts
        tokenizer: Tokenizer object (must have .tokenize() method)
        rare_token_fraction (float): Fraction of tokens considered rare (0~1)

    Returns:
        set: Set of rare tokens
    """
    # 1. Ensure all_data is a list
    if isinstance(all_data, str):
        all_data = [all_data]

    # 2. Count token frequency
    freq_dict = {}
    for text in all_data:
        tokens = tokenizer.tokenize(text)
        for tok in tokens:
            freq_dict[tok] = freq_dict.get(tok, 0) + 1

    if not freq_dict:
        return set()

    # 3. Sort by frequency (ascending)
    sorted_tokens = sorted(freq_dict.items(), key=lambda x: x[1])
    total_tokens = len(sorted_tokens)

    # 4. Determine cutoff count (at least 1 token)
    cutoff = max(1, int(total_tokens * rare_token_fraction))
    rare_part = sorted_tokens[:cutoff]

    # 5. Return set of rare tokens
    return {token for token, _ in rare_part}


class Data_filtering:
    """
    Data filtering utility, used to clean/preprocess data before unlearning.
    This example compares new data with historical unlearning data semantically:
    - If similarity is high, further check for rare tokens and cross-entropy loss difference.
    - Keeps chunks that matter, removes redundant/insignificant ones.

    Uses a SimCSE model for semantic embeddings and a Causal LM for cross-entropy scoring.
    """

    def __init__(
        self,
        simcse_model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased",
        rare_tokens: set = None,
        similarity_threshold: float = 0.95,
        chunk_size: int = 128,
        device: str = "cuda",
        epsilon: float = 0.1,  # threshold for loss difference before/after deletion
        # External language model + tokenizer for cross-entropy computation
        language_model=None,
        language_tokenizer=None
    ):
        """
        Args:
            simcse_model_name: Pretrained SimCSE model name for embeddings
            similarity_threshold: Semantic similarity threshold (cosine sim); above is considered duplicate
            chunk_size: Maximum length of each text chunk
            device: 'cuda' or 'cpu'
            rare_token_fraction: Fraction of tokens considered rare
            epsilon: Loss difference threshold before/after deletion
            language_model: Causal LM (e.g. GPT2, LLaMA) for cross-entropy computation
            language_tokenizer: Corresponding tokenizer
        """
        # ====== 1) Initialize SimCSE for cosine similarity ======
        self.tokenizer = AutoTokenizer.from_pretrained(simcse_model_name) # for rare token detection
        self.embedding_model = AutoModel.from_pretrained(simcse_model_name).to(device) # for embeddings
        self.embedding_model.eval()

        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.device = device
        self.epsilon = epsilon

        # ====== 2) Language model for cross-entropy ======
        self.language_model = language_model
        self.language_tokenizer = language_tokenizer
        if self.language_model is not None:
            self.language_model.to(self.device)
            self.language_model.eval()
        self.rare_tokens = rare_tokens

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of at most chunk_size characters. 
        Filters out empty chunks to avoid invalid model inputs.
        """
        text = text.strip()
        if not text:
            return []

        chunks = []
        for i in range(0, len(text), self.chunk_size):
            sub_text = text[i : i + self.chunk_size]
            if sub_text.strip():
                chunks.append(sub_text)
        return chunks

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get sentence embedding (pooler_output) using SimCSE.
        If text is empty, return a zero vector instead of error.
        """
        text = text.strip()
        if not text:
            return torch.zeros((1, 768), device=self.device)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.chunk_size,
            padding=True
        ).to(self.device)

        if inputs["input_ids"].size(1) == 0:
            return torch.zeros((1, 768), device=self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs, return_dict=True)
            embedding = outputs.pooler_output  # [batch_size, hidden_dim]

        return embedding  # shape: [1, hidden_dim]

    def _contains_rare_token(self, chunk: str, rare_tokens: set) -> bool:
        """
        Check if the chunk contains at least one rare token.
        """
        tokens = self.language_tokenizer.tokenize(chunk)
        return any(t in rare_tokens for t in tokens)

    def _compute_text_loss(self, text: str) -> float:
        """
        Compute average cross-entropy loss of text using the provided Causal LM.
        Returns 0.0 if no model/tokenizer is provided.
        """
        text = text.strip()
        if not text:
            return 0.0

        if self.language_model is None or self.language_tokenizer is None:
            return 0.0

        inputs = self.language_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(self.device)

        if inputs["input_ids"].dtype != torch.long:
            inputs["input_ids"] = inputs["input_ids"].long()

        if inputs["input_ids"].size(1) == 0:
            return 0.0

        with torch.no_grad():
            outputs = self.language_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # scalar (avg cross-entropy)

        return loss.item()

    def filter_data(self, new_data, unlearning_memory):
        """
        Filtering logic:
        1. Split new_data into chunks of size chunk_size
        2. Compare each chunk against historical memory using cosine similarity
        3. If similarity < threshold => keep
        4. If >= threshold => 
           - If contains rare token => keep
           - Else => temporarily delete chunk, compute loss difference
             * diff > epsilon => significant => keep chunk
             * diff <= epsilon => insignificant => delete chunk
        """
        new_data = new_data.strip()
        if not new_data:
            print("[Data_filtering] New data is empty. Returning an empty string.")
            return "","",[]

        # If no historical memory, keep everything
        if len(unlearning_memory) == 0:
            print("[Data_filtering] No unlearning memory found. Keeping the new data.")
            new_chunks = self._chunk_text(new_data)
            new_data_embedding = []
            for chunk in new_chunks:
                new_data_embedding.append(self._get_embedding(chunk)[0]) #[hidden_dim]
            return new_data,"",new_data_embedding
        memory_embeddings = torch.stack(unlearning_memory, dim=0).to(self.device)  # shape [M, hidden_dim]

        # ====== 2) Split new_data into chunks ======
        new_chunks = self._chunk_text(new_data)
        print(f"[Data_filtering] Found {len(self.rare_tokens)} rare tokens.")

        kept_chunks = []
        delete_chunks=[]
        # ====== 4) Process each chunk ======
        for chunk in new_chunks:
            # (a) Compute similarity with historical memory
            new_text_embedding = self._get_embedding(chunk)  # [1, hidden_dim]
            after_remove_chunks = [c for c in new_chunks if c != chunk]
            if torch.all(new_text_embedding == 0):
                # If empty/invalid embedding, just keep
                kept_chunks.append(chunk)
                continue

            cosine_scores = F.cosine_similarity(new_text_embedding, memory_embeddings)
            max_score = float(cosine_scores.max())
            # (b) If similarity < threshold, keep
            if max_score < self.similarity_threshold:
                kept_chunks.append(chunk)
                continue

            # (c) If similarity >= threshold, check rare tokens + loss difference
            if self._contains_rare_token(chunk, self.rare_tokens):
                kept_chunks.append(chunk)
            else:
                text_if_kept = "".join(new_chunks)
                L_kept = self._compute_text_loss(text_if_kept)

                text_if_removed = "".join(after_remove_chunks)
                L_removed = self._compute_text_loss(text_if_removed)

                diff = abs(L_kept - L_removed)
                if diff > self.epsilon:
                    kept_chunks.append(chunk)
                else:
                    delete_chunks.append(chunk)

        filtered_data = "".join(kept_chunks)
        filtered_data_embedding=[]
        for kept_chunk in kept_chunks:
            filtered_data_embedding.append(self._get_embedding(kept_chunk)[0])  # [hidden_dim]
        delete_data= "".join(delete_chunks)
        if filtered_data == "":
            print("[Data_filtering] No data left after filtering.")
        else:
            print("[Data_filtering] Get Filtered data, length =", len(filtered_data))
            if delete_data != "":
                print(f"The delete_data is: {delete_data}")
        if delete_data:
            return filtered_data, delete_data,filtered_data_embedding
        else:
            return filtered_data, "",filtered_data_embedding
