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
from Unlearning_algorithm.GA import Gradient_ascent 
from Unlearning_algorithm.RL import Random_label
from Unlearning_algorithm.NPO_KL import NPO_KL
from Unlearning_algorithm.GA_GD import Gradient_ascent_descent
from Unlearning_algorithm.GA_KL import Gradient_ascent_KL 
from Unlearning_algorithm.NPO import NPO
from collections import Counter
import os
from typing import List, Dict, Any, Optional, Callable, Union
import time
from cycler import cycler
TensorLike = Union[torch.Tensor, List[torch.Tensor]]
# Define a custom learning rate scheduler
class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0.0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_step = max(0, self.last_epoch)
        if current_step < self.num_warmup_steps:
            # Warm-up phase
            lr_scale = float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            # Cosine annealing phase
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = (1 - self.eta_min) * cosine_decay + self.eta_min

        return [base_lr * lr_scale for base_lr in self.base_lrs]
    
# Unlearning algorithm selection
def compute_importance(text, model, tokenizer):
    # Encode the text into token ids
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)

    # Obtain embeddings and enable gradient tracking
    with torch.no_grad():
        # Important: create a gradient-tracked copy outside no_grad
        embeddings = model.model.embed_tokens(input_ids)
    embeddings = embeddings.detach().clone().requires_grad_(True)

    # Forward pass using inputs_embeds, specify labels to compute language model loss
    outputs = model(
        inputs_embeds=embeddings,
        labels=input_ids,
        use_cache=False  # Some models require disabling cache when using inputs_embeds+labels
    )
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Get gradients and compute norm
    grad = embeddings.grad
    importance_score = grad.norm().item()

    # Clear model gradients to avoid affecting subsequent computations
    model.zero_grad()

    return importance_score

class FineTuning:
    def __init__(self, model, infer_model, tokenizer, selected_layers, lr=1e-6, device='cuda'):
        """
        Args:
            model: Pretrained model
            tokenizer: Tokenizer
            selected_layers: List of layer indices to unfreeze
            lr: Learning rate
            device: Device ('cpu' or 'cuda')
        """
        self.selected_layers = selected_layers
        self.device = device

        # Load pretrained model and tokenizer
        self.tokenizer = tokenizer
        self.model = model
        self.infer_model = infer_model
        self.model.to(self.device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze selected layers
        self._unfreeze_selected_layers()
        # Print unfrozen layers
        print(f"[FineTuning] Unfrozen layers: {self.selected_layers}")
        # Prepare optimizer
        params_to_update = [p for name, p in self.model.named_parameters() if p.requires_grad]
        self.params_to_update = params_to_update
        if not params_to_update:
            raise ValueError("No parameters to optimize")
        self.lr = lr

    def _unfreeze_selected_layers(self):
        """
        Unfreeze the selected layers.
        """
        for name, param in self.model.named_parameters():
            for layer_idx in self.selected_layers:
                if f"layers.{layer_idx}.mlp" in name:
                    param.requires_grad = True
        for name, param in self.model.named_parameters():
            for layer_idx in self.selected_layers:
                if f"layers.{layer_idx}.self_attn" in name:
                    param.requires_grad = True

    def run_finetuning(self, retain_set: list[str], forget_data: str, epochs: int = 20, chunk_size: int = 4096):
        """
        Fine-tune the model on the selected layers.

        Args:
            retain_set: Retain dataset (list of texts)
            forget_data: Forget dataset (text to be unlearned)
            epochs: Number of training epochs
            chunk_size: Size of each training chunk
        """
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)  
        
        data_length = len(forget_data)
        num_chunks = math.ceil(data_length / chunk_size)
        total_steps = epochs * num_chunks
        warmup_steps = int(0.1 * total_steps)
        
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            eta_min=0.1
        )
        score = compute_importance(forget_data, self.model, self.tokenizer)
        print(f"The importance of data: {score}")
        if score >= 12:
            unlearning_algorithm = "NPO_KL"
        elif score < 12 and score >= 6:  
            unlearning_algorithm = "NPO"
        else:
            unlearning_algorithm = "RL"
        print(unlearning_algorithm)

        if unlearning_algorithm == "GA":
            Gradient_ascent(self.model, self.tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, self.device)
        elif unlearning_algorithm == "GA_GD":
            Gradient_ascent_descent(self.model, self.tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, self.device)
        elif unlearning_algorithm == "GA_KL":
            Gradient_ascent_KL(self.model, self.infer_model, self.tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, self.device)
        elif unlearning_algorithm == "NPO_KL":
            NPO_KL(self.model, self.infer_model, self.tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, self.device)
        elif unlearning_algorithm == "RL":
            Random_label(self.model, self.tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, self.device)
        elif unlearning_algorithm == "NPO":
            NPO(self.model, self.infer_model, self.tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, self.device)        

        print("[FineTuning] Fine-tuning complete.")

