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

# LayerAttributionSelector
class LayerAttributionSelector:
    """
      2) When new forget data is provided, perform a second round of filtering only among candidate layers
         - Similarly, compute the loss difference for each sample (or a single sample if only one exists),
           then take the average.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 num_layers,
                 device):
        self.num_layers = num_layers
        self.device = device
        self.tokenizer = tokenizer
        self.embedding_model = model
        # # Candidate layers selected on the retain set
        # self.candidate_layers = None
        # Used for temporarily storing and restoring layer parameters
        self._original_params = {}

    def _mask_layers(self, layer_indices: List[int]):
        """
        Set the parameters of the given layers to 0,
        while saving the original parameters for later restoration.
        """
        self._original_params = {}
        for idx in layer_indices:
            layer = self.embedding_model.model.layers[idx]
            layer_params = {}
            for name, param in layer.named_parameters():
                layer_params[name] = param.clone()
                # Zero out parameters
                param.data.zero_()
            self._original_params[idx] = layer_params

    def _restore_layers(self, layer_indices: List[int]):
        """
        Restore the original parameters of layers that were previously zeroed out.
        """
        for idx in layer_indices:
            layer = self.embedding_model.model.layers[idx]
            original_params = self._original_params[idx]
            for name, param in layer.named_parameters():
                param.data.copy_(original_params[name])
        self._original_params.clear()

    def _compute_loss_for_text(self, text: str) -> float:
        """
        Compute the loss for a single text input.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

    def _compute_loss_list_for_dataset(self, texts: List[str]) -> List[float]:
        """
        Return a list of loss values for each text in the dataset
        (differs from the function above by not averaging).
        """
        losses = []
        for t in texts:
            losses.append(self._compute_loss_for_text(t))
        return losses

    def select_forget_layers(self,
                             forget_data: str,
                             topk_for_forget: int = 5) -> List[int]:
        """
        Step 2: From self.candidate_layers (obtained via precompute_retain_candidates),
                filter further based on forget_data and select the top-k layers
                that most impact forget_data.

        Call precompute_retain_candidates() before using this method.
        """
        if not self.num_layers:
            raise ValueError(
                "Candidate layers have not been selected on the retain set. "
                "Please call precompute_retain_candidates() first."
            )

        print("Selecting layers most influential to forget_data ===")

        # If forget_data becomes a list in the future,
        # the same difference operation can be reused.
        # For now, assume forget_data is a single text, so only one loss value is computed.
        orig_loss_forget = self._compute_loss_for_text(forget_data)

        layer_forget_diff = []
        for layer_idx in range(self.num_layers):
            self._mask_layers([layer_idx])
            try:
                masked_loss_forget = self._compute_loss_for_text(forget_data)
            finally:
                self._restore_layers([layer_idx])

            # For a single text, the difference is simply masked_loss_forget - orig_loss_forget
            diff_forget = abs(masked_loss_forget - orig_loss_forget)
            layer_forget_diff.append((layer_idx, diff_forget))

        # Sort layers by their influence on forget_data, descending
        layer_forget_diff_sorted = sorted(layer_forget_diff, key=lambda x: x[1], reverse=True)
        final_selected_layers = [x[0] for x in layer_forget_diff_sorted[:topk_for_forget]]
        print("Attribution layers for forget_data:", layer_forget_diff_sorted)
        print(f"Top {topk_for_forget} most influential layers for forget_data: {final_selected_layers}")
        return final_selected_layers
