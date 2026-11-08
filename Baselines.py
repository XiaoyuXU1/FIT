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
import random
import matplotlib.pyplot as plt
import json
from collections import Counter
from Evaluations.evaluator import Probability, Rouge_L, Accuracy
import os
from typing import List, Dict, Any, Optional, Callable, Union
import time
from cycler import cycler
from Fit.algorithm import Baselines_FineTuning
from pathlib import Path
TensorLike = Union[torch.Tensor, List[torch.Tensor]]

def cubic_root(x):
    sign = 1 if x >= 0 else -1
    return sign * (abs(x) ** (1/3))

def log_results(log_file, user_id, FD, RQ):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)  # 关键：创建目录

    log_data = {
        "user_id": user_id,
        "Forget Degree": FD,
        "Retain Quality": RQ,
    }

    if not log_file.exists():
        with log_file.open("w", encoding="utf-8") as f:
            json.dump([log_data], f, indent=4, ensure_ascii=False)
    else:
        with log_file.open("r+", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):  # 防御：不是list就重置
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(log_data)
            f.seek(0)
            f.truncate()  # 建议：避免新内容比旧内容短导致尾巴残留
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

def Unlearning_Evaluator(model, tokenizer, forget_set, retain_set, forget_question, retain_question, forget_answer, retain_answer, batch_size, max_length, device):
    # Probability
    forget_probability_score=Probability(model, tokenizer, forget_question, forget_answer, batch_size)
    retain_probability_score=Probability(model, tokenizer, retain_question, retain_answer, batch_size)

    # Rouge_L
    forget_rouge=Rouge_L(model, tokenizer, forget_question, forget_answer, batch_size, max_length)
    retain_rouge=Rouge_L(model, tokenizer, retain_question, retain_answer, batch_size, max_length)

    # Accuracy
    forget_acc, retain_acc = Accuracy(model, tokenizer, forget_set, retain_set, batch_size)

    print("forget set \n")
    print(f"prob={forget_probability_score:.4f}, rougeL={forget_rouge:.4f}, acc={forget_acc:.4f}")
    print("retain set \n")
    print(f"prob={retain_probability_score:.4f}, rougeL={retain_rouge:.4f}, acc={retain_acc:.4f}")
    Forget = cubic_root(forget_probability_score * forget_rouge * forget_acc)*100
    Retain = cubic_root(retain_probability_score * retain_rouge * retain_acc)*100
    return Forget, Retain

# Linear interpolation function
def linear_interp_values(y_start, y_end, xs=(60, 120, 180, 240, 300), x0=0, x1=300):
    """
    y(x) = a*x + b, where (x0, y_start) and (x1, y_end) are endpoints.
    Return {x: y(x)} for x in xs.
    """
    if x1 == x0:
        raise ValueError("x1 must be different from x0")
    a = (y_end - y_start) / (x1 - x0)
    b = y_start - a * x0
    return [a * x + b for x in xs]
#Compute FQ and RQ curves
def compute_forget_retain_Q(
                            unlearning_model_name: str,
                            retain_model_name: str,
                            forget_set,
                            retain_set,
                            forget_question,
                            retain_question,
                            forget_answer,
                            retain_answer,
                            batch_size: int,
                            max_length: int,
                            device="cuda:0"):
    # 1) load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(retain_model_name)
    #For Qwen3
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    retain_model = AutoModelForCausalLM.from_pretrained(
        retain_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    unlearning_model = AutoModelForCausalLM.from_pretrained(
        unlearning_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    retain_model.eval()
    unlearning_model.eval()
    Forget_Q_start,Retain_Q_start=Unlearning_Evaluator(unlearning_model, tokenizer, forget_set, retain_set, forget_question, retain_question, forget_answer, retain_answer, batch_size, max_length, device)
    Forget_Q_end,Retain_Q_end=Unlearning_Evaluator(retain_model, tokenizer, forget_set, retain_set, forget_question, retain_question, forget_answer, retain_answer, batch_size, max_length, device)
    Forget_Q_curve = linear_interp_values(Forget_Q_start, Forget_Q_end)
    Retain_Q_curve = linear_interp_values(Retain_Q_start, Retain_Q_end)
    del retain_model
    del unlearning_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Forget_Q:", Forget_Q_curve)
    print("Retain_Q:", Retain_Q_curve)
    return Forget_Q_curve, Retain_Q_curve

# Calculate FD and RQ
def FD_RQ_calculate(unlearn_list, retain_list):
    out = []
    for A, B in zip(unlearn_list, retain_list):
        if B == 0:
            out.append(0)
        else:
            out.append(max(0.0, 1.0 - abs(A / B - 1.0)))
    return out

# ======= Main Pipeline: Comprehensive Example =======
class UnlearningPipeline:
    def __init__(self,
                 target_name="01-ai/Yi-6B",
                 forget_data: list[str] = None,
                 all_forget_retain_data: list[str] = None,
                 device="cuda:2"
                 ):
        self.forget_list=forget_data
        # 2) Load Target Model (actually perform unlearning operation) and infer model
        print(f"[UnlearningPipeline] Loading Target Model: {target_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_name,
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code= True,
                                                                attn_implementation="sdpa",
                                                                device_map="auto"
                                                            )
        self.infer_model=AutoModelForCausalLM.from_pretrained(target_name,
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code= True,
                                                                attn_implementation="sdpa",
                                                            )
        # self.infer_model.to(self.device)
        self.infer_model.to(self.target_model.device)
        self.target_tokenizer.padding_side = "left"
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        # Initialize other submodules
        self.device = self.target_model.device
        self.prev_m_vector=[]
        self.prev_mask_thr=[]
    def _compute_model_output(self, model, tokenizer, text: str, max_length) -> np.ndarray:
        """
        Simplified: Let the Target Model infer the text once to obtain a vector (e.g., softmax logits).
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length').to(self.target_model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # Check if logits contain invalid values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Logits contain invalid values")

        # Clip logits to prevent numerical overflow
        logits = torch.clamp(logits, min=-1e6, max=1e6)

        # Compute log_softmax
        softmax_logits = torch.nn.functional.log_softmax(logits, dim=-1)  # Apply softmax to logits
        vec = softmax_logits.mean(dim=1).squeeze(0)  # Take average => obtain (vocab_size,)

        # Again check if vec contains invalid values
        if torch.isnan(vec).any() or torch.isinf(vec).any():
            raise ValueError("vec contains invalid values")

        # **Convert dtype here to avoid unsupported BFloat16 -> numpy issue**
        vec = vec.to(torch.float16)

        return vec.cpu().numpy()


    def unlearning_request_handler(self, user_id: str, x_forget: str, x_retain: list[str],unlearning_algorithm, fine_tuning_chunk_size, lr, epochs):
        print(f"\n[UnlearningPipeline] Received unlearning request from user={user_id}")  #, data={x_forget}
        # Fine-tuning

        fine_tuner = Baselines_FineTuning(self.target_model,self.infer_model,self.target_tokenizer,lr=lr,device=self.target_model.device)
        if unlearning_algorithm=="ALKN":
            if user_id=="User0":
                m_vector, mask_thr=fine_tuner.run_baselines_finetuning(unlearning_algorithm, x_retain, x_forget,epochs,chunk_size=fine_tuning_chunk_size)
                self.prev_m_vector.append(m_vector)
                self.prev_mask_thr.append(mask_thr)
            else:
                m_vector, mask_thr=fine_tuner.run_baselines_finetuning(unlearning_algorithm, x_retain, x_forget,epochs,chunk_size=fine_tuning_chunk_size,prev_m_vector=self.prev_m_vector[-1],prev_mask_thr=self.prev_mask_thr[-1])
                self.prev_m_vector.append(m_vector)
                self.prev_mask_thr.append(mask_thr)
        else:
            fine_tuner.run_baselines_finetuning(unlearning_algorithm, x_retain, x_forget,epochs,chunk_size=fine_tuning_chunk_size)
        print("[Pipeline] Done.\n")
        return self.target_model,self.target_tokenizer

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Unlearning Pipeline Hyperparameters")

    # Core hyperparameters
    parser.add_argument("--unlearning_model_name",type=str,default="FITPCH/Llama-2-7b-chat-hf_PCH_finetune", help="unlearning model names")
    parser.add_argument("--retain_model_name",type=str,default="FITPCH/Llama-2-7b-chat-hf_PCH_retain", help="retain model names")
    parser.add_argument("--unlearning_algorithm",type=str,default="GA_GD", help="baselines")
    parser.add_argument("--device", type=str, default="cuda:2",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    return parser.parse_args()

# ====================== Test Example ======================
if __name__ == "__main__":
    args = parse_args()
    # Set random seed (ensure reproducibility)
    PCH_file_path = 'FITPCH/PCH'  # Forget set
    PCH_QA_path="FITPCH/PCH_QA"
    Pch=load_dataset(PCH_file_path)["train"]["text"]
    Pch_QA=load_dataset(PCH_QA_path)
    Pch_Q=Pch_QA['train']["question"]
    Pch_A=Pch_QA['train']["answer"]
    # Parameter settings
    lr =5e-6   # Learning rate for fine-tuning
    epochs = 3  # Number of fine-tuning epochs
    batch_size=10
    max_length = 128  # Maximum length for computing p_output
    fine_tuning_chunk_size = 256  # Chunk size for fine-tuning
    device = args.device  # Device to use for training
    Unlearning_model_name=args.unlearning_model_name
    retain_model_name=args.retain_model_name
    unlearning_algorithm=args.unlearning_algorithm
    # Compute FQ and RQ curves
    Forget_Q_list,Retain_Q_list=compute_forget_retain_Q(Unlearning_model_name, retain_model_name, Pch[0:300], Pch[300:], Pch_Q[0:300],Pch_Q[300:], Pch_A[0:300], Pch_A[300:], batch_size, max_length)
    Unlearning_model_file_name=Unlearning_model_name.replace("/", "_")
    seed_list=[42]
    # Initialize main Pipeline
    for m in range(len(seed_list)):
        random.seed(seed_list[m])
        tmp = list(zip(Pch[:300], Pch_Q[:300], Pch_A[:300]))
        random.shuffle(tmp)
        Pch[:300], Pch_Q[:300], Pch_A[:300] = map(list, zip(*tmp))
        Forget_list=[]
        Retain_list=[]
        pipeline = UnlearningPipeline(
            target_name=Unlearning_model_name,
            forget_data=Pch[0:300],
            all_forget_retain_data=Pch,
            device=device,  # Use "cuda" if GPU is available
        )
        log_result_path=f"Experiment_record/continuous/{Unlearning_model_file_name}/seed{seed_list[m]}_evaluations_all.json"
        for i in range(300):
            user_id="User"+str(i)
            model,tokenizer= pipeline.unlearning_request_handler(user_id, Pch[i], Pch[i+1:], unlearning_algorithm, fine_tuning_chunk_size, lr, epochs)
            if (i+1) % 60 == 0:
                #Calculate F and R
                Forget,Retain = Unlearning_Evaluator(model, tokenizer, Pch[0:i+1], Pch[i+1:], Pch_Q[0:i+1], Pch_Q[i+1:], Pch_A[0:i+1], Pch_A[i+1:], batch_size, max_length, device)
                Forget_list.append(Forget)
                Retain_list.append(Retain)
        print("Forget_list:", Forget_list)
        print("Retain_list:", Retain_list)
        #Calculate FQ and RQ
        User_id=[60,120,180,240,300]
        FD_list=FD_RQ_calculate(Forget_list,Forget_Q_list)
        RQ_list=FD_RQ_calculate(Retain_list,Retain_Q_list)
        log_result_path=f"Experiment_record/{Unlearning_model_file_name}/{unlearning_algorithm}_seed{seed_list[m]}_evaluations_all.json"
        # Log results for FD and RQ for each 60 users
        for idx in range(len(User_id)):
            log_results(
                log_file=log_result_path,
                user_id=User_id[idx],
                FD=FD_list[idx],
                RQ=RQ_list[idx]
            )
        model_save_path = Path(f"Model/{Unlearning_model_file_name}/{unlearning_algorithm}_seed{seed_list[m]}")
        model_save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        del model
        del tokenizer
        # Force garbage collection
        gc.collect()
        # Clear PyTorch GPU cache
        torch.cuda.empty_cache()
        # Clear PyTorch IPC cache
        torch.cuda.ipc_collect()
