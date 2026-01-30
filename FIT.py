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
from Evaluations.evaluator import Probability, Rouge_L, Accuracy
import os
from typing import List, Dict, Any, Optional, Callable, Union
import time
from cycler import cycler
from Fit.algorithm import FineTuning
from Fit.filtering import Unlearn_request_memory, Data_filtering
from Fit.layer import LayerAttributionSelector

TensorLike = Union[torch.Tensor, List[torch.Tensor]]

def cubic_root(x):
    sign = 1 if x >= 0 else -1
    return sign * (abs(x) ** (1/3))

def log_results(
    log_file,
    user_id,
    FD,
    RQ
):
    """
    Record log data into a file.

    Args:
      log_file: Path to the log file. Must end with .json
      user_id: User ID.
      Forget Degree: Forget Degree.
      Retain Quality: Retain Quality.
    """
    # Construct the record data dictionary
    log_data = {
        "user_id": user_id,
        "Forget Degree": FD,
        "Retain Quality": RQ,
    }
    
    # If the file does not exist, create a new file and write data (stored as a list for later appending)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([log_data], f, indent=4, ensure_ascii=False)
    else:
        # If the file exists, read existing data, append, and rewrite
        with open(log_file, 'r+') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            existing_data.append(log_data)
            f.seek(0)  # Move the file pointer to the beginning
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
    return {x: a * x + b for x in xs}
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
                            device: str):
    # 1) load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(retain_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    retain_model = AutoModelForCausalLM.from_pretrained(
        retain_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2= True,
    ).to(device)
    unlearning_model = AutoModelForCausalLM.from_pretrained(
        unlearning_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2= True,
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
                 similarity_threshold: float = 0.95,
                 Data_filtering_chunk_size: int = 128,
                 device="cuda",
                 epsilon: float = 0.1
                 ):
        self.delete_data_list=[]
        self.all_chosen_layers_list=[]
        self.device = device
        self.epsilon = epsilon
        self.forget_list=forget_data
        # 2) Load Target Model (actually perform unlearning operation) and infer model
        print(f"[UnlearningPipeline] Loading Target Model: {target_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_name,       
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code= True,
                                                                use_flash_attention_2= True,
                                                            )
        self.infer_model=AutoModelForCausalLM.from_pretrained(target_name,       
                                                                torch_dtype=torch.bfloat16,
                                                                trust_remote_code= True,
                                                                use_flash_attention_2= True,
                                                            )                                             
        self.infer_model.to(self.device)
        self.target_model.to(self.device)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        # Initialize other submodules
        self.request_memory = Unlearn_request_memory()
        num_layers = len(self.target_model.model.layers)
        self.data_filtering = Data_filtering(
                                            simcse_model_name = "princeton-nlp/sup-simcse-bert-base-uncased",
                                            similarity_threshold= similarity_threshold,
                                            chunk_size = Data_filtering_chunk_size,
                                            device = self.device,
                                            epsilon=self.epsilon,
                                            language_model=self.target_model,
                                            language_tokenizer=self.target_tokenizer)
        self.layer_selector = LayerAttributionSelector(model=self.target_model,tokenizer=self.target_tokenizer, num_layers=num_layers,device=self.device)
        
    def _compute_model_output(self, model, tokenizer, text: str, max_length) -> np.ndarray:
        """
        Simplified: Let the Target Model infer the text once to obtain a vector (e.g., softmax logits).
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding='max_length').to(self.device)
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


    def unlearning_request_handler(self, user_id: str, x_forget: str, x_retain: list[str], max_length,topk_for_forget, fine_tuning_chunk_size, lr, epochs):
        print(f"\n[UnlearningPipeline] Received unlearning request from user={user_id}")  #, data={x_forget}
        all_records = self.request_memory.retrieve_unlearning_data()
        all_data = [r["data"] for r in all_records]
        filtered_data, delete_data, filtered_data_embedding = self.data_filtering.filter_data(x_forget, all_data)
        print(f"the length of original_data is: {len(x_forget)}")
        print(f"the length of filtered_data is: {len(filtered_data)}")
        if delete_data:
            self.delete_data_list.append(delete_data)
        if not filtered_data:
            return 0,0 # No need to continue
        print("[Pipeline] Start Layer choices ...")
        chosen_layer_index = self.layer_selector.select_forget_layers(filtered_data,topk_for_forget=topk_for_forget)
        self.all_chosen_layers_list.append(chosen_layer_index)
        # Fine-tuning
        fine_tuner = FineTuning(self.target_model,self.infer_model,self.target_tokenizer,chosen_layer_index,lr=lr,device=self.device)
        fine_tuner.run_finetuning(x_retain , filtered_data,epochs=epochs,chunk_size=fine_tuning_chunk_size)
        # Store data
        self.request_memory.store_unlearning_data(user_id, filtered_data_embedding)
        print("[Pipeline] Done.\n") 
        return self.target_model,self.target_tokenizer,self.all_chosen_layers_list,self.delete_data_list
    
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Unlearning Pipeline Hyperparameters")
    
    # Core hyperparameters
    parser.add_argument("--unlearning_model_name",type=str,default="FITPCH/Llama-2-7b-chat-hf_PCH_finetune", help="unlearning model names")
    parser.add_argument("--retain_model_name",type=str,default="FITPCH/Llama-2-7b-chat-hf_PCH_retain", help="retain model names")
    parser.add_argument("--topk_for_forget", type=int, default=8,
                        help="Number of top-k layers to select for forgetting")
    parser.add_argument("--similarity_threshold", type=float, default=0.90,
                        help="Similarity threshold for Data_filtering")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Epsilon threshold for Data_filtering")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    return parser.parse_args()

# ====================== Test Example ======================
if __name__ == "__main__":
    # Set random seed (ensure reproducibility)
    PCH_file_path = 'FITPCH/PCH'  # Forget set
    PCH_QA_path="FITPCH/PCH_QA"
    Pch=load_dataset(PCH_file_path)["train"]["text"]
    Pch_QA=load_dataset(PCH_QA_path)
    Pch_Q=Pch_QA['train']["question"]
    Pch_A=Pch_QA['train']["answer"]
    # Parameter settings
    lr =3e-5   # Learning rate for fine-tuning
    epochs = 3  # Number of fine-tuning epochs
    batch_size=10
    max_length = 128  # Maximum length for computing p_output
    fine_tuning_chunk_size = 1024  # Chunk size for fine-tuning
    Data_filtering_chunk_size = 8 # Chunk size for Data_filtering 
    args = parse_args()
    topk_for_forget = args.topk_for_forget # Number of topk layers to select
    similarity_threshold = args.similarity_threshold  # Similarity threshold for Data_filtering
    device = args.device  # Device to use for training
    epsilon = args.epsilon  # Epsilon for Data_filtering
    Unlearning_model_name=args.unlearning_model_name
    retain_model_name=args.retain_model_name
    # Compute FQ and RQ curves
    Forget_Q_list,Retain_Q_list=compute_forget_retain_Q(Unlearning_model_name, retain_model_name, Pch[0:300], Pch[300:], Pch_Q[0:300],Pch_Q[300:], Pch_A[0:300], Pch_A[300:], batch_size, max_length, device)
    Unlearning_model_file_name=Unlearning_model_name.replace("/", "_")
    seed_list=[20,30,40,50] 
    # Initialize main Pipeline
    for m in range(len(seed_list)):
        random.seed(seed_list[m])
        # Shuffle lists
        for L in (Pch, Pch_Q, Pch_A):
            tmp = L[:300]
            random.shuffle(tmp)
            L[:300] = tmp
        Forget_list=[]
        Retain_list=[]
        pipeline = UnlearningPipeline(
            target_name=Unlearning_model_name,
            forget_data=Pch[0:300],
            all_forget_retain_data=Pch,
            similarity_threshold=similarity_threshold,
            Data_filtering_chunk_size=Data_filtering_chunk_size,
            device=device,  # Use "cuda" if GPU is available
            epsilon=epsilon
        )
        log_result_path=f"Experiment_record/continuous/{Unlearning_model_file_name}/seed{seed_list[m]}_evaluations_all.json"
        for i in range(300):
            user_id="User"+str(i)
            model,tokenizer,all_chosen_layers_list,delete_list= pipeline.unlearning_request_handler(user_id, Pch[i], Pch[i+1:], max_length, topk_for_forget, fine_tuning_chunk_size, lr, epochs)
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
        log_result_path=f"Experiment_record/continuous/{Unlearning_model_file_name}/seed{seed_list[m]}_evaluations_all.json"
        # Log results for FD and RQ for each 60 users
        for idx in range(len(User_id)):
            log_results(
                log_file=log_result_path,
                user_id=User_id[idx],
                FD=FD_list[idx],
                RQ=RQ_list[idx]
            )
        # # Save model weights
        # # Check if folder exists, if not create it
        with open(f"Experiment_record/continuous/{Unlearning_model_file_name}/seed{seed_list[m]}_comparison.json", "w", encoding="utf-8") as f:
            json.dump(delete_list, f, ensure_ascii=False, indent=4)
        model_save_path=f"Model/continuous/{Unlearning_model_file_name}/seed{seed_list[m]}"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
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









