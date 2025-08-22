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
from Fit.filtering import Unlearn_request_memory, Data_filtering,get_rare_tokens
from Fit.layer import LayerAttributionSelector

TensorLike = Union[torch.Tensor, List[torch.Tensor]]
def log_results(
    log_file,
    user_id,
    forget_probability_score,
    forget_rouge,
    forget_acc,
    retain_probability_score,
    retain_rouge,
    retain_acc,
):
    """
    Record log data into a file.

    Args:
      log_file: Path to the log file. Must end with .json
      user_id: User ID.
      forget_probability_score: Forget probability score.
      forget_rouge: ROUGE score on the forget set.
      retain_probability_score: Retain probability score.
      retain_rouge: ROUGE score on the retain set.
      retain_acc: Token-level accuracy on the retain set.
      mmlu_acc: Accuracy on MMLU dataset.
      gsm8k_acc: Accuracy on GSM8K dataset.
      commonsense_acc: Accuracy on commonsense dataset.
    """
    # Construct the record data dictionary
    log_data = {
        "user_id": user_id,
        "forget_probability_score": forget_probability_score,
        "forget_rouge": forget_rouge,
        "forget_acc": forget_acc,
        "retain_probability_score": retain_probability_score,
        "retain_rouge": retain_rouge,
        "retain_acc": retain_acc,
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

def read_json_as_list(file_path):
    """
    Read a JSON file and return a list.
    If the root object of the JSON file is not a list, wrap it into a list and return.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    else:
        return [data]



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
    

def Unlearning_Evaluator(user_id, model_name, model, tokenizer, forget_set, retain_set, forget_question, retain_question, forget_answer, retain_answer, batch_size, max_length, log_result_path, device):
    # Probability
    forget_probability_score=Probability(model, tokenizer, forget_question, forget_answer, batch_size)
    retain_probability_score=Probability(model, tokenizer, retain_question, retain_answer, batch_size)

    # Rouge_L
    forget_rouge=Rouge_L(model, tokenizer, forget_question, forget_answer, batch_size, max_length)
    retain_rouge=Rouge_L(model, tokenizer, retain_question, retain_answer, batch_size, max_length)

    # Accuracy
    retain_acc,forget_acc=Accuracy(model, tokenizer, forget_set, retain_set, batch_size)

    print("forget set \n")
    print(f"prob={forget_probability_score:.4f}, rougeL={forget_rouge:.4f}, acc={forget_acc:.4f}")
    print("retain set \n")
    print(f"prob={retain_probability_score:.4f}, rougeL={retain_rouge:.4f}, acc={retain_acc:.4f}")
    log_results(log_result_path, 
                user_id,
                forget_probability_score,
                forget_rouge,
                forget_acc,
                retain_probability_score,
                retain_rouge,
                retain_acc,
                )

# ======= Main Pipeline: Comprehensive Example =======
class UnlearningPipeline:
    def __init__(self,
                 target_name="01-ai/Yi-6B",
                 forget_data: list[str] = None,
                 all_forget_retain_data: list[str] = None,
                 similarity_threshold: float = 0.95,
                 Data_filtering_chunk_size: int = 128,
                 device="cuda",
                 rare_token_fraction: float = 0.05,
                 epsilon: float = 0.1
                 ):
        self.delete_data_list=[]
        self.all_chosen_layers_list=[]
        self.device = device
        self.rare_token_fraction = rare_token_fraction
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
        self.rare_tokens=get_rare_tokens(all_forget_retain_data, self.target_tokenizer, self.rare_token_fraction)
        self.data_filtering = Data_filtering(
                                            simcse_model_name = "princeton-nlp/sup-simcse-bert-base-uncased",
                                            rare_tokens=self.rare_tokens,
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
    parser.add_argument("--topk_for_forget", type=int, default=8,
                        help="Number of top-k layers to select for forgetting")
    parser.add_argument("--similarity_threshold", type=float, default=0.90,
                        help="Similarity threshold for Data_filtering")
    parser.add_argument("--rare_token_fraction", type=float, default=0.1,
                        help="Fraction of rare tokens used in Data_filtering")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Epsilon threshold for Data_filtering")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
# ====================== Test Example ======================
if __name__ == "__main__":
    # Set random seed (ensure reproducibility)
    forget_file_path = 'Dataset/Forget_set/Forget_merged.json'  # Forget set
    retain_file_path = 'Dataset/Retain_set/Retain_merged.json'  # Retain set
    forget_QA_path="Dataset/Forget_set/Forget_QA.json"
    retain_QA_path="Dataset/Retain_set/Retain_QA.json"
    forget_list = read_json_as_list(forget_file_path)
    retain_list= read_json_as_list(retain_file_path)
    forget_QA_list = read_json_as_list(forget_QA_path)
    retain_QA_list= read_json_as_list(retain_QA_path)
    forget_question = [data['question'] for data in forget_QA_list]
    forget_answer = [data['answer'] for data in forget_QA_list]
    retain_question= [data['question'] for data in retain_QA_list]
    retain_answer= [data['answer'] for data in retain_QA_list]
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
    rare_token_fraction = args.rare_token_fraction  # Rare token fraction for Data_filtering
    device = args.device  # Device to use for training
    epsilon = args.epsilon  # Epsilon for Data_filtering
    Unlearning_model_name=["Model/Finetune/Llama-2-7b-chat-hf","Model/Finetune/Meta-Llama-3-8B" ,"Model/Finetune/Meta-Llama-3-8B-Instruct" , "Model/Finetune/Yi-6B"]
    Unlearning_model_file_name=["Llama-2-7b-chat-hf","Meta-Llama-3-8B","Meta-Llama-3-8B-Instruct" ,"Yi-6B"]
    seed_list=[20,30,40,50] 
    all_data=retain_list+forget_list
    # Initialize main Pipeline
    for m in range(len(seed_list)):
        random.seed(seed_list[m])
        # Shuffle lists
        random.shuffle(forget_list)
        random.shuffle(retain_list)
        for j in range(len(Unlearning_model_name)):
            pipeline = UnlearningPipeline(
                target_name=Unlearning_model_name[j],
                forget_data=forget_list,
                all_forget_retain_data=all_data,
                similarity_threshold=similarity_threshold,
                Data_filtering_chunk_size=Data_filtering_chunk_size,
                device=device,  # Use "cuda" if GPU is available
                rare_token_fraction=rare_token_fraction,
                epsilon=epsilon
            )
            # Initiate unlearning request
            start_time = time.time()
            log_result_path=f"Experiment_record/continuous/{Unlearning_model_file_name[j]}/seed{seed_list[m]}_evaluations_all.json"
            for i in range(len(forget_list)):
                user_id="User"+str(i)
                model,tokenizer,all_chosen_layers_list,delete_list= pipeline.unlearning_request_handler(user_id, forget_list[i], retain_list, max_length,topk_for_forget, fine_tuning_chunk_size, lr, epochs)
                if (i+1) % 60 == 0:
                    Unlearning_Evaluator(user_id, Unlearning_model_name[j], model, tokenizer, forget_list, retain_list, forget_question, retain_question, forget_answer, retain_answer, batch_size, max_length, log_result_path,device)
            end_time = time.time() 
            print(f"Training time: {end_time - start_time:.2f} seconds")
            # Figure 1: Retain Accuracy vs Forget Requests  

            # # Save model weights
            # # Check if folder exists, if not create it
            with open(f"Experiment_record/continuous/{Unlearning_model_file_name[j]}/seed{seed_list[m]}_comparison.json", "w", encoding="utf-8") as f:
                json.dump(delete_list, f, ensure_ascii=False, indent=4)
            model_save_path=f"Model/continuous/{Unlearning_model_file_name[j]}/seed{seed_list[m]}"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            # Flatten the list of lists into a single list
            # -------------------- Research-style plotting settings --------------------
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "axes.labelsize": 20,
                "axes.titlesize": 22,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 20,
                "figure.dpi": 150,
                "axes.prop_cycle": cycler('color', [
                    "#A6CEE3", "#FDBF6F", "#B2DF8A", "#FB9A99",
                    "#CAB2D6", "#E8E817", "#CCEBC5", "#BE0C0C"
                ])
            })
            # -------------------- Data processing --------------------
            flat_layers = [layer for sublist in all_chosen_layers_list for layer in sublist]
            layer_counts = Counter(flat_layers)
            layers = sorted(layer_counts.keys())
            counts = [layer_counts[l] for l in layers]

            # -------------------- Draw histogram --------------------
            plt.figure(figsize=(6, 4))
            plt.bar(layers, counts, width=0.6)
            plt.xlabel("Layer Index")
            plt.ylabel("Frequency")
            plt.title("Histogram of Chosen Layers")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            # -------------------- Save as PDF --------------------
            save_path = f"Figures/layer/{Unlearning_model_file_name[j]}_layer_histogram.pdf"
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Histogram saved to {save_path}")
            del model
            del tokenizer
            # Force garbage collection
            gc.collect()

            # Clear PyTorch GPU cache
            torch.cuda.empty_cache()
            # Clear PyTorch IPC cache
            torch.cuda.ipc_collect()
