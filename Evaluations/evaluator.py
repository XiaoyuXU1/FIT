import os
import lm_eval
import random
import matplotlib.pyplot as plt
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import math
from rouge_score import rouge_scorer

# Probability
def Probability(model, tokenizer, questions, answers, batch_size=8):
    """
    Compute the normalized conditional probability P(a|q)^(1/|a|) for a batch of question-answer pairs.
    
    Args:
        model: Autoregressive language model.
        tokenizer: Tokenizer corresponding to the model.
        questions: List of questions.
        answers: List of answers aligned with the questions.
        batch_size: Batch size for processing.

    Returns:
        Average normalized probability across all pairs.
    """
    model.eval()
    results = []
    
    # Process data in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        
        sequences = []  # Store concatenated token sequences for each example
        q_lens = []     # Store token lengths of questions
        a_lens = []     # Store token lengths of answers
        
        # Encode and concatenate each pair
        for q, a in zip(batch_questions, batch_answers):
            q_tokens = tokenizer.encode(q, add_special_tokens=False)
            a_tokens = tokenizer.encode(a, add_special_tokens=False)
            q_lens.append(len(q_tokens))
            a_lens.append(len(a_tokens))
            sequences.append(q_tokens + a_tokens)
        
        # Pad sequences to the maximum length in the batch
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for seq in sequences:
            padded = seq + [pad_token_id] * (max_len - len(seq))
            padded_sequences.append(padded)
        
        # Build tensor on the model’s device
        input_tensor = torch.tensor(padded_sequences, dtype=torch.long, device=model.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        logits = outputs.logits  # shape: (batch_size, max_len, vocab_size)
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Compute log-probability for the answer portion of each example
        for b in range(len(sequences)):
            total_log_prob = 0.0
            q_len = q_lens[b]
            a_len = a_lens[b]
            seq = sequences[b]
            for pos in range(q_len, q_len + a_len):
                token_id = seq[pos]
                token_log_prob = log_probs[b, pos-1, token_id].item()
                total_log_prob += token_log_prob
            
            normalized_log_prob = total_log_prob / a_len
            normalized_prob = math.exp(normalized_log_prob)
            results.append(normalized_prob)
    
    ave_results = sum(results) / len(results) if results else float("nan")
    return ave_results


# Rouge_L
def Rouge_L(model, tokenizer, questions, target_texts, batch_size=8, max_length=128):
    """
    Compute the ROUGE score between generated answers and reference answers.

    Args:
        model: Generative model supporting `generate`.
        tokenizer: Tokenizer corresponding to the model.
        questions: List of input questions.
        target_texts: List of reference answers aligned with questions.
        batch_size: Batch size for processing.
        max_length: Maximum length of generated answers.

    Returns:
        avg_rougeL: The average ROUGE-L F-measure score across all pairs.
    """
    # Initialize ROUGE scorer with stemming
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    generated_texts = []

    # Generate answers in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        inputs = tokenizer(batch_questions, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        batch_generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts.extend(batch_generated)

    # Accumulate ROUGE scores
    total_rouge1 = total_rouge2 = total_rougeL = 0.0
    count = len(target_texts)
    
    for target, generated in zip(target_texts, generated_texts):
        score = scorer.score(target, generated)
        total_rouge1 += score['rouge1'].fmeasure
        total_rouge2 += score['rouge2'].fmeasure
        total_rougeL += score['rougeL'].fmeasure

    avg_scores = {
        'rouge1': total_rouge1 / count if count > 0 else 0,
        'rouge2': total_rouge2 / count if count > 0 else 0,
        'rougeL': total_rougeL / count if count > 0 else 0,
    }
    avg_rougeL = avg_scores['rougeL']
    return avg_rougeL


# Accuracy
def compute_acc_for_batch(texts, model, tokenizer, batch_size):
    """
    Compute token-level accuracy for a list of texts, in batches.

    Args:
        texts: List of text samples.
        model: Autoregressive language model.
        tokenizer: Tokenizer corresponding to the model.
        batch_size: Batch size for processing.

    Returns:
        List of accuracies, one per input text.
    """
    all_acc = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        preds = outputs.logits.argmax(dim=-1)  # [batch_size, seq_len]
        
        # Shift to align predictions and labels (ignore the first token)
        shift_preds = preds[:, :-1]
        shift_labels = inputs["input_ids"][:, 1:]
        shift_mask = inputs["attention_mask"][:, 1:]
        
        # Compute ACC per sequence
        for j in range(shift_preds.size(0)):
            valid_tokens = shift_mask[j].bool()
            num_valid = valid_tokens.sum().item()
            if num_valid == 0:
                acc = float("nan")
            else:
                correct = (shift_preds[j][valid_tokens] == shift_labels[j][valid_tokens]).sum().item()
                acc = correct / num_valid
            all_acc.append(acc)
    return all_acc


def Accuracy(model, tokenizer, forget_set, retain_set, batch_size):
    """
    Compute token-level accuracy on both forget and retain sets.

    Args:
        model: Autoregressive model.
        tokenizer: Corresponding tokenizer.
        forget_set: List of texts to forget.
        retain_set: List of texts to retain.
        batch_size: Batch size for processing.

    Returns:
        (avg_retain_acc, avg_forget_acc): Average accuracy for retain and forget sets.
    """
    # Accuracy for forget set
    forget_accs = compute_acc_for_batch(forget_set, model, tokenizer, batch_size)
    avg_forget_acc = sum(forget_accs) / len(forget_accs) if forget_accs else float("nan")
    
    # Accuracy for retain set
    retain_accs = compute_acc_for_batch(retain_set, model, tokenizer, batch_size)
    avg_retain_acc = sum(retain_accs) / len(retain_accs) if retain_accs else float("nan")
    return avg_retain_acc, avg_forget_acc
