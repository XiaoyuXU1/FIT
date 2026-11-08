"""Answer-only ROUGE and causal token metrics; independent of lm_eval."""
import math

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

GENERATION_BATCH_SIZE = 64


@torch.no_grad()
def evaluate(model, tokenizer, records, max_new_tokens=128, batch_size=GENERATION_BATCH_SIZE):
    model.eval()
    probabilities, accuracies, rouges = [], [], []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if hasattr(model,'prepare_queries'):
        model.prepare_queries([row['question'] for row in records])
    for row in records:
        if hasattr(model,'prepare_query'):
            model.prepare_query(row['question'])
        question = tokenizer.encode(row["question"], add_special_tokens=False)
        answer = tokenizer.encode(row["answer"], add_special_tokens=False)
        if not question or not answer:
            raise ValueError(f"Empty question/answer at PCH index {row['index']}")
        ids = torch.tensor([question + answer], device=model.device)
        logits = model(input_ids=ids, use_cache=False).logits[:, len(question)-1:-1]
        labels = torch.tensor(answer, device=model.device).unsqueeze(0)
        loss = F.cross_entropy(logits.float().transpose(1, 2), labels)
        probabilities.append(math.exp(-loss.item()))
        del logits
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True,
                           max_length=4096).to(model.device)
        logits = model(**inputs, use_cache=False).logits
        correct = logits[:, :-1].argmax(-1).eq(inputs.input_ids[:, 1:])
        accuracies.append(correct.float().mean().item())
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        if hasattr(model,'prepare_query_batch'):
            model.prepare_query_batch([row['question'] for row in batch])
        elif hasattr(model,'prepare_query') and len(batch)>1:
            raise ValueError('Gated generation requires query-specific batch gates')
        elif hasattr(model,'prepare_query'):
            model.prepare_query(batch[0]['question'])
        inputs = tokenizer([r["question"] for r in batch], padding=True,
                           truncation=True, max_length=4096, return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id)
        answers = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)
        rouges.extend(scorer.score(r["answer"], a)["rougeL"].fmeasure
                      for r, a in zip(batch, answers))
    if not records:
        raise ValueError("Cannot evaluate an empty set")
    p, a, r = (sum(x) / len(x) for x in (probabilities, accuracies, rouges))
    if not all(math.isfinite(value) for value in (p,a,r)):
        raise FloatingPointError('Evaluation produced non-finite metrics; refusing a completed result')
    return {"probability": p, "accuracy": a, "rougeL": r,
            "quality": (p * a * r) ** (1/3) * 100,
            "n": len(records)}


def relative_quality(value, reference):
    return max(0.0, 1.0 - abs(value/reference - 1.0)) if reference else 0.0
