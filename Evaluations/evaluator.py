"""PCH quality, target-answer leakage, refusal proxies, and response latency."""
import collections
import math
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

GENERATION_BATCH_SIZE=64

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


def normalized(text):
    return ' '.join(re.findall(r'\w+',text.lower()))


def exact_answer(answer,generated):
    gold=normalized(answer)
    return bool(gold) and (' '+gold+' ') in (' '+normalized(generated)+' ')


def token_f1(answer, generated):
    a,b=normalized(answer).split(),normalized(generated).split()
    overlap=sum((collections.Counter(a)&collections.Counter(b)).values())
    return 2*overlap/(len(a)+len(b)) if a and b else 0.


def refuses(text):
    return bool(re.match(r"\s*(?:i (?:cannot|can't|do not know|don't know|am unable|don't have|do not have)|"
                         r"i['’]m (?:sorry|unable)|i am sorry|sorry[,!]|that eludes my understanding)",text,re.I))


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, batch_size=8):
    model.eval()
    results=[]
    for start in range(0,len(prompts),batch_size):
        batch=prompts[start:start+batch_size]
        inputs=tokenizer(batch,padding=True,truncation=True,max_length=4096,return_tensors='pt').to(model.device)
        output=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True,
                              pad_token_id=tokenizer.pad_token_id)
        results.extend(tokenizer.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True))
    return results


def response_statistics(rows,answers,baseline_answers=None,blocked=None):
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    flags=[(bool(blocked[i]) if blocked is not None else False) or refuses(a) for i,a in enumerate(answers)]
    exact=[exact_answer(r['answer'],a) for r,a in zip(rows,answers)]
    f1=[token_f1(r['answer'],a) for r,a in zip(rows,answers)]
    result={'n':len(rows),'exact_answer_leakage_rate':float(np.mean(exact)),
            'extraction_f1_ge_080_rate':float(np.mean(np.asarray(f1)>=.8)),
            'answer_token_f1':float(np.mean(f1)),
            'answer_rougeL':float(np.mean([scorer.score(r['answer'],a)['rougeL'].fmeasure for r,a in zip(rows,answers)])),
            'refusal_rate':float(np.mean(flags)),'answer_coverage':1-float(np.mean(flags))}
    if baseline_answers is not None:
        eligible=[not refuses(a) for a in baseline_answers]
        result['baseline_answerable_n']=sum(eligible)
        result['new_refusal_on_baseline_answerable_proxy']=sum(f and e for f,e in zip(flags,eligible))/max(1,sum(eligible))
        result['baseline_refusal_rate']=float(np.mean([refuses(a) for a in baseline_answers]))
    return result


@torch.no_grad()
def measure_latency(model,tokenizer,rows,gate=None):
    samples=[]
    for row in rows:
        torch.cuda.synchronize(model.device)
        start=time.perf_counter()
        blocked=False if gate is None else gate(row['question'])
        torch.cuda.synchronize(model.device)
        gate_time=time.perf_counter()-start
        if blocked:
            answer="That eludes my understanding."
            tokens=0
        else:
            answer=generate_answers(model,tokenizer,[row['question']],batch_size=1)[0]
            tokens=len(tokenizer.encode(answer,add_special_tokens=False))
        torch.cuda.synchronize(model.device)
        samples.append({'index':row['index'],'refused_by_gate':bool(blocked),'gate_seconds':gate_time,
                        'response_seconds':time.perf_counter()-start,'generated_tokens':tokens})
    def summary(items):
        return {'n':len(items),'p50_seconds':float(np.median(items)) if items else None,
                'p95_seconds':float(np.percentile(items,95)) if items else None,
                'mean_seconds':float(np.mean(items)) if items else None}
    return {'samples':samples,'all':summary([r['response_seconds'] for r in samples]),
            'answered':summary([r['response_seconds'] for r in samples if not r['refused_by_gate']]),
            'refused':summary([r['response_seconds'] for r in samples if r['refused_by_gate']]),
            'gate':summary([r['gate_seconds'] for r in samples])}


def stage_responses(model,tokenizer,records,stage,baseline,gate=None):
    prompts=[r['question'] for r in records]
    if gate is None:
        blocked=None
        answers=generate_answers(model,tokenizer,prompts)
    else:
        # Metrics can reuse deterministic frozen-backbone responses. Reported
        # latency below always executes the full, uncached response path.
        blocked=[gate(q) for q in prompts]
        answers=["That eludes my understanding." if b else a for b,a in zip(blocked,baseline)]
    result={'requests':stage,
            'forget':response_statistics(records[:stage],answers[:stage],baseline[:stage],
                                         blocked[:stage] if blocked is not None else None),
            'retain':response_statistics(records[stage:],answers[stage:],baseline[stage:],
                                         blocked[stage:] if blocked is not None else None),
            'latency':measure_latency(model,tokenizer,records[:min(10,stage)]+records[stage:stage+10],gate)}
    return result,[{'index':r['index'],'answer':a,'gold':r['answer'],
                    'gate_blocked':bool(blocked[i]) if blocked is not None else None}
                   for i,(r,a) in enumerate(zip(records,answers))]
