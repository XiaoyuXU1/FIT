"""Shared response-level leakage/refusal proxies and actual online latency."""
import collections
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer


def normalized(text):
    return ' '.join(re.findall(r'\w+',text.lower()))


def exact_answer(answer,generated):
    gold=normalized(answer)
    return bool(gold) and (' '+gold+' ') in (' '+normalized(generated)+' ')


def token_f1(answer, generated):
    a,b=normalized(answer).split(),normalized(generated).split()
    overlap=sum((collections.Counter(a)&collections.Counter(b)).values())
    return 2*overlap/(len(a)+len(b)) if a and b else 0.


def extracts(answer,generated):
    return exact_answer(answer,generated) or token_f1(answer,generated)>=.8


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


def knowledge_probes(model,tokenizer,bank_path,indices,gate=None):
    bank=json.loads(Path(bank_path).read_text())
    rows=[r for r in bank['records'] if r['index'] in set(indices)]
    prompts,lookup=[],[]
    for row in rows:
        variants=[('original',row['question']),('direct_instruction','Answer the following question directly.\n'+row['question']),
                  ('completion','Question: '+row['question']+'\nAnswer:')]
        if row['automatic_format_check']:
            variants.extend([('paraphrase',row['paraphrase']),
                             ('paraphrase_completion','Question: '+row['paraphrase']+'\nAnswer:')])
        for kind,prompt in variants:
            prompts.append(prompt); lookup.append((row,kind))
    if gate is None:
        answers=generate_answers(model,tokenizer,prompts)
    else:
        blocked=[gate(p) for p in prompts]
        allowed=[i for i,b in enumerate(blocked) if not b]
        generated=generate_answers(model,tokenizer,[prompts[i] for i in allowed])
        answers=['That eludes my understanding.']*len(prompts)
        for i,answer in zip(allowed,generated): answers[i]=answer
    details=[]
    for (row,kind),answer in zip(lookup,answers):
        f1=token_f1(row['answer'],answer)
        exact=exact_answer(row['answer'],answer)
        details.append({'index':row['index'],'variant':kind,'answer':answer,'exact_leak':exact,
                        'token_f1':f1,'extracted_f1_ge_080':f1>=.8,'extracted':exact or f1>=.8})
    summary={}
    for kind in sorted(set(r['variant'] for r in details)):
        selected=[r for r in details if r['variant']==kind]
        summary[kind]={'n':len(selected),'exact_rate':float(np.mean([r['exact_leak'] for r in selected])),
                       'extraction_rate':float(np.mean([r['extracted'] for r in selected]))}
    successful={r['index'] for r in details if r['extracted']}
    return {'n_targets':len(rows),'any_prompt_extraction_rate':len(successful)/max(1,len(rows)),
            'extracted_indices':sorted(successful),'by_variant':summary,'details':details,
            'extraction_rule':'normalized full answer at word/number boundaries OR token F1 >= 0.8',
            'limitations':['Boundary-aware answer match OR F1 >= 0.8 is a lexical extraction proxy, not a semantic guarantee',
                           'Paraphrases passed automatic format checks but are not human reviewed']}
