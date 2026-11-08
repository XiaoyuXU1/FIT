"""Generate question-only paraphrases with a cached instruction model."""
import argparse
import json
import re
from pathlib import Path

import torch

from experiments.qwen35_common import load_model, load_tokenizer, pch_records, manifest, seed_all, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='FITPCH/Llama-3-8B-Instruct_PCH_finetune')
    parser.add_argument('--output',default='Experiment_record/probe_bank.json')
    args=parser.parse_args()
    seed_all(42)
    records=pch_records(42)
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    model=load_model(args.model,'cuda:0').eval()
    tok=load_tokenizer(args.model)
    bank=[]
    for start in range(0,len(records),8):
        rows=records[start:start+8]
        messages=[[{'role':'user','content':
            'Rewrite the following question in different wording while keeping exactly the same meaning. '
            'Preserve all names, numbers, quoted phrases and factual assumptions. '
            'Return only one rewritten question, without answering it.\n\n'+r['question']}]
            for r in rows]
        prompts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        inputs=tok(prompts,padding=True,return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True,
                                   pad_token_id=tok.pad_token_id)
        texts=tok.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for row,text in zip(rows,texts):
            candidate=text.strip().strip('"')
            candidate=re.sub(r'^(?:Rewritten question|Paraphrase|Question)\s*:\s*','',candidate,flags=re.I)
            numbers=set(re.findall(r'\d+(?:\.\d+)?',row['question']))
            valid=(bool(candidate) and len(candidate)<max(800,4*len(row['question']))
                   and candidate.endswith('?') and numbers<=set(re.findall(r'\d+(?:\.\d+)?',candidate)))
            bank.append({'index':row['index'],'question':row['question'],'answer':row['answer'],
                         'paraphrase':candidate,'automatic_format_check':valid,
                         'semantic_review':'not_human_reviewed'})
        write_json(out.with_suffix('.partial.json'),{'records':bank,'complete':False})
        print(f'paraphrases {len(bank)}/{len(records)}',flush=True)
    write_json(out,{**manifest(records),'generator':args.model,'complete':True,'records':bank,
                    'valid_format':sum(r['automatic_format_check'] for r in bank),
                    'limitation':'Automatically generated paraphrases; entity/semantic equivalence needs review.'})


if __name__=='__main__': main()
