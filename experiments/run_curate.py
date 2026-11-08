"""Full CURaTE + unchanged-model controls on the shared 300-request PCH stream."""
import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from experiments.behavioral import generate_answers, response_statistics, measure_latency, stage_responses
from experiments.qwen35_common import load_model,load_tokenizer,pch_records,manifest,seed_all,write_json


class CurateGate:
    def __init__(self,embedder):
        self.embedder=embedder
        self.database=None

    @torch.no_grad()
    def register(self,text):
        vector=self.embedder.encode([text],convert_to_tensor=True,normalize_embeddings=True)
        self.database=vector if self.database is None else torch.cat([self.database,vector])

    @torch.no_grad()
    def __call__(self,query):
        vector=self.embedder.encode([query],convert_to_tensor=True,normalize_embeddings=True)
        score=(vector@self.database.T).max().item()
        return score>.9


def main():
    wall_start=time.perf_counter()
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='FITPCH/Llama-3-8B_PCH_finetune')
    parser.add_argument('--embedder',default='Model/CURaTE-mpnet-NQ18K-seed42')
    parser.add_argument('--output',default='Experiment_record/Llama-3-8B-PCH/pch_v2/CURaTE')
    parser.add_argument('--max_requests',type=int,default=300)
    parser.add_argument('--eval_limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(8)
    seed_all(42)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    records=pch_records(42)
    write_json(out/'config.json',{**vars(args),**manifest(records),'protocol':'pch-v2',
        'pilot':bool(args.eval_limit or args.max_requests!=300),'gate_threshold':.9,
        'request_representation':'raw PCH request text, not evaluation QA',
        'refusal_proxy':'new refusals among initially non-refused queries; not human safety annotations',
        'metric_generation':'frozen deterministic backbone answers cached for metrics only; latency is uncached'})
    start=time.perf_counter()
    model=load_model(args.model,'cuda:0').eval()
    tok=load_tokenizer(args.model)
    embedder=SentenceTransformer(args.embedder,device='cuda:0')
    gate=CurateGate(embedder)
    torch.cuda.synchronize()
    loaded=time.perf_counter()-start
    evaluation_records=records if not args.eval_limit else records[:args.eval_limit]+records[300:300+args.eval_limit]
    baseline=generate_answers(model,tok,[r['question'] for r in evaluation_records])
    base_map={r['index']:a for r,a in zip(evaluation_records,baseline)}
    write_json(out/'baseline_responses.json',{'records':[{'index':i,'answer':a} for i,a in base_map.items()],
                                            'dataset_sha256':manifest(records)['dataset_sha256']})
    results,base_results,processing=[],[],[]
    for index,row in enumerate(records[:args.max_requests]):
        torch.cuda.synchronize()
        start=time.perf_counter()
        gate.register(row['text'])
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        processing.append({'request':index+1,'pch_index':row['index'],'seconds':elapsed,
                           'database_entries':index+1})
        with (out/'requests.jsonl').open('a') as f: f.write(json.dumps(processing[-1])+'\n')
        if (index+1)%60==0 or index+1==args.max_requests:
            stage=index+1
            if args.eval_limit:
                selected=records[:min(stage,args.eval_limit)]+records[stage:stage+args.eval_limit]
                missing=[r for r in selected if r['index'] not in base_map]
                new=generate_answers(model,tok,[r['question'] for r in missing])
                base_map.update({r['index']:a for r,a in zip(missing,new)})
                boundary=min(stage,args.eval_limit)
            else:
                selected,boundary=records,stage
            base=[base_map[r['index']] for r in selected]
            result,details=stage_responses(model,tok,selected,boundary,base,gate)
            result['requests']=stage
            results.append(result)
            bresult={'requests':stage,'forget':response_statistics(selected[:boundary],base[:boundary]),
                     'retain':response_statistics(selected[boundary:],base[boundary:]),
                     'latency':measure_latency(model,tok,selected[:min(10,boundary)]+selected[boundary:boundary+10])}
            base_results.append(bresult)
            write_json(out/'evaluations.json',results)
            write_json(out/'initial_control_evaluations.json',base_results)
            write_json(out/f'responses_stage{stage}.json',details)
            print('CURaTE completed stage',stage,flush=True)
    write_json(out/'completed.json',{'pilot':bool(args.eval_limit or args.max_requests!=300),
        'requests':args.max_requests,'load_seconds':loaded,'online_seconds':sum(r['seconds'] for r in processing),
        'wall_seconds_main':time.perf_counter()-wall_start,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})


if __name__=='__main__': main()
