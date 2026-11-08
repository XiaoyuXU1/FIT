"""PCH continual FIT / fixed primitive experiments with synchronized timing.

Protocol v2 deliberately records raw metrics and corrected KL/ROUGE definitions.
Results must not be silently combined with legacy evaluation JSONs.
"""
import argparse
import gc
import json
import math
import os
import resource
import time
from importlib.metadata import version
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from Fit.filtering import Unlearn_request_memory
from experiments.filtering import BatchedFiltering
from experiments.qwen35_common import (
    decoder, encode, load_model, load_tokenizer, manifest, pch_records, seed_all, write_json, attention_backend,
)
from experiments.rebuttal_metrics import GENERATION_BATCH_SIZE, evaluate, relative_quality


@contextmanager
def timed(timings, name, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def select_layers(model, tokenizer, text, ratio):
    layers = decoder(model).layers
    inputs = encode(tokenizer, text, model.device, 128)
    model.eval()
    with torch.no_grad():
        original = model(**inputs, labels=inputs.input_ids).loss.item()
        scores = []
        for index, layer in enumerate(layers):
            saved = [p.detach().clone() for p in layer.parameters()]
            try:
                for p in layer.parameters():
                    p.zero_()
                masked = model(**inputs, labels=inputs.input_ids).loss.item()
                scores.append((index, abs(masked-original)))
            finally:
                for p, old in zip(layer.parameters(), saved):
                    p.copy_(old)
            del saved
    return [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)
            [:max(1, math.ceil(len(layers)*ratio))]]


def importance(model, tokenizer, text):
    inputs = encode(tokenizer, text, model.device, 4096)
    embeddings = model.get_input_embeddings()(inputs.input_ids).detach().requires_grad_(True)
    model.eval()
    loss = model(inputs_embeds=embeddings, attention_mask=inputs.attention_mask,
                 labels=inputs.input_ids, use_cache=False).loss
    return torch.autograd.grad(loss, embeddings)[0].norm().item()


def train_request(model, reference, tokenizer, text, retain, algorithm, args, deadline):
    chunks = [text[i:i+args.chunk_size] for i in range(0, len(text), args.chunk_size)]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.1, foreach=False)
    total = max(1, args.epochs*len(chunks))
    warmup = int(0.1*total)
    def scale(step):
        if step < warmup:
            return step/max(1, warmup)
        return 0.1 + 0.45*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    updates = 0
    model.train()
    for _ in range(args.epochs):
        for index, text_chunk in enumerate(chunks):
            if time.perf_counter() >= deadline:
                model.zero_grad(set_to_none=True)
                return updates
            inputs = encode(tokenizer, text_chunk, model.device, args.chunk_size)
            if inputs is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            if algorithm == "RL":
                labels = torch.randint(model.config.vocab_size, inputs.input_ids.shape,
                                       device=model.device)
                loss = model(**inputs, labels=labels).loss
                loss.backward()
            else:
                # Sequence-summed NPO log ratio, matching the existing beta=0.1.
                with torch.no_grad():
                    oracle = reference(**inputs, labels=inputs.input_ids).loss
                current = model(**inputs, labels=inputs.input_ids).loss
                ratio = (current-oracle)*(inputs.input_ids.shape[1]-1)
                loss = -F.logsigmoid(0.1*ratio)*20
                loss.backward()
                if algorithm == "NPO_KL":
                    retained = retain[index % len(retain)]["text"]
                    # Fixed character window per forget chunk; retain membership
                    # remains identical in paired filter-on/filter-off runs.
                    retained = retained[:args.chunk_size]
                    rinputs = encode(tokenizer, retained, model.device, args.chunk_size)
                    with torch.no_grad():
                        ref_logits = reference(**rinputs).logits[:, :-1].float()
                        target = ref_logits.softmax(-1)
                    logits = model(**rinputs).logits[:, :-1].float()
                    kl = F.kl_div(logits.log_softmax(-1), target,
                                  reduction="sum") / target.shape[1]
                    kl.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.step()
            scheduler.step()
            updates += 1
    model.zero_grad(set_to_none=True)
    return updates


def main():
    wall_start=time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["FIT", "NPO_KL", "RL", "ALKN", "O3"], required=True)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_requests", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=60)
    parser.add_argument("--eval_limit", type=int, default=0,
                        help="Smoke evaluation only; nonzero marks run as a pilot")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--topk", type=float, default=0.25)
    parser.add_argument("--imp_low", type=float, default=0.5)
    parser.add_argument("--imp_high", type=float, default=1.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--budget_seconds", type=float, default=0,
                        help="Online compute budget; all 300 requests still enter evaluation sets")
    parser.add_argument("--reference_metrics",
                        help="JSON with initial/retain forget/retain raw quality for legacy interpolation")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--behavioral_baseline",help="Frozen-backbone responses from the CURaTE control run")
    parser.add_argument("--probe_bank",help="Run the shared paraphrase/prompt knowledge probes after request 300")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not (1 <= args.max_requests <= 300 and args.eval_every >= 1
            and args.epochs >= 1 and args.chunk_size >= 2 and 0 < args.topk <= 1
            and args.imp_low < args.imp_high and args.budget_seconds >= 0):
        parser.error("Invalid request, update, layer, importance or budget settings")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    records = pch_records(args.seed)
    if args.probe_bank:
        bank=json.loads(Path(args.probe_bank).read_text())
        if not bank.get('complete') or bank['dataset_sha256']!=manifest(records)['dataset_sha256']:
            raise ValueError('Knowledge probe bank is incomplete or uses different PCH data')
    metadata = {**vars(args), **manifest(records), "protocol": "rebuttal-v2",
                "filter_enabled":bool(args.filter or args.method=='FIT'),
                "layer_attribution_enabled":args.method=='FIT',
                "importance_routing_enabled":args.method=='FIT',
                "filter_chunk_size_characters":8,
                "request_seed_policy":"seed + zero-based request position; independent of auxiliary initialization",
                "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
                "gpu": torch.cuda.get_device_name(args.device),
                "visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
                "cuda_allocator":os.environ.get('PYTORCH_CUDA_ALLOC_CONF','native defaults'),
                "attention_backend": "recorded_after_model_load",
                "filter_backend": "batched embeddings/cosine and deletion losses; unchanged filtering rule",
                "kl": "KL(reference || current), mean over causal positions",
                "generation": "raw question, greedy, answer-only ROUGE, 128 new tokens",
                "metric_generation_batch_size":GENERATION_BATCH_SIZE,
                "pilot": bool(args.eval_limit or args.max_requests != 300)}
    if args.method == 'ALKN':
        metadata['baseline_implementation']={'type':'paper_reimplementation','source':'https://arxiv.org/html/2611.02620v1',
            'lambda':.8,'s':10,'low_lr_ratio':.1,'end_active_ratio':.3,
            'adaptations':['one fixed retain minibatch per request','global quantile interpolation from a fixed uniform 131072-element sample',
                           'CPU offload of mask/retain-gradient/task-start weights',
                           'persistent CPU snapshot/gradient/mask buffers and same-dtype D2H staging; unchanged FP32 calculations'],
            'official_source_status':'linked repository returns 404'}
    if args.method == 'O3':
        metadata['baseline_implementation']={'type':'official_detector_and_architecture_port',
            'source_revision':'839afea7fc09001433ef30bd6670a6d6804a9045',
            'lora_rank':8,'lora_alpha':16,'lora_lr':3e-4,'lora_epochs':15,'lora_weight_decay':0.,'max_grad_norm':1.,
            'lora_scheduler':'linear; warmup=min(100,floor(0.1*request_steps))',
            'inference_gate_profile':'released ScienceQA eval_o3.py piecewise remapping',
            'covariance':'sklearn EmpiricalCovariance, same CPU centering as official estimator','orthogonal_weight':.1,
            'adapter_recurrence':'resume latest A/B; previous A snapshots for orthogonality; inference uses latest BA only',
            'orthogonality_dtype':'float32',
            'detector':'RoBERTa-base with official MLM+contrastive-entropy loss',
            'metric_gate_input':'associated PCH question for probability/text accuracy; prompt question for generation; no gold answer',
            'adaptations':['all decoder linear projections including Qwen linear attention',
                           '128-character windows / 64-character stride from current paragraph for OOD training',
                           '64-character / 32-stride fallback for paragraphs with fewer than four long windows',
                           'random token targets for raw-text PCH unlearning',
                           '80/20 in-request OOD train/calibration split','maximum gate across request detectors',
                           'historical B snapshots and dormant FP32 calibration precision matrices stored on CPU; latest adapter remains on GPU',
                           'batched cached gates; latest low-rank adapter forward only; per-layer orthogonality backward to bound memory; warmup scaled to short requests; microbatch 1']}
    write_json(out / "config.json", metadata)
    timers = {}
    with timed(timers, "load", args.device):
        model = load_model(args.model, args.device)
        tokenizer = load_tokenizer(args.model)
        reference = None
        if args.method in ("FIT", "NPO_KL"):
            reference = load_model(args.model, args.device).eval().requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if 'linear_chunk' in attention_backend(model):
            with timed(timers,'warmup',args.device):
                with torch.random.fork_rng(devices=[model.device.index]):
                    ids=torch.full((1,64),tokenizer.eos_token_id or 0,dtype=torch.long,device=model.device)
                    model.train()
                    model(input_ids=ids,labels=ids,use_cache=False).loss.backward()
                    model.zero_grad(set_to_none=True)
                    model.eval()
                    with torch.no_grad():
                        model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,
                                       min_new_tokens=2,do_sample=False,use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id)
        memory = Unlearn_request_memory()
        filtering = None
        if args.filter or args.method == "FIT":
            filtering = BatchedFiltering(similarity_threshold=args.similarity_threshold,
                                       chunk_size=8, device=args.device,
                                       epsilon=args.epsilon, language_model=model,
                                       language_tokenizer=tokenizer)
        special = None
        if args.method == 'ALKN':
            from experiments.alkn import ALKN
            special = ALKN()
        elif args.method == 'O3':
            from experiments.o3 import O3
            special = O3(model,args.device)
    timers['load']-=timers.get('warmup',0.)
    metadata['loaded_model_commit']=getattr(model.config,'_commit_hash',None)
    metadata['attention_backend']=attention_backend(model)
    if 'linear_chunk' in metadata['attention_backend']:
        metadata['versions'].update({n:version(n) for n in ['fla-core','flash-linear-attention','causal-conv1d','triton']})
        metadata['warmup']='synthetic 64-token forward/backward/decode, RNG restored; excluded from online budget'
    write_json(out/'config.json',metadata)
    references = json.loads(Path(args.reference_metrics).read_text()) if args.reference_metrics else None
    if references and (references.get("dataset_sha256") != metadata["dataset_sha256"]
                       or references.get("protocol") != metadata["protocol"]):
        raise ValueError("Reference metrics have a different dataset ordering or metric protocol")
    if references and references.get('metric_generation_batch_size',GENERATION_BATCH_SIZE)!=GENERATION_BATCH_SIZE:
        raise ValueError('Reference generation batch size differs from this trial')
    baseline_answers = None
    if args.behavioral_baseline:
        baseline_data=json.loads(Path(args.behavioral_baseline).read_text())
        if baseline_data['dataset_sha256']!=metadata['dataset_sha256']:
            raise ValueError('Behavioral baseline uses a different request order')
        by_index={r['index']:r['answer'] for r in baseline_data['records']}
        baseline_answers=[by_index[r['index']] for r in records]
    stages = []
    online = 0.0
    peak_online = 0.0
    peak_evaluation = 0.0
    torch.cuda.reset_peak_memory_stats(args.device)
    for index in range(args.max_requests):
        row = records[index]
        trace = {"request": index+1, "pch_index": row["index"], "updates": 0,
                 "algorithm": args.method, "input_chars": len(row["text"])}
        timings = {}
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        request_seed=args.seed+index
        seed_all(request_seed)
        trace['request_seed']=request_seed
        deadline = start + max(0, args.budget_seconds-online) if args.budget_seconds else math.inf
        text = row["text"]
        embeddings = []
        if time.perf_counter() < deadline:
            if filtering is not None:
                model.eval()
                with timed(timings, "filter", args.device):
                    text, deleted, embeddings = filtering.filter_data(
                        text, [r["data"] for r in memory.retrieve_unlearning_data()])
                trace["deleted_chars"] = len(deleted)
            eligible = bool(text) and encode(tokenizer,text,args.device,128) is not None
            if text and not eligible:
                trace['skip_reason']='fewer_than_two_tokens_after_filtering'
            if eligible and time.perf_counter() < deadline:
                if args.method in ('ALKN','O3'):
                    with timed(timings, 'update', args.device):
                        if args.method == 'ALKN':
                            trace['updates'] = special.run(model,tokenizer,text,records[index+1:],args,deadline)
                        else:
                            trace['updates'] = special.run(tokenizer,text,args,deadline)
                elif args.method == "FIT":
                    with timed(timings, "attribution", args.device):
                        layers = select_layers(model, tokenizer, text, args.topk)
                    model.requires_grad_(False)
                    for i in layers:
                        # Include both linear and full attention in Qwen3.5.
                        for name, p in decoder(model).layers[i].named_parameters():
                            p.requires_grad_(name.startswith(("mlp.", "self_attn.", "linear_attn.")))
                    trace["layers"] = layers
                    with timed(timings, "importance", args.device):
                        score = importance(model, tokenizer, text)
                    trace["importance"] = score
                    trace["algorithm"] = "NPO_KL" if score >= args.imp_high else "NPO" if score >= args.imp_low else "RL"
                else:
                    model.requires_grad_(True)
                if args.method not in ('ALKN','O3'):
                    with timed(timings, "update", args.device):
                        trace["updates"] = train_request(model, reference, tokenizer, text,
                                                         records[index+1:], trace["algorithm"], args, deadline)
                if trace["updates"]:
                    memory.store_unlearning_data(index, embeddings)
        trace["kept_chars"] = len(text)
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter()-start
        online += elapsed
        trace.update(seconds=elapsed, modules=timings, cumulative_online_seconds=online,
                     peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30,
                     budget_exhausted=bool(args.budget_seconds and online >= args.budget_seconds))
        peak_online = max(peak_online, trace["peak_allocated_gib"])
        with (out / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(trace) + "\n")
        print(json.dumps(trace), flush=True)
        if (index+1) % args.eval_every == 0 or index+1 == args.max_requests:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            forget, retain = records[:index+1], records[index+1:]
            if args.eval_limit:
                forget, retain = forget[:args.eval_limit], retain[:args.eval_limit]
            torch.cuda.reset_peak_memory_stats(args.device)
            eval_batch = GENERATION_BATCH_SIZE
            with timed(timers, "evaluation", args.device):
                stage = {"requests": index+1, "forget": evaluate(model, tokenizer, forget, batch_size=eval_batch),
                         "retain": evaluate(model, tokenizer, retain, batch_size=eval_batch), "online_seconds": online}
                if baseline_answers is not None and not args.eval_limit:
                    from experiments.behavioral import stage_responses
                    behavior, details=stage_responses(model,tokenizer,records,index+1,baseline_answers)
                    stage['behavioral']=behavior
                    write_json(out/f'responses_stage{index+1}.json',details)
            peak_evaluation = max(peak_evaluation, torch.cuda.max_memory_allocated(args.device)/2**30)
            if references:
                fraction = (index+1)/300
                for label, key in (("FD", "forget"), ("RU", "retain")):
                    a, b = references["initial"][key]["quality"], references["retain"][key]["quality"]
                    stage[label] = relative_quality(stage[key]["quality"], a+(b-a)*fraction)
            stages.append(stage)
            write_json(out / "evaluations.json", stages)
    if args.probe_bank and args.max_requests==300 and not args.eval_limit:
        from experiments.behavioral import knowledge_probes
        with timed(timers,'knowledge_probes',args.device):
            probes=knowledge_probes(model,tokenizer,args.probe_bank,[r['index'] for r in records[:300]])
        write_json(out/'knowledge_probes.json',probes)
    if args.save_model:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    write_json(out / "completed.json", {"timings": timers, "online_seconds": online,
               "wall_seconds_main":time.perf_counter()-wall_start,
               "total_compute_seconds": online + sum(timers.values()),
               "budget_overshoot_seconds": max(0, online-args.budget_seconds) if args.budget_seconds else 0,
               "peak_online_allocated_gib": peak_online,
               "peak_evaluation_allocated_gib": peak_evaluation,
               "peak_cpu_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
               "pilot": metadata["pilot"]})


if __name__ == "__main__":
    main()
