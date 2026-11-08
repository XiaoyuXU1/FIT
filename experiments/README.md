# PCH main experiments

The main comparison evaluates FIT and baseline methods under the same sequential
PCH request stream, using existing initial and reference checkpoints.

## Data and evaluation

`pch_records(42)` aligns `FITPCH/PCH` with `FITPCH/PCH_QA`. It shuffles the first
300 records with seed 42 and requests them sequentially. Unrequested records form
the retain set. Each run saves the ordering and dataset SHA256.

The `pch-v2` protocol uses answer-only ROUGE-L, 128 new greedy tokens, and
`KL(reference || current)` for retention regularization. Intrinsic generation uses
batch size 64. Raw probability, ROUGE-L, accuracy, and quality are retained.
The original evaluators have different behavior and must be kept separate.
Reference metrics must use the same protocol version as the runner.

## FIT and fixed baselines

```bash
bash scripts/python_qwen35.sh -m experiments.run_unlearning \
  --model /path/to/initial --method FIT --output Experiment_record/FIT
bash scripts/python_qwen35.sh -m experiments.run_unlearning \
  --model /path/to/initial --method NPO_KL --output Experiment_record/NPO_KL
```

The other fixed methods are `RL`, `ALKN`, and `O3`. FIT always enables its request
filtering, layer attribution, and importance-based routing. Fixed methods do not
use those FIT components. For O³, first run `python scripts/fetch_o3_source.py`.
Baseline provenance and implementation adaptations are documented in
[baseline notes](../docs/BASELINES.md).

## FD/RU references

Evaluate supplied initial and retain-reference checkpoints without training:

```bash
bash scripts/python_qwen35.sh -m experiments.reference_metrics \
  --initial /path/to/initial --retain /path/to/retain_reference \
  --output Experiment_record/reference.json
bash scripts/python_qwen35.sh -m experiments.run_unlearning \
  --model /path/to/initial --method FIT --output Experiment_record/FIT \
  --reference_metrics Experiment_record/reference.json
```

Without reference metrics, the runner reports raw quality but not FD/RU.
Quality is `100 * (Probability * ROUGE-L * Accuracy)^(1/3)`. FD/RU uses
`max(0, 1 - abs(current_quality / reference_quality - 1))`, with reference quality
interpolated between initial and retain endpoints. This endpoint approximation
is not a knowledge-deletion percentage; even an unchanged model can score highly.
Do not reuse references for different checkpoints, ordering, or evaluation settings.

## CURaTE baseline

Supply an existing embedder trained with the author's recipe. A raw pretrained
MPNet model is not an equivalent substitute. The LLM remains frozen, retrieval
uses threshold 0.9, and only each current request paragraph enters the database.

```bash
bash scripts/python_qwen35.sh -m experiments.run_curate \
  --model FITPCH/Llama-3-8B_PCH_finetune \
  --embedder /path/to/trained_CURaTE_embedder --output Experiment_record/CURaTE
bash scripts/python_qwen35.sh -m experiments.run_unlearning \
  --model FITPCH/Llama-3-8B_PCH_finetune --method FIT \
  --behavioral_baseline Experiment_record/CURaTE/baseline_responses.json \
  --output Experiment_record/FIT_response_comparison
```

Both methods evaluate target-answer leakage, retain utility and coverage, refusal
proxies, and actual uncached response latency. Full-answer matching uses word/number
boundaries. The refusal measure is relative to initially non-refused queries, not
a human safety annotation. Latency includes generated-token counts and separates
answered/refused queries. Offline embedder preparation is external to these runs.

## Outputs and validation

Each full run writes `config.json`, `requests.jsonl`, `evaluations.json`, and
`completed.json`. Loading/warmup, online work, and evaluation are timed separately.
GPU memory is PyTorch allocated/reserved memory, and CPU memory is main-process
RSS. Interpret timing with the actual hardware and shared system load. Single-seed
CUDA runs are not guaranteed to be bitwise reproducible.

```bash
bash scripts/python_qwen35.sh -m unittest experiments.test_unlearning -v
```

For a short validation run, add `--max_requests 2 --eval_every 2 --eval_limit 2`.
Such runs are marked as pilots and are not complete 300-request experiments.
