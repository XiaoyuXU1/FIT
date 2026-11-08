# Continual unlearning experiments

These entry points perform unlearning and evaluation using supplied checkpoints.
Initial-model fine-tuning, reference-checkpoint retraining, and CURaTE embedder
training are deliberately outside this code release.

## Shared data and protocol

`pch_records(42)` aligns the public `FITPCH/PCH` text with `FITPCH/PCH_QA`.
The first 300 records are shuffled with seed 42 and requested sequentially;
the remaining unrequested records form each stage's retain set. Every run records
the ordering and dataset SHA256.

The modern `rebuttal-v2` protocol uses answer-only ROUGE-L, 128 new greedy tokens,
and `KL(reference || current)` for retention regularization. The generated-token
batch size is 64 for the current runner. Historical Llama intrinsic evaluations
used batch size 4; behavioral/probe generation uses batch size 8. These historical
runs are not reproduced simply by selecting Llama with the current defaults.
Raw probability, ROUGE-L, accuracy, and quality are retained.

## FIT and fixed-method controls

```bash
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/initial --method FIT --output Experiment_record/FIT
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/initial --method NPO_KL --output Experiment_record/NPO_KL
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/initial --method NPO_KL --filter \
  --output Experiment_record/NPO_KL_filtered
```

Use `RL` for RLabel and repeat with/without `--filter`. Fixed-method controls
update all parameters and do not use layer attribution or importance routing.
FIT always filters, selects decoder layers, and routes requests among RLabel,
NPO, and NPO+KL. The runner supports both full and linear attention in Qwen3.5.

ALKN is invoked with `--method ALKN`; O³ with `--method O3`, after running
`python scripts/fetch_o3_source.py`. Implementation adaptations and provenance
are in [baseline notes](../docs/BASELINES.md) and each run's `config.json`.

## FD/RU from existing reference checkpoints

```bash
bash scripts/python_qwen35.sh -m experiments.reference_metrics \
  --initial /path/to/initial --retain /path/to/retain_reference \
  --output Experiment_record/reference.json
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/initial --method FIT --output Experiment_record/FIT_with_FD_RU \
  --reference_metrics Experiment_record/reference.json
```

The reference command **only evaluates**, and never trains either checkpoint.
Without `--reference_metrics`, the runner reports raw quality metrics without
FD/RU. Quality is `100 * (Probability * ROUGE-L * Accuracy)^(1/3)`.
The recorded FD/RU design interpolates between initial and retain endpoints:
`max(0, 1 - abs(current_quality / reference_quality - 1))`.
This is an endpoint approximation, not a knowledge-deletion percentage. Even an
unchanged model can score highly. Intermediate explicit retraining references
are not included in this release. Never reuse references for a different
checkpoint, data ordering, or evaluator.

## Efficiency and common online-time budget

`requests.jsonl` records synchronized request and module times, selected layers,
importance, update counts, filtered characters, and allocated/reserved GPU memory.
`completed.json` separates loading, warmup, online work, and evaluation. GPU peaks
are PyTorch allocations rather than whole-device use; CPU peaks are main-process
RSS. Setup of supplied checkpoints/embedders is external and must be reported
separately for a full-system cost comparison.

```bash
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/initial --method FIT --budget_seconds 1339.0139914250467 \
  --output Experiment_record/budget_FIT
```

Use the same budget for all compared methods. The example value is the measured
budget of the original supplementary run; recompute a budget for a new experiment.
After exhaustion, all 300 requests still enter the evaluation sets but later
requests receive no updates. Checks occur between operations, so inspect the
reported overshoot and update coverage. Use matched hardware and account for
shared CPU/GPU load. Single-seed independent runs do not guarantee bitwise CUDA
reproducibility, and their differences cannot all be attributed to the budget.

## CURaTE and shared knowledge probes

CURaTE consumes an existing embedder trained using the author's recipe. A raw
pretrained MPNet model is not a substitute. The LLM stays frozen and the retrieval
threshold is fixed at 0.9; only current request paragraphs enter the database.

Generate a common probe bank first, using a supplied paraphrasing checkpoint:

```bash
bash scripts/python_qwen35.sh -m experiments.build_probes \
  --model /path/to/paraphrasing_model --output Experiment_record/probe_bank.json
bash scripts/python_qwen35.sh -m experiments.run_curate \
  --model FITPCH/Llama-3-8B_PCH_finetune \
  --embedder /path/to/trained_CURaTE_embedder \
  --probe_bank Experiment_record/probe_bank.json --output Experiment_record/CURaTE
bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model FITPCH/Llama-3-8B_PCH_finetune --method FIT \
  --behavioral_baseline Experiment_record/CURaTE/baseline_responses.json \
  --probe_bank Experiment_record/probe_bank.json --output Experiment_record/FIT_P1
```

Both methods use the same questions, targets, and probe bank. The five prompt
variants include original, instruction/completion, and paraphrase forms.
A target is extracted if its normalized full answer appears at word/number
boundaries or token F1 is at least 0.8. The refusal measure is an operational
proxy relative to initially non-refused queries, not a human safety annotation.
Automatically generated paraphrases are not human-reviewed. Latency uses actual
uncached responses and reports generated-token counts and answered/refused groups.

## Validation and outputs

```bash
bash scripts/python_qwen35.sh -m unittest experiments.test_rebuttal -v
```

For a real-model pilot, add `--max_requests 2 --eval_every 2 --eval_limit 2`.
It is explicitly marked as a pilot and never counted as a complete 300-request
experiment. Outputs are not bundled with the code. The historical supplementary
aggregator is `python -m experiments.summarize`; it expects the directory names
specified at the top of that module, and is not a generic experiment registry.
