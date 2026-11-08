# FIT: Continual LLM Unlearning

Unlearning code for FIT, baseline methods, ablations, and PCH evaluation.
This release starts from **existing initial and reference checkpoints**. It
contains no initial-model fine-tuning, retain-checkpoint retraining, or relearning
scripts. Checkpoints, caches, logs, notebooks, and local experiment outputs are
not included.

FIT combines request filtering, layer attribution, and importance-based routing
between RLabel, NPO, and NPO+KL updates. The legacy class name `FineTuning` in
`Fit/algorithm.py` refers to these **unlearning updates**, not checkpoint preparation.

## Code map

| Path | Purpose |
| --- | --- |
| `Fit/` | Original FIT filtering, attribution, and adaptive unlearning updates |
| `Unlearning_algorithm/` | Legacy primitive and experimental baseline implementations |
| `FIT.py` / `Baselines.py` | Legacy FIT / fixed-baseline entry points |
| `FIT_wo_*.py` | Legacy filtering, selection, and routing ablations |
| `Evaluations/` | Legacy evaluation functions |
| `experiments/run_rebuttal.py` | Modern PCH runner: FIT, NPO+KL, RLabel, ALKN, O³ |
| `experiments/alkn.py` / `experiments/o3.py` | Supplementary baseline adaptations |
| `experiments/run_curate.py` | CURaTE retrieval/refusal baseline |
| `experiments/behavioral.py` | Answer leakage, refusal proxies, knowledge probes, latency |
| `experiments/reference_metrics.py` | Evaluate supplied checkpoints; does not train them |
| `experiments/build_probes.py` | Generate a shared paraphrase/prompt probe bank |
| `scripts/` | Portable unlearning launchers and pinned O³ source fetcher |

The modern and legacy evaluators use different protocols. Do not mix their
numbers. See [experiment instructions](experiments/README.md) and
[baseline implementation notes](docs/BASELINES.md).

## Modern runtime

Use Python 3.10 or later, a CUDA GPU, and a dedicated environment. Install a
CUDA-compatible build of PyTorch 2.6.0, then install the tested core requirements:

```bash
python -m pip install -r requirements-qwen35.txt
python -m unittest experiments.test_rebuttal -v
```

For the native Qwen3.5 kernels used in the supplementary runs, install
`requirements-qwen35-kernels.txt` **after** PyTorch is installed, with
`--no-build-isolation`. A matching CUDA toolkit and compiler are required.
The native convolution in the recorded runtime was built with CUDA 12.6 against
PyTorch 2.6.0+cu124; use a build matching your own PyTorch ABI. Without these
optional kernels Transformers uses its fallback; record that backend when
comparing timing. Modern scripts record the loaded backend and software versions.

`scripts/python_qwen35.sh` uses `python` from the active environment. Set `PYTHON`
to an interpreter path if needed. An existing local `.runtime/qwen35` overlay is
supported, but is not required or distributed.

## Run unlearning

A short validation run using an existing public PCH checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model FITPCH/Llama-3-8B_PCH_finetune \
  --method FIT --max_requests 2 --eval_every 2 --eval_limit 2 \
  --output Experiment_record/pilot/FIT
```

A full Qwen experiment consumes your already prepared Qwen3.5-9B-Base PCH
checkpoint. It does not create or fine-tune that checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_qwen35.sh -m experiments.run_rebuttal \
  --model /path/to/Qwen3.5-9B-Base-PCH/stage0 \
  --method FIT --output Experiment_record/qwen/FIT
```

Use `NPO_KL`, `RL`, `ALKN`, or `O3` for the other modern methods. For O³, fetch
the checksum-verified author detector first:

```bash
python scripts/fetch_o3_source.py
```

For fixed-algorithm filtering ablations, add `--filter` to `NPO_KL` or `RL`.
FIT always enables its filtering, attribution, and routing components.
To sequentially launch all five modern methods on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_unlearning.sh \
  /path/to/initial_checkpoint Experiment_record/comparison
```

Output directories are never silently overwritten. Full runs process 300
sequential requests and evaluate every 60 requests. They save configurations,
per-request timing/memory/update traces, evaluations, and completion markers.
See [the experiment guide](experiments/README.md) for FD/RU references,
equal-time budgets, CURaTE, and knowledge probes.

## Legacy entry points

The legacy runtime uses `requirements.txt`; it is separate from the modern
Transformers 5.3 runtime. The original setup used PyTorch 2.2.0, CUDA 11.8,
Transformers 4.46.2, and FlashAttention 2.5.3. Install compatible dependencies
before using these historical scripts.

```bash
python FIT.py --unlearning_model_name FITPCH/Llama-3-8B_PCH_finetune \
  --retain_model_name FITPCH/Llama-3-8B_PCH_retain --device cuda:0
python Baselines.py --unlearning_algorithm NPO_KL \
  --unlearning_model_name FITPCH/Llama-3-8B_PCH_finetune \
  --retain_model_name FITPCH/Llama-3-8B_PCH_retain --device cuda:0
```

Legacy ALKN and PISCES code is retained for traceability. The supplementary ALKN
implementation is `experiments/alkn.py`; these implementations are not
interchangeable. RULE/UNDIAL are experimental function-level implementations,
not additional methods in the modern runner.
