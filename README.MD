# FIT: Continual LLM Unlearning

Main FIT and baseline experiments on the PCH benchmark. All methods consume
existing initial checkpoints; checkpoint training and model weights are external.

## Layout

```text
FIT.py                       FIT experiment entry point
Baselines.py                 Fixed-baseline experiment entry point
utils.py                     Shared model loading, data ordering, and JSON utilities
Fit/                         FIT pipeline, filtering, and layer attribution
Unlearning_algorithm/        Gradient primitives, ALKN, O³, and CURaTE
Evaluations/                 Quality/response metrics and reference evaluation
tests/                       Regression checks
requirements.txt             Single dependency list
```

There is one implementation of each main component. Model-specific loading,
including Qwen3.5's text backbone, is handled inside the shared loader.

## Setup

Use Python 3.10 or later. Install a CUDA-compatible PyTorch 2.6.0 build, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Optional Qwen native kernels are listed as comments in `requirements.txt`.
They require a compatible CUDA toolkit/compiler and must be installed after
PyTorch with `--no-build-isolation`. The recorded convolution build used CUDA
12.6 with PyTorch 2.6.0+cu124. Without these kernels, the loader uses a fallback;
actual backends and package versions are saved with each run.

## Run FIT

Run commands from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python FIT.py \
  --model /path/to/initial_checkpoint --output Experiment_record/FIT
```

FIT enables request filtering, layer attribution, and importance-based routing
between RLabel, NPO, and NPO+KL. Full runs process 300 sequential requests and
evaluate every 60 requests. No initial-model training is launched.

## Run baselines

```bash
CUDA_VISIBLE_DEVICES=0 python Baselines.py \
  --model /path/to/initial_checkpoint --method NPO_KL \
  --output Experiment_record/NPO_KL
```

Methods: `GA`, `GA_GD`, `GA_KL`, `NPO`, `NPO_KL`, `RL`, `ALKN`, `O3`, `CURATE`.
The gradient-ascent family preserves its original update functions; the other
gradient primitives use the shared request optimizer. GA-family methods do not
support online time budgets.

CURaTE requires a separately trained author-recipe embedder:

```bash
CUDA_VISIBLE_DEVICES=0 python Baselines.py \
  --model FITPCH/Llama-3-8B_PCH_finetune --method CURATE \
  --embedder /path/to/trained_CURaTE_embedder --output Experiment_record/CURATE
```

It keeps LLM weights frozen and evaluates its actual retrieval/refusal path.
For a response-level FIT comparison, pass
`--behavioral_baseline Experiment_record/CURaTE/baseline_responses.json` to FIT.

## Evaluation and outputs

PCH text and QA are aligned, the first 300 records use a seed-42 permutation,
and unrequested records form the retain set. Each run records a data hash and
request ordering. The `pch-v2` evaluator uses answer-only ROUGE-L, 128 new greedy
tokens, and `KL(reference || current)` for NPO+KL retention regularization.
Raw probability, ROUGE-L, accuracy, quality, timing, memory, and update counts
are saved separately. Output directories are never silently overwritten.

To obtain FD/RU references from supplied checkpoints, without training:

```bash
python -m Evaluations.reference \
  --initial /path/to/initial --retain /path/to/retain_reference \
  --output Experiment_record/reference.json
python FIT.py --model /path/to/initial --output Experiment_record/FIT \
  --reference_metrics Experiment_record/reference.json
```

Quality is `100 * (Probability * ROUGE-L * Accuracy)^(1/3)`. FD/RU is
`max(0, 1 - abs(current_quality / reference_quality - 1))`, with reference quality
interpolated between initial and retain endpoints. This is an approximation,
not a knowledge-deletion percentage; even an unchanged model can score highly.
References must match the checkpoints, ordering, and evaluator version.
Response-level leakage and refusal scores are lexical/operational proxies,
not human safety annotations. Interpret timings with hardware and shared load.

For a short validation run, add
`--max_requests 2 --eval_every 2 --eval_limit 2` to the main entry points.

## Baseline provenance

**ALKN:** [paper](https://arxiv.org/abs/2611.02620). The PCH paper-based
implementation includes task-vector negation. Its fixed retain minibatch,
sampled global quantiles, and CPU-offloaded state are explicit adaptations
that affect memory and runtime.

**O³:** [paper](https://arxiv.org/abs/2407.10223),
[author code](https://github.com/GCYZSL/O3-LLM-UNLEARNING), revision
`839afea7fc09001433ef30bd6670a6d6804a9045`. The implementation continues the latest
LoRA adapter and applies only that adapter at inference; previous A snapshots
contribute to orthogonality. On first use it fetches the pinned author detector
into `.cache/` and verifies its SHA256. The ScienceQA gate mapping is retained;
PCH paragraph windows, random-token targets, and hybrid-attention projections
are task/architecture adaptations, not the author's original benchmark.

**CURaTE:** [paper](https://aclanthology.org/2026.findings-acl.1102/),
[author code](https://github.com/bsu1313/CURaTE), revision
`5194055ab4dd2966931ebd3f1dc0d4775a7dd747`. Use the trained NQ-18K MPNet embedder
recipe: one epoch, batch size 16, learning rate 2e-5, 100 warmup steps, cosine
ContrastiveLoss margin 0.5. A raw pretrained embedder is not equivalent.
The threshold is fixed at 0.9 and only current request paragraphs enter the
forget database. Offline embedder preparation is external to the online runs.
