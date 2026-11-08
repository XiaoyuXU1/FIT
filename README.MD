# FIT: Continual LLM Unlearning

Main experiments for FIT and baseline methods on the PCH benchmark.
All methods start from existing checkpoints and process sequential unlearning
requests. Checkpoint preparation and model weights are supplied separately.

FIT combines request filtering, layer attribution, and importance-based routing
between RLabel, NPO, and NPO+KL updates.

## Code

| Path | Purpose |
| --- | --- |
| `FIT.py`, `Fit/` | Original FIT entry point and method components |
| `Baselines.py`, `Unlearning_algorithm/` | Original fixed-method baseline implementations |
| `experiments/run_unlearning.py` | PCH comparison runner: FIT, NPO+KL, RLabel, ALKN, O³ |
| `experiments/run_curate.py` | CURaTE baseline with a frozen LLM and trained retrieval embedder |
| `experiments/metrics.py`, `Evaluations/` | Current and original evaluation functions |
| `scripts/run_unlearning.sh` | Sequential main-method launcher |

See [running instructions](experiments/README.md) and
[baseline implementation notes](docs/BASELINES.md).
The class name `FineTuning` in the original code denotes unlearning updates.

## Environment

Use Python 3.10 or later and install a CUDA-compatible PyTorch 2.6.0 build, then:

```bash
python -m pip install -r requirements-qwen35.txt
python -m unittest experiments.test_unlearning -v
```

Run commands from the repository root. `scripts/python_qwen35.sh` uses `python`
from the active environment; set `PYTHON` to choose another interpreter.
Optional native Qwen kernels are pinned in `requirements-qwen35-kernels.txt`.
Install them after PyTorch with `--no-build-isolation`, using a compatible CUDA
toolkit and compiler. The recorded convolution build used CUDA 12.6 and PyTorch
2.6.0+cu124; build against your installed PyTorch ABI. Runtime metadata records
the actual attention backend, including any fallback.

## Main experiment

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python_qwen35.sh -m experiments.run_unlearning \
  --model /path/to/initial_checkpoint \
  --method FIT --output Experiment_record/FIT
```

Choose `NPO_KL`, `RL`, `ALKN`, or `O3` for a fixed baseline. Before using O³:

```bash
python scripts/fetch_o3_source.py
```

To run FIT and all four fixed baselines sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_unlearning.sh \
  /path/to/initial_checkpoint Experiment_record/comparison
```

Full runs process 300 requests and evaluate every 60 requests. Outputs include
raw quality metrics, method configurations, synchronized timing/memory traces,
and completion markers. Existing output directories are never overwritten.
For FD/RU, pass reference metrics evaluated from matching supplied checkpoints;
see the [experiment guide](experiments/README.md).

## Original entry points

The original scripts use the separate `requirements.txt` runtime: PyTorch 2.2.0,
CUDA 11.8, Transformers 4.46.2, and FlashAttention 2.5.3.

```bash
python FIT.py --unlearning_model_name FITPCH/Llama-3-8B_PCH_finetune \
  --retain_model_name FITPCH/Llama-3-8B_PCH_retain --device cuda:0
python Baselines.py --unlearning_algorithm NPO_KL \
  --unlearning_model_name FITPCH/Llama-3-8B_PCH_finetune \
  --retain_model_name FITPCH/Llama-3-8B_PCH_retain --device cuda:0
```

Original and current evaluators use different protocols; do not combine their
metrics. The older ALKN implementation is not equivalent to the PCH implementation
in `experiments/alkn.py`; details are documented in the baseline notes.
