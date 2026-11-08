# Baseline implementations and provenance

| Method | Entry point | Status |
| --- | --- | --- |
| FIT | `experiments/run_rebuttal.py --method FIT` | Filtering, layer attribution, importance routing |
| NPO+KL | same runner, `--method NPO_KL` | Fixed primitive; optional `--filter` control |
| RLabel | same runner, `--method RL` | Fixed random-label primitive; optional filtering |
| ALKN | `experiments/alkn.py` via `--method ALKN` | Paper reimplementation adapted to PCH |
| O³ | `experiments/o3.py` via `--method O3` | Architecture/task port using the author detector |
| CURaTE | `experiments/run_curate.py` | Frozen LLM plus trained embedding retrieval/refusal |
| GA, GA+GD, GA+KL, NPO | `Baselines.py` and `Unlearning_algorithm/` | Legacy evaluation protocol |
| RULE, UNDIAL, PISCES | `Unlearning_algorithm/` | Legacy/experimental functions; not modern-runner methods |

## ALKN

[Paper](https://arxiv.org/abs/2611.02620). The linked author repository was
unavailable when this implementation was prepared. The supplementary port includes
ESL-style soft labels, mask-based updates and task-vector negation. It uses a
fixed retain minibatch per request, a fixed uniform 131072-element sample for
global robust quantiles, and CPU-offloaded snapshots, gradients, and masks.
These adaptations materially affect runtime/memory and must be disclosed.

`Unlearning_algorithm/ALKN.py` is an older implementation retained because the
legacy driver imports it. It omits task-vector negation and must not be used as
an equivalent implementation of the supplementary method.

## O³

[Paper](https://arxiv.org/abs/2407.10223) and
[author repository](https://github.com/GCYZSL/O3-LLM-UNLEARNING), revision
`839afea7fc09001433ef30bd6670a6d6804a9045`.

`python scripts/fetch_o3_source.py` fetches the exact author detector file into an
ignored cache and verifies SHA256
`08ed1cf87eb3f9d4d8737a2c6d5c6f5da27c94e4aae98d16ae248a7409ce65fd`.
Third-party source and weights are not vendored in this release.

The port resumes the latest A/B adapter on every request; earlier A snapshots
enter only the orthogonality penalty. Inference applies only the latest adapter,
with a distinct gate for each query. It uses the released ScienceQA inference
remapping and CPU `EmpiricalCovariance` estimator. The profile uses rank 8,
alpha 16, 15 epochs, zero AdamW weight decay, clipping at 1, and FP32 orthogonality.
PCH adaptations include paragraph windows, random-token targets, an in-request
80/20 OOD split, decoder linear projections for Qwen hybrid attention, microbatch
1, and warmup capped at 10% of the short request. This is not the author's
original ScienceQA/TOFU experiment, and results must be labeled accordingly.

## CURaTE

[Paper](https://aclanthology.org/2026.findings-acl.1102/) and
[author repository](https://github.com/bsu1313/CURaTE), revision
`5194055ab4dd2966931ebd3f1dc0d4775a7dd747`.

Supply an embedder prepared using the author's NQ 18K recipe: MPNet, one epoch,
batch size 16, learning rate 2e-5, 100 warmup steps, cosine ContrastiveLoss with
margin 0.5. Its training code and weights are outside this unlearning-only release.
The online threshold is fixed at 0.9. Current request paragraphs, not evaluation
QA, gold answers, future requests, or probe paraphrases, enter the database.
Both target leakage and retain refusal/utility are measured through the actual
gated response path. Offline embedder cost must be separated from online
insertion and uncached response latency.
