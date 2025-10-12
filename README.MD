# FIT: Scalable and Continual Unlearning for Large Language Models

This repository provides the official implementation of **FIT: Scalable and Continual Unlearning for Large Language Models**.

We introduce **FIT**, a framework designed to address the challenges of unlearning in large language models (LLMs) through:

* **F**iltering: rigorous data filtering to ensure high-quality forget sets,
* **I**mportance-aware updates: adaptive strategies guided by sample importance, and
* **T**argeted layer attribution: precise identification of layers most responsible for memorization.

Together, these components enable effective forgetting without compromising model utility.

To support realistic evaluation, we propose the **PCH benchmark**, which encompasses diverse deletion scenarios:

* **P**ersonal information,
* **C**opyrighted material, and
* **H**armful content.

We also introduce comprehensive metrics to jointly measure forgetting effectiveness and utility retention.
Across multiple LLMs, FIT consistently outperforms strong baselines, achieving better forgetting–utility trade-offs, reducing catastrophic forgetting under *malicious unlearning*, and remaining competitive under *relearning via fine-tuning* while robust to *quantization attacks*.

---

## Reproducing Results

### 1. Create a Conda environment and install requirements

```bash
conda create -n FIT python=3.10.15
conda activate FIT
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn==2.5.3 --no-build-isolation
python -m spacy download en_core_web_sm
```
---
### 2. Run experiments

Use the following command to reproduce experimental results:

```bash
python main.py \
  --topk_for_forget 8 \
  --similarity_threshold 0.90 \
  --epsilon 0.2 \
  --device "cuda:0"
```

---

### 3. Evaluate the unlearned model

Run the provided notebook:

```bash
FQ_RU_cal.ipynb
```


