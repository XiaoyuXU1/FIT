# FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning

<div align="center">

[🌐 Website](https://xiaoyuxu1.github.io/FIT_PCH/)

</div>

---

## ✨ Overview

This repository provides the official implementation of **FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning**.

We introduce **FIT**, a framework designed to address the challenges of unlearning in large language models (LLMs) through:

- **🧹 Filtering**: rigorous data filtering to reduce repeated forgetting  
- **📌 Importance-aware updates**: adaptive strategies guided by sample importance  
- **🎯 Targeted layer attribution**: precise identification of layers most responsible for forget set  

Together, these components enable effective forgetting without compromising model utility.

---

## 🧪 Benchmark: PCH

To support realistic evaluation, we propose the **PCH benchmark**, which encompasses diverse deletion scenarios:

- **👤 Personal information**
- **📄 Copyrighted material**
- **⚠️ Harmful content**

We also introduce comprehensive metrics to jointly measure forgetting effectiveness and utility retention.
Across multiple LLMs, FIT consistently outperforms strong baselines, achieving better forgetting–utility trade-offs, reducing catastrophic forgetting under *malicious unlearning*, and remaining competitive under *relearning via fine-tuning* while robust to *quantization attacks*.

---

## 🔁 Reproducing Results

### 🧰 1. Create a Conda environment and install requirements

```bash
conda create -n FIT python=3.10.15
conda activate FIT
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn==2.5.3 --no-build-isolation
python -m spacy download en_core_web_sm
````

---

### 🚀 2. Run experiments and evaluations

Use the following command to reproduce experimental results:

```bash
python FIT.py \
  --unlearning_model_name "FITPCH/Llama-2-7b-chat-hf_PCH_finetune" \
  --retain_model_name "FITPCH/Llama-2-7b-chat-hf_PCH_retain" \
  --topk_for_forget 0.25 \
  --similarity_threshold 0.90 \
  --epsilon 0.2 \
  --device "cuda:0"
```
**Evaluation (MMLU / CommonsenseQA / GSM8K):** we use [OpenCompass](https://github.com/open-compass/opencompass).  
**Quantization attack:** we follow [FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning).

## 📝 Citation

If you find FIT or the PCH benchmark useful, please cite:

```bibtex
@misc{xufit2026,
  title={{FIT}: Defying Catastrophic Forgetting in Continual {LLM} Unlearning},
  author={Xiaoyu Xu and Minxin Du and Kun Fang and Zi Liang and Yaxin Xiao and Zhicong Huang and Cheng Hong and Qingqing Ye and Haibo Hu},
  howpublished={arXiv:2601.21682},   
  year={2026}
}
```



