<div align="center">

# HybridThinker

### Efficient Chain-of-Thought Reasoning via Compressed Memory and Transient Thought Steps

**Xin Liu**<sup>★</sup>, **Runsong Zhao**<sup>★</sup>, Xinyu Liu, Junhao Ruan, Pengcheng Huang, Shichao Dong,<br>
Chunyang Xiao, Chenglong Wang, Changliang Li, Jingbo Zhu, Tong Xiao<sup>†</sup>

<sub>Northeastern University · China Unicom Cloud-Link · NiuTrans Research</sub>

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-7A5AF8.svg)](https://2026.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03768-B31B1B.svg)](https://arxiv.org/abs/2606.03768)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Accepted at EMNLP 2026**

[Paper](https://arxiv.org/abs/2606.03768) · [PDF](https://arxiv.org/pdf/2606.03768) · [Code](https://github.com/lx-Meteors/HybridThinker)

<sub><sup>★</sup>Co-first authors (equal contribution). <sup>†</sup>Corresponding author.</sub>

</div>

> [!NOTE]
> This repository contains the implementation of **HybridThinker**. To reproduce **MemoSight** or **LightThinker**, switch to the `MemoSight` branch. The environment setup and dependencies are shared across these implementations.

## News

- **2026-08** — HybridThinker was accepted at **EMNLP 2026**.
- **2026-06** — The paper and code were released.

## Overview

Long chain-of-thought (CoT) traces improve reasoning, but they also make inference increasingly expensive: self-attention grows quadratically with sequence length, while the KV cache grows linearly. Existing CoT compression methods reduce this overhead by replacing completed thought steps with compact memory tokens, but immediately discarding the original steps can erase exact values, constraints, and intermediate conclusions.

**HybridThinker keeps both levels of information:** compact memory tokens preserve the global reasoning history, while recent thought steps remain temporarily available as fine-grained local context. During training, **Hybrid Attention** mixes two complementary attention patterns to prevent shortcut learning:

| Path | Training behavior | Purpose |
| --- | --- | --- |
| **Bottleneck Attention** | Hides completed thought steps from subsequent steps | Forces the model to compress and retrieve information through memory tokens |
| **Shortcut Attention** | Keeps selected thought steps visible for a short horizon | Aligns training with the transient-memory behavior used at inference time |

<p align="center">
  <a href="https://arxiv.org/abs/2606.03768">
    <img src="https://arxiv.org/html/2606.03768v1/x1.png" width="1000" alt="Comparison of standard reasoning, existing CoT compression methods, and HybridThinker">
  </a>
  <br>
  <sub><strong>Figure 1.</strong> HybridThinker temporarily retains recent thought steps alongside compressed memory tokens, balancing reasoning accuracy and inference efficiency.</sub>
</p>

This hybrid design preserves both coarse-grained memory and fine-grained local evidence, improving accuracy without giving up the efficiency benefits of CoT compression.

## Highlights

- **Accurate:** improves the previous state of the art in CoT compression by **5.8 average accuracy points** across four reasoning benchmarks.
- **Efficient:** matches the uncompressed Qwen2.5-7B baseline while using substantially fewer peak KV-cache tokens and less total inference time.
- **Simple:** requires no auxiliary encoder or retriever—only memory tokens, step segmentation, and a custom attention mask.
- **General:** evaluated on **GSM8K, MMLU, GPQA, and BBH** with **Qwen2.5-7B** and **Llama3.1-8B** backbones.

### Main results

The following averages are taken from Table 1 of the paper. `Time` is total inference time over the evaluation samples, and `Peak` is the maximum number of tokens stored in the KV cache.

| Backbone | Method | Avg. Acc. ↑ | Time ↓ | Peak ↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B | Vanilla | 68.78 | 21.54 | 3,299 |
|  | LightThinker | 62.94 | 17.45 | **1,115** |
|  | **HybridThinker** | **68.78** | **17.15** | 1,264 |
| Llama3.1-8B | Vanilla | **68.99** | 27.32 | 3,460 |
|  | LightThinker | 64.25 | 22.12 | **1,132** |
|  | **HybridThinker** | 66.98 | **20.83** | 1,233 |

## Installation

The released experiments use Linux, CUDA, and multi-GPU training with DeepSpeed ZeRO-3 offload.

```bash
git clone https://github.com/lx-Meteors/HybridThinker.git
cd HybridThinker

conda create -n hybridthinker python=3.9 -y
conda activate hybridthinker
pip install -r requirements.txt
```

The pinned environment uses PyTorch 2.5.1, Transformers 4.46.3, and DeepSpeed 0.15.3. Make sure the installed PyTorch build matches your CUDA environment.

## Data preparation

Training data are read from JSONL. Each record is expected to contain the following fields:

```json
{
  "question": "...",
  "system_prompt": "...",
  "system_list": [],
  "question_list": [],
  "thoughts_list": ["thought step 1", "thought step 2"],
  "gt_output": "..."
}
```

For evaluation, place the benchmark files at:

```text
data/eval/
├── bbh.json
├── gpqa.json
├── gsm8k.json
└── mmlu.json
```

The paper trains on **Bespoke-Stratos-17k**. Data files are not tracked in the current repository, so prepare them locally and pass the training JSONL path through `--train_data_path`.

## Quick start

The recommended entry point is [`scripts/pipeline.sh`](scripts/pipeline.sh), which provides a unified interface for training, inference, and evaluation.

```bash
bash scripts/pipeline.sh --help
```

### End-to-end workflow

Update the model, tokenizer, data, and output paths for your environment:

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag hybridthinker-qwen-7b \
  --output_base_dir /path/to/experiments \
  --model_type qwen \
  --tokenizer_path /path/to/Qwen2.5-7B-Instruct \
  --train_model_path /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --train_data_path /path/to/train.jsonl \
  --mode aug-wo-pc \
  --conf_version v1 \
  --comp_config configs/LightThinker/qwen/v1.json \
  --use_epl false \
  --lr 2e-5 \
  --max_length 4096 \
  --epochs 5 \
  --micro_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --train_gpus 0,1,2,3,4,5,6,7 \
  --target_gpus 0,1,2,3,4,5,6,7 \
  --process_per_gpu 1 \
  --max_new_tokens 10240 \
  --datasets gsm8k,mmlu,gpqa,bbh
```

The pipeline supports the following stages:

| Stage | Behavior |
| --- | --- |
| `train` | Train and save checkpoints only |
| `infer` | Run inference and then evaluate the generated files |
| `eval` | Evaluate existing inference outputs only |
| `all` | Run training, inference, and evaluation end to end |

When `--ckpt` is omitted, inference automatically selects the latest `checkpoint-*` directory. You can also bypass the training directory and run a model directly with `--infer_model_path`.

### Paper settings

The main experiments reported in the paper use:

| Setting | Value |
| --- | --- |
| Initialization | DeepSeek-R1-Distill Qwen2.5-7B / Llama3.1-8B |
| Training data | Bespoke-Stratos-17k |
| Epochs | 5 |
| Maximum sequence length | 4,096 |
| Global batch size | 64 |
| Learning rate | 2e-5 |
| Scheduler | Cosine, 0.05 warmup ratio |
| Memory tokens `L` | 9 |
| Retention duration `w` | 4 thought steps |
| Shortcut steps `|I|` | 2 per training instance |
| Decoding | Greedy, repetition penalty 1.1 |
| Maximum output length | 10,240 |
| Hardware used in the paper | 8 × NVIDIA H200, DeepSpeed ZeRO-3 offload |

For smaller machines, reduce `--micro_batch_size`, `--max_length`, and the number of inference workers specified by `--process_per_gpu`.

### Outputs

Each experiment is self-contained under `<output_base_dir>/<exp_tag>/`:

```text
<exp_tag>/
├── train/                  # checkpoints and training logs
├── inference/              # generated JSONL files and worker logs
├── eval/                   # evaluation logs and metrics
├── run_<timestamp>.txt     # resolved argument snapshot
└── pipeline_<timestamp>.sh # pipeline snapshot for reproducibility
```

The pipeline also maintains `*_latest` symbolic links for the most recent argument, script, and log snapshots.

## Configuration

Compression behavior is controlled by JSON files under [`configs/LightThinker`](configs/LightThinker):

```text
configs/LightThinker/
├── qwen/
│   ├── v1.json
│   ├── adaptive_v1.json
│   └── adaptive_mtp_v1.json
└── llama/
    ├── v1.json
    ├── apa_mtp.json
    └── adaptive_mtp_v1.json
```

The `v1` presets use a fixed memory-token budget; the `adaptive_*` and `*_mtp` presets expose experimental adaptive-compression and multi-token-prediction variants. For the complete low-level argument reference, see [`ARGS.md`](ARGS.md).

Frequently used arguments:

| Argument | Description |
| --- | --- |
| `--mode` | Training mode; `aug-wo-pc` is the paper-oriented compressed-CoT path |
| `--conf_version` | Training configuration name under `configs/LightThinker/<model_type>/` |
| `--comp_config` | Compression configuration used during inference and evaluation |
| `--train_model_path` | Base model used to initialize training |
| `--infer_model_path` | Optional direct path for inference without a local training run |
| `--train_gpus` / `--target_gpus` | Comma-separated GPU IDs for training and inference |
| `--datasets` | Comma-separated evaluation datasets |
| `--ckpt` | Checkpoint number; defaults to the latest available checkpoint |

## Repository structure

```text
HybridThinker/
├── LightThinker/           # main training and HybridThinker inference implementation
├── AnLLM/                  # AnLLM baseline implementation
├── evaluation/             # benchmark readers and evaluation scripts
├── configs/                # model, compression, and DeepSpeed configurations
├── scripts/pipeline.sh     # unified reproducible workflow
├── assets/                 # documentation assets
├── ARGS.md                 # detailed argument reference
└── requirements.txt
```

## Acknowledgements

This implementation builds on [LightThinker](https://github.com/zjunlp/LightThinker). We thank its authors and the developers of [Transformers](https://github.com/huggingface/transformers), [DeepSpeed](https://github.com/microsoft/DeepSpeed), and the open-source models and datasets used in this project.

## Citation

If HybridThinker is useful for your research, please cite our paper. The citation will be updated with the ACL Anthology entry after the EMNLP proceedings are published.

```bibtex
@article{liu2026hybridthinker,
  title   = {HybridThinker: Efficient Chain-of-Thought Reasoning via Compressed Memory and Transient Thought Steps},
  author  = {Liu, Xin and Zhao, Runsong and Liu, Xinyu and Ruan, Junhao and Huang, Pengcheng and Dong, Shichao and Xiao, Chunyang and Wang, Chenglong and Li, Changliang and Zhu, Jingbo and Xiao, Tong},
  journal = {arXiv preprint arXiv:2606.03768},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
