<div align="center">

# MemoSight

### Unifying Context Compression and Multi-Token Prediction for Reasoning Acceleration

Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan,<br>
Bei Li, Chunyang Xiao, Chenglong Wang, Tong Xiao, Jingbo Zhu

<sub>Northeastern University · Meituan Inc. · NiuTrans Research</sub>

[![arXiv](https://img.shields.io/badge/arXiv-2604.14889-B31B1B.svg)](https://arxiv.org/abs/2604.14889)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2604.14889) · [PDF](https://arxiv.org/pdf/2604.14889) · [Code](https://github.com/helldog-star/MemoSight)

**English** · [中文](README_zh.md)

</div>

> [!NOTE]
> This repository implements **MemoSight** and also provides a **LightThinker-compatible** training and inference path. Both methods share the same environment, data format, evaluation suite, and pipeline; select the desired method through the training mode, configuration preset, and decoding flags described below.

## Overview

Long chain-of-thought (CoT) reasoning improves the problem-solving ability of large language models, but the KV cache grows continuously with the generated sequence, creating substantial memory and inference overhead. Context compression and multi-token prediction (MTP) address this bottleneck from complementary directions: the former shortens the historical context, while the latter predicts future tokens in parallel. Combining them effectively, however, is difficult because they rely on different training paradigms and architectural assumptions.

**MemoSight** (Memory-Foresight-Based Reasoning) unifies both techniques in an architecture-free framework. **Memory tokens** compress completed reasoning steps to control KV-cache growth, while **foresight tokens** predict multiple future tokens for self-speculative decoding. Adaptive Memory Allocation (AMA) assigns memory capacity according to step length, and Uniform Position Layout (UPL) preserves the positional structure of compressed reasoning. All components are jointly optimized through standard supervised fine-tuning.

Across four reasoning benchmarks, MemoSight reduces KV-cache usage by up to **66%** and improves inference speed by up to **56%** over vanilla SFT, with less than a **3-point** drop in average accuracy. It also achieves a better accuracy-efficiency trade-off than existing CoT compression methods.

<p align="center">
  <a href="https://arxiv.org/abs/2604.14889">
    <img src="https://arxiv.org/html/2604.14889v2/x1.png" width="900" alt="MemoSight foresight-token-based acceleration">
  </a>
  <br>
  <sub><strong>Foresight.</strong> Multiple future tokens are drafted in one forward pass and verified in parallel.</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.14889">
    <img src="https://arxiv.org/html/2604.14889v2/x2.png" width="900" alt="MemoSight memory-token-based compression">
  </a>
  <br>
  <sub><strong>Memory.</strong> Completed reasoning steps are compressed into compact memory tokens before decoding continues.</sub>
</p>

## Highlights

- **Unified:** combines context compression and MTP through two types of special tokens and token-specific positional layouts.
- **Architecture-free:** requires no auxiliary encoder, retriever, prediction head, or additional Transformer block.
- **Memory-efficient:** reduces peak KV-cache usage by up to **66%** compared with vanilla decoding.
- **Fast:** improves inference speed by up to **56%**, and is **23.8% / 29.6%** faster than LightThinker on Qwen / Llama on average.
- **Accurate:** stays within three average accuracy points of vanilla SFT while outperforming LightThinker on both model families.

## Main Results

The following averages are taken from Table 1 of the paper. `Speed` is generated tokens per second, and `Peak` is the maximum number of context tokens retained during inference.

| Backbone | Method | Avg. Acc. ↑ | Speed ↑ | Peak ↓ |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B | Vanilla | **68.78** | 19.68 | 3,299 |
|  | LightThinker | 62.94 | 23.91 | **1,115** |
|  | **MemoSight** | 66.84 | **29.60** | 1,202 |
| Llama3.1-8B | Vanilla | **68.99** | 17.60 | 3,460 |
|  | LightThinker | 64.25 | 21.98 | 1,132 |
|  | **MemoSight** | 66.38 | **28.48** | **1,123** |

MemoSight is evaluated on **GSM8K**, **MMLU**, **GPQA**, and **BBH**.

## Supported Methods

MemoSight and the LightThinker-compatible path use the same entry script. Their essential settings are:

| Method | Training mode | Config preset | `--use_epl` | `--spec_decode` |
| --- | --- | --- | --- | --- |
| **MemoSight** | `aug-wo-pc-apa-mtp` | `adaptive_mtp_v1` | `true` | `true` |
| **LightThinker-compatible** | `aug-wo-pc` | `v1` | `false` | `false` |

## Installation

The released code targets Linux, CUDA, and multi-GPU training with DeepSpeed ZeRO-3 offload.

```bash
git clone https://github.com/helldog-star/MemoSight.git
cd MemoSight

conda create -n memosight python=3.9 -y
conda activate memosight
pip install -r requirements.txt
```

The pinned environment uses PyTorch 2.5.1, Transformers 4.46.3, and DeepSpeed 0.15.3. Make sure the installed PyTorch build matches your CUDA environment.

## Data Preparation

Training data are read from JSONL. Each record should contain:

```json
{
  "question": "...",
  "system_prompt": "...",
  "system_list": [],
  "question_list": [],
  "thoughts_list": ["reasoning step 1", "reasoning step 2"],
  "gt_output": "..."
}
```

Place evaluation files under `data/eval/`:

```text
data/eval/
├── bbh.json
├── gpqa.json
├── gsm8k.json
└── mmlu.json
```

The paper uses **Bespoke-Stratos-17k** for supervised fine-tuning. Prepare the training JSONL locally and pass its path through `--train_data_path`.

## Quick Start

The recommended entry point is [`scripts/pipeline.sh`](scripts/pipeline.sh), which provides a unified interface for training, inference, and evaluation.

```bash
bash scripts/pipeline.sh --help
```

The pipeline supports four stages:

| Stage | Behavior |
| --- | --- |
| `train` | Train and save checkpoints only |
| `infer` | Run inference with a checkpoint or model directory |
| `eval` | Evaluate existing inference outputs only |
| `all` | Run training, inference, and evaluation end to end |

### MemoSight

Update the model, tokenizer, data, and output paths for your environment:

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag memosight-qwen-7b \
  --output_base_dir /path/to/experiments \
  --model_type qwen \
  --tokenizer_path /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --train_model_path /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --train_data_path /path/to/train.jsonl \
  --mode aug-wo-pc-apa-mtp \
  --conf_version adaptive_mtp_v1 \
  --comp_config configs/LightThinker/qwen/adaptive_mtp_v1.json \
  --use_epl true \
  --spec_decode true \
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

### LightThinker-compatible path

The same environment, data, and evaluation pipeline can reproduce the LightThinker-style compression path:

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag lightthinker-qwen-7b \
  --output_base_dir /path/to/experiments \
  --model_type qwen \
  --tokenizer_path /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --train_model_path /path/to/DeepSeek-R1-Distill-Qwen-7B \
  --train_data_path /path/to/train.jsonl \
  --mode aug-wo-pc \
  --conf_version v1 \
  --comp_config configs/LightThinker/qwen/v1.json \
  --use_epl false \
  --spec_decode false \
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

When `--ckpt` is omitted, inference automatically selects the latest `checkpoint-*` directory. To run inference from an existing model directory without a local training run, use `--infer_model_path`.

## Paper Settings

| Setting | Value |
| --- | --- |
| Backbones | Qwen2.5-7B / Llama3.1-8B |
| Initialization | DeepSeek-R1-Distill models |
| Training data | Bespoke-Stratos-17k |
| Epochs | 5 |
| Global batch size | 64 |
| Token budget | 4,096 |
| Learning rate | 2e-5 |
| Scheduler | Cosine with 0.05 warmup ratio |
| NTP / MTP loss weights | 0.7 / 0.3 |
| Compression ratio `c` | 5 |
| Maximum foresight offset `d` | 2 |
| Decoding | Greedy, maximum 10,240 output tokens |
| Hardware used in the paper | 8 × NVIDIA H200, DeepSpeed ZeRO-3 offload |

> [!IMPORTANT]
> Some checked-in configuration presets contain development values that differ from the paper settings. For strict paper reproduction, align `compression_ratio`, `lm_loss_weight`, `mtp_loss_weight`, and `max_offset` with the table above before training and inference.

For smaller machines, reduce `--micro_batch_size`, `--max_length`, and `--process_per_gpu`.

## Configuration

Model-specific presets are stored under [`configs/LightThinker`](configs/LightThinker):

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

- `v1`: fixed memory-token budget; used by the LightThinker-compatible path.
- `adaptive_v1`: adaptive memory allocation without MTP.
- `adaptive_mtp_v1`: adaptive memory allocation with foresight-token MTP; used by MemoSight.
- `apa_mtp`: fixed-budget MTP ablation / experimental preset.

See [`ARGS.md`](ARGS.md) for the low-level argument reference. MTP acceptance-rate and runtime-breakdown tools are documented in [`scripts/analysis_readme.md`](scripts/analysis_readme.md).

## Outputs

Each experiment is self-contained under `<output_base_dir>/<exp_tag>/`:

```text
<exp_tag>/
├── train/                  # checkpoints and training logs
├── inference/              # generated JSONL files and worker logs
├── eval/                   # evaluation logs and metrics
├── run_<timestamp>.txt     # resolved argument snapshot
└── pipeline_<timestamp>.sh # pipeline snapshot for reproducibility
```

The pipeline also maintains `*_latest` symbolic links for the latest argument, script, and log snapshots.

## Repository Structure

```text
MemoSight/
├── LightThinker/           # model, training, and inference implementation
├── AnLLM/                  # AnLLM baseline implementation
├── configs/                # model, compression, and DeepSpeed configurations
├── evaluation/             # benchmark readers and evaluation scripts
├── scripts/                # unified pipeline and analysis tools
├── data/                   # local training and benchmark data
├── ARGS.md                 # detailed argument reference
└── requirements.txt
```

For the traditional MTP baseline, use [`scripts/pipeline_traditional_MTP.sh`](scripts/pipeline_traditional_MTP.sh).

## Citation

If MemoSight is useful for your research, please cite:

```bibtex
@article{liu2026memosight,
  title   = {MemoSight: Unifying Context Compression and Multi-Token Prediction for Reasoning Acceleration},
  author  = {Liu, Xinyu and Liu, Xin and Jin, Bo and Zhao, Runsong and Huang, Pengcheng and Ruan, Junhao and Li, Bei and Xiao, Chunyang and Wang, Chenglong and Xiao, Tong and Zhu, Jingbo},
  journal = {arXiv preprint arXiv:2604.14889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14889}
}
```

## Acknowledgments

This implementation builds on [LightThinker](https://github.com/zjunlp/LightThinker). We thank its authors and the developers of [Transformers](https://github.com/huggingface/transformers), [DeepSpeed](https://github.com/microsoft/DeepSpeed), and the open-source models and datasets used in this project.

## License

This project is released under the [MIT License](LICENSE).
