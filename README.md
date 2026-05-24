
```
/*
 *                                                     __----~~~~~~~~~~~------___
 *                                    .  .   ~~//====......          __--~ ~~
 *                    -.            \_|//     |||\\  ~~~~~~::::... /~
 *                 ___-==_       _-~o~  \/    |||  \\            _/~~-
 *         __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
 *     _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
 *   .~       .~       |   \\ -_    /  /-   /   ||      \   /
 *  /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
 *  |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
 *           '         ~-|      /|    |-~\~~       __--~~
 *                       |-~~-_/ |    |   ~\_   _-~            /\
 *                            /  \     \__   \/~                \__
 *                        _--~ _/ | .-~~____--~-/                  ~~==.
 *                       ((->/~   '.|||' -_|    ~~-/ ,              . _||
 *                                  -_     ~\      ~~---l__i__i__i--~~_/
 *                                  _-~-__   ~)  \--______________--~~
 *                                //.-~~~-~_--~- |-------~~~~~~~~
 *                                       //.-~~~--\
 *                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *                               神兽保佑            永无BUG
 */
```

## Table of Contents

- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🎁[Acknowledgement](#acknowledgement)
- 🚩[Citation](#citation)


## 🔧Installation

```bash
git clone https://github.com/xxx
cd HybridThinker
conda create -n HybridThinker python=3.9 -y
conda activate HybridThinker
pip install -r requirements.txt
cd data && unzip data.zip && cd ..
```


## 🏃Quick Start

本项目推荐通过统一脚本 `scripts/pipeline.sh` 运行训练、推理、评估，避免手动拼接多段命令。

脚本支持 4 个阶段：

- `--stage train`：只训练
- `--stage infer`：只推理（默认自动选择最新 checkpoint）
- `--stage eval`：只评估
- `--stage all`：训练 + 推理 + 评估全流程

可先查看帮助：

```bash
bash scripts/pipeline.sh -h
```

### 参数约定

必传通用参数：

- `--stage`
- `--exp_tag`：实验名（同时作为模型 tag）
- `--output_base_dir`：输出根目录

训练常用参数：

- `--use_epl`、`--lr`、`--mode`
- `--tokenizer_path`、`--model_path`、`--train_data_path`
- `--train_gpus`（逗号分隔，如 `0,1,2,3`）

推理常用参数：

- `--target_gpus`（逗号分隔）
- `--process_per_gpu`（每卡并发进程数）
- `--datasets`（逗号分隔）
- `--ckpt`（可选，不传时自动取最新）

评估常用参数：

- `--eval_method`（默认 `normal`）
- `--datasets`
- `--comp_config`
- `--interaction`（`true/false`）

### 示例 1：仅训练

```bash
bash scripts/pipeline.sh \
  --stage train \
  --exp_tag vanilla_qwen \
  --output_base_dir /mnt/HybridThinker/experiments \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path /mnt/hf_models/Qwen2.5-1.5B-Instruct \
  --model_path /mnt/hf_models/DeepSeek-R1-Distill-Qwen-1.5B \
  --train_data_path /mnt/RRcot/data/train/train_debug.jsonl \
  --train_gpus 0,1,2,3
```

### 示例 2：仅推理（自动使用最新 checkpoint）

```bash
bash scripts/pipeline.sh \
  --stage infer \
  --exp_tag vanilla_qwen \
  --output_base_dir /mnt/RRcot/experiments \
  --use_epl false \
  --model_type qwen \
  --tokenizer_path /mnt/hf_models/Qwen2.5-1.5B-Instruct \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

### 示例 3：全流程（train + infer + eval）

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag vanilla_qwen \
  --output_base_dir /mnt/RRcot/experiments \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path /mnt/hf_models/Qwen2.5-1.5B-Instruct \
  --model_path /mnt/hf_models/DeepSeek-R1-Distill-Qwen-1.5B \
  --train_data_path /mnt/RRcot/data/train/train_debug.jsonl \
  --train_gpus 0,1,2,3 \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

### 输出目录说明

运行后所有产物会放在：

`<output_base_dir>/<exp_tag>/`

常见内容包括：

- `train/`：训练日志和 checkpoint
- `inference/`：推理输出和子进程日志
- `eval/`：评估日志与结果
- `run_*.txt`：本次运行参数快照
- `pipeline_*.sh`：运行时脚本快照（便于复现）

并且会维护软链接：

- `run_latest.txt`
- `pipeline_latest.sh`
- 各阶段 `*_latest.log`

### 常见问题

1. `--stage infer` 报找不到 checkpoint  
请先执行训练，或手动指定 `--ckpt`。

2. 显存不足（OOM）  
优先降低 `--micro_batch_size`，其次减小 `--max_length`，并适当调低 `--process_per_gpu`。

3. 参数拼写错误导致脚本退出  
可先执行 `bash scripts/pipeline.sh -h`，确认参数名与取值。

