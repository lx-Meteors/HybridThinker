import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import argparse
import random
import numpy as np
import torch
from typing import *
from tqdm import tqdm
from copy import deepcopy
from model_qwen import Qwen2ForCausalLM
from model_llama import LlamaForCausalLM
from transformers import Trainer, TrainingArguments, set_seed as hf_set_seed
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback


# ===== 关键：在任何其他操作前绑定GPU设备 =====
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank) 
    print(f"Process {os.getpid()} set to device: cuda:{local_rank}")
else:
    local_rank = 0
    print("Running in single-GPU mode")
# =========================================

from config import Config
from LightThinker.utils import _print, IGNORE_LABEL_ID, str2bool, count_mtp_lm_loss_denominators
from tokenizer import Tokenizer
from dataset import MyDataset, MyDataCollator


class CustomTrainer(Trainer):
    """
    在「一次 optimizer step」内先 prefetch 全部 micro-batch，再为本 step 聚合 MTP / LM 各自的全局有效 token 数，
    并注入 forward，使 fixed_cross_entropy 的 sum / N_global 与梯度累计、多卡 gather 语义一致。
    """

    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        self._mtp_num_items_in_batch = None
        self._lm_num_items_in_batch = None

        if not self.model_accepts_loss_kwargs:
            return batch_samples, None

        num_items_in_batch = None
        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            try:
                num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples)
            except (TypeError, AttributeError):
                pass

        mtp_total = 0
        lm_total = 0
        has_reg = False
        for batch in batch_samples:
            reg = batch.get("register_token_index", None)
            if reg is not None:
                has_reg = True
                m, l = count_mtp_lm_loss_denominators(batch["labels"], reg)
                mtp_total += m
                lm_total += l

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

        if has_reg:
            if self.args.average_tokens_across_devices:
                device = self.args.device
                mtp_t = torch.tensor(mtp_total, device=device, dtype=torch.long)
                lm_t = torch.tensor(lm_total, device=device, dtype=torch.long)
                mtp_total = self.accelerator.gather(mtp_t).sum().item()
                lm_total = self.accelerator.gather(lm_t).sum().item()
            self._mtp_num_items_in_batch = mtp_total
            self._lm_num_items_in_batch = lm_total

        return batch_samples, num_items_in_batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            mtp_n = getattr(self, "_mtp_num_items_in_batch", None)
            lm_n = getattr(self, "_lm_num_items_in_batch", None)
            if mtp_n is not None:
                loss_kwargs["mtp_num_items_in_batch"] = mtp_n
            if lm_n is not None:
                loss_kwargs["lm_num_items_in_batch"] = lm_n
            inputs = {**inputs, **loss_kwargs}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=None)


class SaveTokenizerCallback(TrainerCallback):
    """保存checkpoint同时保存tokenizer"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def on_save(self, args, state, control, **kwargs):
        """在保存checkpoint时调用"""
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )
        
        if os.path.exists(checkpoint_folder):
            self.tokenizer.save_pretrained(checkpoint_folder)


# ==========================================
# 额外记录 自定义loss 到 TensorBoard（按实际 batch 聚合）
# ==========================================
class LossTracker:
    def __init__(self):
        self.micro_lm_sum = 0.0
        self.micro_mtp_sum = 0.0
        self.micro_total_sum = 0.0
        self.micro_count = 0
        self.window_lm_sum = 0.0
        self.window_mtp_sum = 0.0
        self.window_total_sum = 0.0
        self.window_step_count = 0

    def add_micro(self, lm_loss: float, mtp_loss: float, total_loss: float):
        self.micro_lm_sum += float(lm_loss)
        self.micro_mtp_sum += float(mtp_loss)
        self.micro_total_sum += float(total_loss)
        self.micro_count += 1

    def flush_micro_to_step(self):
        if self.micro_count == 0:
            return
        # 一个 optimizer step 内多个 micro-batch 先求和，再作为一个“实际 batch”
        self.window_lm_sum += self.micro_lm_sum
        self.window_mtp_sum += self.micro_mtp_sum
        self.window_total_sum += self.micro_total_sum
        self.window_step_count += 1
        self.micro_lm_sum = 0.0
        self.micro_mtp_sum = 0.0
        self.micro_total_sum = 0.0
        self.micro_count = 0

    def pop_window_avg(self):
        if self.window_step_count == 0:
            return None
        avg_lm = self.window_lm_sum / self.window_step_count
        avg_mtp = self.window_mtp_sum / self.window_step_count
        avg_total = self.window_total_sum / self.window_step_count
        self.window_lm_sum = 0.0
        self.window_mtp_sum = 0.0
        self.window_total_sum = 0.0
        self.window_step_count = 0
        return avg_lm, avg_mtp, avg_total


loss_tracker = LossTracker()


def capture_loss_hook(module, input, output):
    lm_loss = getattr(module, "_last_lm_loss", 0.0)
    mtp_loss = getattr(module, "_last_mtp_loss", 0.0)
    lm_weight = float(getattr(module, "lm_loss_weight", 1.0))
    mtp_weight = float(getattr(module, "mtp_loss_weight", 1.0))
    if isinstance(lm_loss, torch.Tensor):
        lm_loss = lm_loss.detach().float().item()
    if isinstance(mtp_loss, torch.Tensor):
        mtp_loss = mtp_loss.detach().float().item()
    total_loss = lm_weight * float(lm_loss) + mtp_weight * float(mtp_loss)
    loss_tracker.add_micro(lm_loss, mtp_loss, total_loss)


class CustomLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        loss_tracker.flush_micro_to_step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        window_avg = loss_tracker.pop_window_avg()
        if window_avg is None:
            return
        lm_loss, mtp_loss, total_loss = window_avg
        logs["lm_loss"] = round(float(lm_loss), 4)
        logs["mtp_loss"] = round(float(mtp_loss), 4)
        logs["step_total_loss"] = round(float(total_loss), 4)
        # 终端/TensorBoard 的 loss 使用同一 step 口径与同一权重公式
        logs["loss"] = round(float(total_loss), 4)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="just used for deepspeed.")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--train_path', type=str, help='training dataset path')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_length', type=int, default=768)
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama'])

    parser.add_argument('--compress_config', type=str)
    parser.add_argument('--bos_token', type=str)
    parser.add_argument('--eos_token', type=str)
    parser.add_argument('--see_current', type=str2bool)
    parser.add_argument('--bi_directional', type=str2bool)
    parser.add_argument('--diagonal', type=str2bool)
    parser.add_argument('--mode', type=str, choices=['recover', 'normal', 'aug', 'aug-wo-pc', 'aug-wo-pc-apa-mtp'])
    parser.add_argument('--exclude_continue', type=str2bool)
    parser.add_argument('--qkv', type=str)
    parser.add_argument('--freeze_model', type=str2bool)
    parser.add_argument('--train_on_input', type=str2bool)
    parser.add_argument('--output_compress_instruction', type=str)
    parser.add_argument('--hybrid', type=str2bool)  
    parser.add_argument('--prefill_compress', type=str2bool, default=True)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--deepspeed', type=str, help="file path")
    parser.add_argument('--micro_batch_size', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_EPL', type=str2bool, default=False)
    args = parser.parse_args()
    return args

def get_model_and_tokenizer(
    args,
    comp_config:Config
) -> Tuple[Union[Qwen2ForCausalLM, LlamaForCausalLM], Tokenizer, Any]:

    special_token_list:List[str] = list()
    special_token_desp_dict = dict()
    tokenizer: Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else args.model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )
    assert len(comp_config.special_token_desp_list) == len(comp_config.special_token_name_list)
    for desp, token in zip(comp_config.special_token_desp_list, comp_config.special_token_name_list):
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
            # special_token_desp_list.append(desp)
            special_token_desp_dict[token] = desp
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)
    
    if args.model_type == 'llama':
        model_class = LlamaForCausalLM
    elif args.model_type == 'qwen':
        model_class = Qwen2ForCausalLM
    else:
        assert False, "We only support llama and qwen model."

    if comp_config.mtp_cfg:
        _print(f"use ce + mtp loss...")
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        mtp_params = comp_config.mtp_cfg
        _print(f"mtp config={mtp_params}")

        model_config.update(mtp_params)
        model = model_class.from_pretrained(
            args.model_path, config=model_config, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    else:
        _print(f"use ce loss...")
        model = model_class.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )

    hook_handle = model.register_forward_hook(capture_loss_hook)
    
    model.add_qkv(
        q='q' in args.qkv,
        k='k' in args.qkv,
        v='v' in args.qkv,
    )

    if model.model.config.vocab_size != len(tokenizer):
        # Expand the token embedding and lm_head
        _print(f"before.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"before.lm_head.shape={model.lm_head.weight.shape}")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _print(f"now.embedding.shape={model.model.embed_tokens.weight.shape}")
        _print(f"now.lm_head.shape={model.lm_head.weight.shape}")
    
    if args.freeze_model:
        _print(f"Freezing Model:\nnew_token: {len(special_token_list)}\norigin_length: {len(tokenizer) - len(special_token_list)}")
        model.freeze_embed(
            new_token_cnt=len(special_token_list), 
            origin_length=len(tokenizer) - len(special_token_list)
        )
    else:
        _print("mean ...")
        with torch.no_grad():
            for idx, token in enumerate(reversed(special_token_list), start=1):
                description = special_token_desp_dict[token]
                tokenized = tokenizer.tokenize(description)
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)

                # embedding layer
                new_embedding = model.model.embed_tokens.weight[tokenized_ids].mean(axis=0)
                model.model.embed_tokens.weight[-idx, :] = new_embedding.clone().detach().requires_grad_(True)

                # lm_head layer
                last_embedding = model.lm_head.weight[tokenized_ids].mean(axis=0)
                model.lm_head.weight[-idx, :] = last_embedding.clone().detach().requires_grad_(True)
    
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Trainable Parameters:")
    for param_name in trainable_params:
        print(param_name)

    return model, tokenizer, hook_handle

def get_dataset_and_data_collator(
    args,
    comp_config:Config,
    tokenizer:Tokenizer,
    padding_config:Dict,
    attention_config:Dict,
    sample_config:Dict,
) -> Tuple[MyDataset, MyDataCollator]:

    dataset = MyDataset(
        file_path=args.train_path,
        config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        train_on_input=args.train_on_input,
        change_rope=False,
        output_compress_instruction=args.output_compress_instruction,
        use_EPL=args.use_EPL,
    )

    data_collator = MyDataCollator(
        dataset=dataset,
        attention_config=attention_config,
        exclude_continue=args.exclude_continue,
        sample_config=sample_config
    )

    return dataset, data_collator


def main():
    args = get_parser()
    set_global_seed(args.seed)
    if args.output_compress_instruction == "None":
        args.output_compress_instruction = ""
    print(args)
    
    resume_from_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            resume_from_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
            print(f"发现检查点，将从 {resume_from_checkpoint} 恢复训练")

    comp_config = Config.from_file(config_path=args.compress_config)

    model, tokenizer, hook_handle = get_model_and_tokenizer(
        args, comp_config
    )

    sample_config:Dict = dict(
        mode=args.mode,
        hybrid=args.hybrid
    )
    attention_config:Dict = dict(
        diagonal=args.diagonal,
        bi_directional=args.bi_directional,
        see_current=args.see_current,
        prefill_compress=args.prefill_compress,
    )
    padding_config = dict(
        padding_side='right',
        label_padding_id=IGNORE_LABEL_ID,
        input_padding_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        position_ids_padding_id=0,
    )

    dataset, data_collator = get_dataset_and_data_collator(
        args=args, 
        comp_config=comp_config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        attention_config=attention_config,
        sample_config=sample_config,
    )

    training_config = TrainingArguments(
        lr_scheduler_type=args.lr_scheduler_type,
        local_rank=args.local_rank,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        do_eval=False,
        optim="adamw_torch",
        save_strategy="epoch",      # the default value is step
        save_steps=args.save_steps, # if the strategy is epoch, the save_steps is not used.
        output_dir=args.output_dir,
        save_only_model=False,       # don't save the global_steps
        load_best_model_at_end=False,
        # deepspeed=args.deepspeed,
        save_total_limit=1,
        report_to="tensorboard",
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    
    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        data_collator=data_collator,
        callbacks=[SaveTokenizerCallback(tokenizer), CustomLossCallback()]  # 增加 mtp_loss lm_loss 监控
    )

    # 1. 弹出默认的 TensorBoardCallback (它目前排在队列最前面)
    tb_callback = trainer.pop_callback(TensorBoardCallback)

    # 2. 如果成功移除了，再加回来保证顺序正确
    if tb_callback is not None:
        trainer.add_callback(tb_callback)

    try:
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    finally:
        hook_handle.remove()
    


if __name__ == '__main__':
    main()