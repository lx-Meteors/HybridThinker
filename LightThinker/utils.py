
import jsonlines
from typing import *
import time
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch

IGNORE_LABEL_ID = -100

def count_mtp_lm_loss_denominators(
    labels: torch.Tensor,
    register_token_index: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> Tuple[int, int]:
    """
    与 model_qwen / model_llama 中 MTP+LM 分支一致：用于 fixed_cross_entropy 的全局分母。
    - mtp：register 位置且 label 非 ignore 的数量（与 CE 中未被 ignore 的项一致）。
    - lm：非 register 展平后 shift，再扣掉跨序列边界，再统计非 ignore 的数量。
    """
    if labels is None or register_token_index is None:
        return 0, 0
    register_token_index = register_token_index.to(labels.device).bool()
    labels = labels.to(register_token_index.device)
    mtp_count = int((register_token_index & (labels != ignore_index)).sum().item())

    non_register_mask = ~register_token_index
    filtered_lm_labels = labels[non_register_mask]
    if filtered_lm_labels.numel() == 0:
        return mtp_count, 0

    shift_lm_labels = filtered_lm_labels[1:].contiguous()
    non_register_counts = non_register_mask.sum(dim=1)
    if shift_lm_labels.numel() == 0:
        return mtp_count, 0

    valid_shift_mask = torch.ones(shift_lm_labels.size(0), dtype=torch.bool, device=labels.device)
    boundary_pos = non_register_counts.cumsum(dim=0)[:-1] - 1
    boundary_pos = boundary_pos[(boundary_pos >= 0) & (boundary_pos < valid_shift_mask.size(0))]
    valid_shift_mask[boundary_pos] = False

    safe_lm_labels = shift_lm_labels[valid_shift_mask]
    lm_count = int((safe_lm_labels != ignore_index).sum().item())
    return mtp_count, lm_count

def str2bool(str_bool:str) -> bool:
    str_bool = str_bool[0].upper() + str_bool[1:].lower()
    return eval(str_bool)

def _print(messages:Any):
    print(f"[{time.ctime()}] {messages}")

def read_jsonl(file_path:str, progress:bool=True) -> List[Dict]:
    _print(f"reading jsonl file from `{file_path}` ...")
    data:List = list()
    if progress:
        pbar = tqdm(total=1)
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
            if progress:
                pbar.update(1)
    if progress:
        pbar.close()
    return data

def read_json(file_path:str) -> Dict:
    _print(f"reading json file from `{file_path}` ...")
    with open(file_path, 'r') as reader:
        data:dict = json.load(reader)
    return data

def padding_item(
    item:Dict, 
    padding_side:str, 
    label_padding_id:int, 
    input_padding_id:int, 
    max_length:int, 
    position_ids_padding_id:int=None
) -> Dict:
    for key in ['input_ids', 'labels']:
        assert key in item
    assert padding_side in ['left', 'right']#根据推理和训练区分方向
    copy_item = deepcopy(item)

    remainder_for_label = [label_padding_id] * (max_length - len(copy_item['labels']))
    remainder_for_input = [input_padding_id] * (max_length - len(copy_item['input_ids']))
    remainder_for_position_ids = None
    if 'position_ids' in item and position_ids_padding_id != None:
        remainder_for_position_ids = [position_ids_padding_id] * (max_length - len(copy_item['labels']))

    if padding_side == 'left':
        copy_item['input_ids'] = remainder_for_input + copy_item['input_ids']
        copy_item['labels'] = remainder_for_label + copy_item['labels']
        if remainder_for_position_ids:
            copy_item['position_ids'] = remainder_for_position_ids + copy_item['position_ids']
    else:
        copy_item['input_ids'] = copy_item['input_ids'] + remainder_for_input
        copy_item['labels'] = copy_item['labels'] + remainder_for_label
        if remainder_for_position_ids:
            copy_item['position_ids'] = copy_item['position_ids'] + remainder_for_position_ids
    
    return copy_item

#构建注意力掩码，都是在这里实现的
def create_attention_for_aug_data(
    input_ids:List[int],
    locate_index_list:List[List[int]],
    # [start, end, n_inst]
    locate_indicator_list:List[str],
    bi_directional:bool,
    see_current:bool,
    diagonal:bool,
    exclude_continue:bool,
    prefill_compress:bool=True,     
    delete_delay_steps:int=0,
    full_sliding_window:bool=False,
    random_sliding_window:bool=False,
    random_skip_sliding_window:bool=False,
    full_skip_sliding_window:bool=False,
    full_dropout_sliding_window:bool=False,
    random_keep_visible_count:int=0,
    max_length:int=None,
):
    # 0-False mask
    # 1-True don't mask
    length = len(input_ids)
    mask = np.ones([length, length], dtype=np.bool)
    mask = np.tril(mask)#下三角矩阵

    assert len(locate_index_list) == len(locate_indicator_list)

    pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = None, None, None, None, None
    pre_state = None

    # 窗口大小
    delete_delay_steps = max(0, int(delete_delay_steps))
    # 见过的cot step数量
    seen_output_comp = 0
    # 总共cot step数量
    total_output_comp = sum(1 for x in locate_indicator_list if x == 'compressed-output')
    last_mask_row = 0

    output_comp_mask_start_list = []
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        if index_state == 'compressed-output':
            _start, _end, _l_inst, _n_comp, _n_continue = index_item
            output_comp_mask_start_list.append((_start, _end, _l_inst, _n_comp, _n_continue))

    keep_visible_window_indices: set = set()
    valid_window_count = max(0, total_output_comp - delete_delay_steps)
    if valid_window_count > 0:
        keep_count = min(int(random_keep_visible_count), valid_window_count)
        keep_visible_window_indices = set(np.random.choice(valid_window_count, keep_count, replace=False))
        keep_visible_window_indices = {int(i) for i in keep_visible_window_indices}


    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        mask_start_row = end+l_inst+n_comp

        if full_sliding_window and index_state == 'compressed-output':
            # delete_delay_steps=2, [r1 c1 r2 c2 r3 c3] 2指的是可看见的r数量，这是一个窗口，下面的列子都是如此：
            #    r1 c1 r2 c2 r3 c3 r4 c4 r5 c5
            # r1 1
            # c1 1  1
            # r2 1  1  1
            # c2 1  1  1  1
            # r3 1  1  1  1  1
            # c3 1  1  1  1  1  1
            # r4 0  1  1  1  1  1  1
            # c4 0  1  1  1  1  1  1  1
            # r5 0  1  0  1  1  1  1  1  1
            # c5 0  1  0  1  1  1  1  1  1  1
            seen_output_comp += 1
            delete_trigger_idx = seen_output_comp + delete_delay_steps
            if delete_trigger_idx <= total_output_comp:
                _, full_end, full_l_inst, full_n_comp, _ = output_comp_mask_start_list[delete_trigger_idx - 1]
                mask_start_row = full_end + full_l_inst + full_n_comp
                mask[mask_start_row:, start:end+l_inst] = 0
        elif random_sliding_window and index_state == 'compressed-output':
            #    r1 c1 r2 c2 r3 c3 r4 c4 r5 c5
            # r1 1
            # c1 1  1
            # r2 1  1  1
            # c2 1  1  1  1
            # r3 1  1  1  1  1
            # c3 1  1  1  1  1  1
            # r4 0  1  0  1  0  1  1
            # c4 0  1  0  1  0  1  1  1
            # r5 0  1  0  1  0  1  0  1  1
            # c5 0  1  0  1  0  1  0  1  1  1
            seen_output_comp += 1
            use_keep_window = (seen_output_comp - 1) in keep_visible_window_indices
            if use_keep_window:
                delete_trigger_idx = seen_output_comp + delete_delay_steps
                _, full_end, full_l_inst, full_n_comp, _ = output_comp_mask_start_list[delete_trigger_idx - 1]
                mask_start_row = full_end + full_l_inst + full_n_comp

            mask_start_row = max(mask_start_row, int(last_mask_row))
            mask[mask_start_row:, start:end+l_inst] = 0
            last_mask_row = mask_start_row
        elif random_skip_sliding_window and index_state == 'compressed-output':
            #    r1 c1 r2 c2 r3 c3 r4 c4 r5 c5 r6 c6 r7 c7 r8 c8
            # r1 1
            # c1 1  1
            # r2 1  1  1
            # c2 1  1  1  1
            # r3 1  1  0  1  1
            # c3 1  1  0  1  1  1
            # r4 0  1  0  1  0  1  1
            # c4 0  1  0  1  0  1  1  1
            # r5 0  1  0  1  0  1  0  1  1
            # c5 0  1  0  1  0  1  0  1  1  1
            seen_output_comp += 1
            use_keep_window = (seen_output_comp - 1) in keep_visible_window_indices
            if use_keep_window:
                delete_trigger_idx = seen_output_comp + delete_delay_steps
                _, full_end, full_l_inst, full_n_comp, _ = output_comp_mask_start_list[delete_trigger_idx - 1]
                mask_start_row = full_end + full_l_inst + full_n_comp

            mask[mask_start_row:, start:end+l_inst] = 0
        elif full_skip_sliding_window and index_state == 'compressed-output':
            #    r1 c1 r2 c2 r3 c3 r4 c4 r5 c5
            # r1 1
            # c1 1  1
            # r2 1  1  1
            # c2 1  1  1  1
            # r3 1  1  0  1  1
            # c3 1  1  0  1  1  1
            # r4 0  1  1  1  0  1  1
            # c4 0  1  1  1  0  1  1  1
            # r5 0  1  0  1  1  1  0  1  1
            # c5 0  1  0  1  1  1  0  1  1  1
            # full_skip: 交错跳窗
            # 规则：
            # - 在窗口内最后一个可以看见第一个，其他都不可以看见（讨论的是原始内容r）
            seen_output_comp += 1
            current_idx = seen_output_comp - 1

            delete_trigger_idx = seen_output_comp + delete_delay_steps
            if delete_trigger_idx <= total_output_comp:
                _, full_end, full_l_inst, full_n_comp, _ = output_comp_mask_start_list[delete_trigger_idx - 1]
                mask_start_row = full_end + full_l_inst + full_n_comp
                mask[mask_start_row:, start:end+l_inst] = 0

            first_visible_idx = current_idx - delete_delay_steps
            if first_visible_idx >= 0:
                for i in range(first_visible_idx + 1, current_idx):
                    _, next_end, next_l_inst, next_n_comp, _ = output_comp_mask_start_list[i+1]
                    next_mask_start_row = next_end + next_l_inst + next_n_comp
                    old_start, old_end, old_l_inst, old_n_comp, old_n_continue = output_comp_mask_start_list[i]
                    mask_start_row = old_end + old_l_inst + old_n_comp
                    mask[mask_start_row:next_mask_start_row, old_start:old_end + old_l_inst] = 0
        elif full_dropout_sliding_window and index_state == 'compressed-output':
            seen_output_comp += 1
            delete_trigger_idx = seen_output_comp + delete_delay_steps
            # 80% 概率走 LightThinker 注意力，20% 保持训推一致滑窗注意力
            if delete_trigger_idx <= total_output_comp and np.random.random() < 0.20:
                _, full_end, full_l_inst, full_n_comp, _ = output_comp_mask_start_list[delete_trigger_idx - 1]
                mask_start_row = full_end + full_l_inst + full_n_comp
            mask[mask_start_row:, start:end+l_inst] = 0
        else:
            mask[mask_start_row:, start:end+l_inst] = 0
        
        # 1. attention_mask
        if exclude_continue:# 让后续的 Token（未来）无法看到 原始文本 和 压缩指令（过去）
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0
            if pre_n_continue is not None and pre_n_continue != 0:
                mask[end+l_inst+n_comp:, pre_end+pre_n_inst+pre_n_comp:pre_end+pre_n_inst+pre_n_comp+pre_n_continue] = 0

        # 1.1 prefill remove compress（默认为true，可以忽略）
        if not prefill_compress and index_state == 'compressed-prompt':
            mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp] = np.tri(len(mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp]), dtype=int)
            # print("prefill:", end+l_inst+n_comp)
            
        if not prefill_compress and index_state == 'compressed-output' and pre_state == 'compressed-prompt':
            mask[0:start, 0:start] = np.tri(start, dtype=int)
            # print("last:", start)
        
        # if not prefill_compress and index_state == 'save' and pre_state == 'compressed-prompt':
        #     mask[0:end, 0:end] = np.tri(start, dtype=int)
        
        pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = start, end, l_inst, n_comp, n_continue

        # if not prefill_compress and index_state == 'save' and pre_state == 'compressed-prompt':
        #     pre_state = 'compressed-prompt'
        # else:
        pre_state = index_state

        # 2. bi_directional
        if bi_directional:
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = 1
        
        # 3. diagonal
        if diagonal and end+l_inst+n_comp < len(mask):
            # mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
            #     np.eye(n_comp, dtype=int)
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
                np.eye(len(mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp]), dtype=int)
        
        # 4. see_current
        if see_current:
            mask[end+l_inst:end+l_inst+n_comp, 0:start] = 0

    # 5. padding（填充）
    if max_length is not None and max_length > length:
        padding_mask = np.zeros([max_length, max_length], dtype=np.bool)
        padding_mask[:length, :length] = mask
        padding_mask[length:, :length] = mask[-1:]
        results = torch.as_tensor(padding_mask)
    else:
        results = torch.as_tensor(mask)

    return results
def create_attention_for_aug_data_apa_mtp(
    input_ids:List[int],
    locate_index_list:List[List[int]],
    # [start, end, n_inst]
    locate_indicator_list:List[str],
    exclude_continue:bool,
    prefill_compress:bool=True,     
    max_length:int=None
):
    """
    构建 apa_mtp 模式的注意力掩码。
    规则：
    - R(register) 与 T(文本token) 注意力一致：只能看到 P(prompt)、C(compress)、T(文本)，不能看到其他 R
    - T 和 C 也不能看到任何 R
    实现：将所有 R 位置从 key 维度 mask 掉，使得任何 token 都无法 attend 到 R
    """
    # 0-False mask, 1-True don't mask
    length = len(input_ids)
    mask = np.ones([length, length], dtype=np.bool)
    mask = np.tril(mask)  # 下三角矩阵

    assert len(locate_index_list) == len(locate_indicator_list)

    # 收集所有 register token 的位置，构建 register_token_index tensor
    # 在 apa_mtp 中，abandoned 段 [start:end] 格式为 [R,T,R,T,...,R,T]，R 在偶数偏移处
    register_token_index = np.zeros(length, dtype=np.int64)
    for index_item in locate_index_list:
        start, end, l_inst, n_comp, n_continue = index_item
        # [start:end] 为 abandoned 段，R 在 start, start+2, start+4, ...
        for k in range((end - start) // 2):
            r = start + 2 * k
            if r < length:
                register_token_index[r] = 1
                mask[:, r] = 0  # 默认任何 token 都不能 attend 到该 R
                mask[r, r] = 1  # 允许该 R attend 到自己，但不能 attend 到其他 R

    # pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = None, None, None, None, None
    # pre_state = None

    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        
        mask[end+l_inst+n_comp:, start:end+l_inst] = 0  # 让后续 token 无法看到 原始文本 和 压缩指令
               
        # # 1. attention_mask：让后续 token 无法看到 原始文本 和 压缩指令
        # if exclude_continue:
        #     mask[end+l_inst+n_comp:, start:end+l_inst] = 0
        #     if pre_n_continue is not None and pre_n_continue != 0:
        #         mask[end+l_inst+n_comp:, pre_end+pre_n_inst+pre_n_comp:pre_end+pre_n_inst+pre_n_comp+pre_n_continue] = 0
        # else:
        #     mask[end+l_inst+n_comp:, start:end+l_inst] = 0

        # # 1.1 prefill remove compress
        # if not prefill_compress and index_state == 'compressed-prompt':
        #     mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp] = np.tri(len(mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp]), dtype=int)
            
        # if not prefill_compress and index_state == 'compressed-output' and pre_state == 'compressed-prompt':
        #     mask[0:start, 0:start] = np.tri(start, dtype=int)
        
        # pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = start, end, l_inst, n_comp, n_continue
        # pre_state = index_state

    # 5. padding（填充）
    if max_length is not None and max_length > length:
        padding_mask = np.zeros([max_length, max_length], dtype=np.bool)
        if length > 0:
            padding_mask[:length, :length] = mask
            padding_mask[length:, :length] = mask[-1:]
        results = torch.as_tensor(padding_mask)
        # register_token_index 右填充 0 至 max_length
        register_token_index_padded = np.zeros(max_length, dtype=np.int64)
        register_token_index_padded[:length] = register_token_index
        register_token_index = register_token_index_padded
    else:
        results = torch.as_tensor(mask)

    return results, register_token_index 
def create_attention_for_recover_data(
    input_ids:List[int],
    locate_index_list:List[List[int]],
    locate_indicator_list:List[str],
    bi_directional:bool,
    see_current:bool,
    diagonal:bool,
    exclude_continue:bool,
    added_input_ids:List[int],
    added_labels:List[int],
    # [end, l_inst, n_comp, n_continnue]
    added_corresp_attention:List[int],
    max_length:int=None,
    prefill_compress:bool=True,
    return_offset:bool=False,
) -> torch.Tensor:

    # 0-False mask
    # 1-True don't mask
    length = len(input_ids) + len(added_input_ids)
    mask = np.ones([length, length], dtype=np.bool)
    mask = np.tril(mask)
    
    assert len(locate_index_list) == len(locate_indicator_list)

    # 1. augument part
    pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = None, None, None, None, None
    pre_state = None
    
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        
        # 1. attention_mask
        if exclude_continue:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0
            if pre_n_continue is not None and pre_n_continue != 0:
                mask[end+l_inst+n_comp:, pre_end+pre_n_inst+pre_n_comp:pre_end+pre_n_inst+pre_n_comp+pre_n_continue] = 0
        else:
            mask[end+l_inst+n_comp:, start:end+l_inst] = 0

        # 1.1 prefill remove compress
        if not prefill_compress and index_state == 'compressed-prompt':
            mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp] = np.tri(len(mask[0:end+l_inst+n_comp, 0:end+l_inst+n_comp]), dtype=int)
        if not prefill_compress and index_state == 'compressed-output' and pre_state == 'compressed-prompt':
            mask[0:start, 0:start] = np.tri(len(mask[0:start, 0:start]), dtype=int)

        pre_start, pre_end, pre_n_inst, pre_n_comp, pre_n_continue = start, end, l_inst, n_comp, n_continue
        pre_state = index_state

        # 2. bi_directional
        if bi_directional:
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = 1
        
        # 3. diagonal
        if diagonal:
            assert diagonal == False, "we do not "
            mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp] = \
                np.eye(len(mask[end+l_inst:end+l_inst+n_comp, end+l_inst:end+l_inst+n_comp]), dtype=int)
        
        # 4. see_current
        if see_current:
            mask[end+l_inst:end+l_inst+n_comp, 0:start] = 0
        
    # 2. recover part

    start, end, l_inst, n_comp, n_continue = None, None, None, None, None
    pre_state = None
    for index_item, index_state in zip(locate_index_list, locate_indicator_list):
        assert index_state in ['compressed-prompt', 'compressed-output']
        start, end, l_inst, n_comp, n_continue = index_item
        if index_state == 'compressed-output':
            need_mask = True
            for item in added_corresp_attention:
                _end, _l_inst, _n_comp, _n_continue = item
                if end == _end and l_inst == _l_inst and \
                    n_comp == _n_comp and n_continue == _n_continue:
                    need_mask = False
                    break
            if need_mask:
                mask[
                    len(input_ids):, start:end
                ] = 0
                mask[
                    len(input_ids):, end:end+l_inst+n_comp+n_continue
                ] = 0

    if start != None:
        if len(input_ids) > end+l_inst+n_comp+n_continue:
            mask[len(input_ids):, end+l_inst+n_comp+n_continue:len(input_ids)] = 0
    offset = None
    for i in range(len(input_ids)-1, -1, -1):
        if mask[-1, i] == 1:
            offset = i+1
            break
    
    # 5. padding
    if max_length is not None and max_length > length:
        padding_mask = np.zeros([max_length, max_length], dtype=np.bool)
        padding_mask[:length, :length] = mask
        padding_mask[length:, :length] = mask[-1:]
        results = torch.as_tensor(padding_mask)
    else:
        results = torch.as_tensor(mask)

    if return_offset:
        return results, offset
    else:
        return results

def create_attention_mask(attention_mask_list:List[torch.Tensor], dtype):
    # [bs, 1, max_length, max_length]
    min_dtype = torch.finfo(dtype).min
    for i in range(len(attention_mask_list)):
        attention_mask_list[i] = attention_mask_list[i].to(dtype)
        attention_mask_list[i][torch.where(attention_mask_list[i]==0)] = min_dtype
        attention_mask_list[i][torch.where(attention_mask_list[i]==1)] = 0.
        attention_mask_list[i] = attention_mask_list[i].unsqueeze(dim=0).unsqueeze(dim=1) # [1, 1, max_length, max_length]
    return torch.cat(attention_mask_list, dim=0)

def visualize_attention_mask(
    attention_mask:List[List], 
    input_ids:List[int], 
    tokenizer,
    output_file_name:str="attention_mask4.png",
    position_id:List=None,
    start_offset:int=None,
    end_offset:int=None
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    plt.rcParams['font.size'] = 5

    if start_offset != None:
        attention_mask = np.array(attention_mask)[start_offset:end_offset, start_offset:end_offset].tolist()
        input_ids = input_ids[start_offset:end_offset]

    if position_id == None:
        position_id = list(range(len(input_ids)))
    

    # True -> don't mask
    # False -> mask
    # in dataset.py, the value in attention_mask is 0 and -inf,
    # so if the value is 0, it means we don't mask
    for i in range(len(attention_mask)):
        for j in range(len(attention_mask[i])):
            if attention_mask[i][j] == 0.:
                attention_mask[i][j] = False
            else:
                attention_mask[i][j] = True
    attention_mask = np.array(attention_mask)
    print(input_ids)
    label = ["#"+tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids] + ['#']
    print(label)
    position_id.append(-1)
    xlabel = ["\n" + l for l in label]
    ylabel = ["\n" + ("" if position_id == None else f"({position_id[idx]})") + l for idx, l in enumerate(label)]

    cmap = mcolors.ListedColormap(['yellow', 'lightgray'])
    plt.imshow(attention_mask, cmap=cmap)
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-.5, len(attention_mask), 1), xlabel, rotation=90)
    plt.yticks(np.arange(-.5, len(attention_mask[0]), 1), ylabel)
    plt.savefig(output_file_name, bbox_inches='tight', pad_inches=0.1)

def visualize_labels(
    input_ids:List[int],
    labels:List[int],
    tokenizer,
    position_ids:List[int]=None
):
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()
    if position_ids is None:
        position_ids = list(range(len(input_ids)))
    if not isinstance(labels, list):
        labels = labels.tolist()
    assert len(input_ids) == len(labels)
    results = []
    for input_token_id, label_id, pos_id in zip(input_ids[0:-1], labels[1:], position_ids[0:-1]):
        input_token:str = tokenizer.convert_ids_to_tokens(input_token_id)
        if label_id == -100:
            results.append(f"`{input_token} ({pos_id})` -> `No`")
        else:
            results.append(f"`{input_token} ({pos_id})` -> `{tokenizer.convert_ids_to_tokens(label_id)}`")
    # print("\n".join(results))
    return "\n".join(results)

