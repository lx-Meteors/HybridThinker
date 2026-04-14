
import os
import time
import torch
import argparse
import jsonlines
import numpy as np
from typing import *
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, DynamicCache, GenerationConfig,RepetitionPenaltyLogitsProcessor

from LightThinker.utils import *
from config import Config
from tokenizer import Tokenizer
from model_llama import LlamaForCausalLM
from model_qwen import Qwen2ForCausalLM
from dataset_reader import GPQAReader, MMLUReader, BBHReader, GSM8KReader, Reader
from dataset_reader_cot import MMLUCOTReader, BBHCOTReader, GSM8KCOTReader, GPQACOTReader, DISTILLCOTReader

DEBUG:bool=False
BLOCK:bool=False
TIMER:bool=True
_SAVE:str = "save"
_COMP_OUT:str = "compressed-output"
_COMP_PMP:str = "compressed-prompt"
_ABANDONED:str = "abandoned"
INDICATOR_LIST:List[str] = [
    _SAVE,
    _COMP_OUT,
    _COMP_PMP,
    _ABANDONED
]

class DebugUtils:

    @classmethod
    def show_global_attention(
        cls, 
        tokenizer:Tokenizer,
        attention_mask:List[List[Union[bool, float]]], 
        input_ids:List[int],
        position_ids:List[int]=None,
        block:bool=False,
        start_offset:int=None,
        end_offset:int=None,
        file_name:str="debug_global.png",
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        plt.rcParams['font.size'] = 5

        if position_ids == None:
            position_ids = list(range(len(input_ids)))

        if start_offset != None:
            attention_mask = np.array(
                attention_mask
            )[start_offset:end_offset, start_offset:end_offset]
            input_ids = input_ids[start_offset:end_offset]
            if position_ids != None:
                position_ids = position_ids[start_offset:end_offset]
        
        # True -> don't mask
        # False -> mask
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if isinstance(attention_mask[i][j], float):
                    if attention_mask[i][j] == 0.:
                        attention_mask[i][j] = True
                    else:
                        attention_mask[i][j] = False
        attention_mask = np.array(attention_mask)
        # print(attention_mask)
        label = [
            '# ' + tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids
        ] + ['# PlaceHolder']
        position_ids.append(-1)
        xlabel = ['\n' + l for l in label]
        ylabel = ["\n" + ("" if position_ids == None else f"({position_ids[idx]})") + l for idx, l in enumerate(label)]
        
        cmap = mcolors.ListedColormap(['lightgray', 'yellow'])
        plt.imshow(attention_mask, cmap=cmap)
        plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, len(attention_mask[0]), 1), xlabel, rotation=90)
        plt.yticks(np.arange(-.5, len(attention_mask), 1), ylabel)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
        if block:
            input("Blocking! Please enter the enter to finish blocking ...")

    @classmethod
    def show_local_attention(
        cls,
        tokenizer:Tokenizer,
        attention_mask:List[List[Union[bool, float]]],
        input_ids:List[int],
        position_ids:List[int]=None,
        block:bool=False,
        start_offset:int=None,
        end_offset:int=None,
        file_name:str="debug_local.png",
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        plt.rcParams['font.size'] = 10
        if position_ids == None:
            position_ids = list(range(len(input_ids)))

        y_input_ids:List[int] = input_ids[-len(attention_mask):]
        y_position_ids:List[int] = position_ids[-len(attention_mask):]

        if start_offset != None:
            attention_mask = np.array(
                attention_mask
            )[:, start_offset:end_offset]
            input_ids = input_ids[start_offset:end_offset]
            if position_ids != None:
                position_ids = position_ids[start_offset:end_offset]
        
        # True -> don't mask
        # False -> mask
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if isinstance(attention_mask[i][j], float):
                    if attention_mask[i][j] == 0.:
                        attention_mask[i][j] = True
                    else:
                        attention_mask[i][j] = False
        attention_mask = np.array(attention_mask)

        position_ids.append(-1)
        xlabel = [
            '\n# ' + tokenizer.convert_ids_to_tokens(token_id) + f"({position_ids[idx]})" for idx, token_id in enumerate(input_ids)
        ] + ['# PlaceHolder (-1)']
        ylabel = [
            '\n# ' + tokenizer.convert_ids_to_tokens(token_id) + f"({y_position_ids[idx]})" for idx, token_id in enumerate(y_input_ids)
        ] + ['# PlaceHolder (-1)']

        cmap = mcolors.ListedColormap(['lightgray', 'yellow'])
        plt.imshow(attention_mask, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, len(attention_mask[0]), 1), xlabel, rotation=90)
        plt.yticks(np.arange(-.5, len(attention_mask), 1), ylabel)
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
        if block:
            input("Blocking! Please enter the enter to finish blocking ...")

class InferenceUtils:

    @classmethod
    def get_predicted_token_ids(
        cls,
        model_output,
        idx:int=-1,
        token_utils=None,
        repetition_penalty:float=1.0,
        tokenizer=None,
        extra_generated_ids:List[int]=None,
    ) -> int:
        # [bs, seq_length, vocab_size]
        logits = model_output.logits    
        # [vocab_size]
        target_logits = logits[0, idx, :]

        # 使用transformers实现
        if token_utils is not None and repetition_penalty != 1.0:
            #获取上下文token
            generated_ids = set(token_utils.show_output_input_ids) | set(token_utils.show_prompt_input_ids)
            if extra_generated_ids:
                generated_ids = generated_ids | set(extra_generated_ids)
            #过滤special token
            if tokenizer is not None:
                special_ids = set(tokenizer.all_special_ids)
                generated_ids = generated_ids - special_ids

            if len(generated_ids)>0:
                #构造tensor
                input_ids_tensor = torch.tensor([list(generated_ids)],dtype=torch.long,device=target_logits.device)       
                #构造processor
                processor=RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                #处理logits
                target_logits = target_logits.unsqueeze(0)
                target_logits = processor(input_ids_tensor,target_logits)
                target_logits = target_logits.squeeze(0)

        predicted_token_ids: int = torch.argmax(target_logits).item()
        return predicted_token_ids

class AttentionUtils:

    def __init__(
        self, 
        max_length:int, 
        device:str, 
        dtype,
        attention_config:Dict,
        prefill_compress:bool,
        max_comp_size:int=10,
        n_inst:int=0,
        n_continue:int=1,
    ):
        """
        args:
            - max_length: 
                max generated tokens
            - device: 
                "cuda"
            - dtype: 
                fp16 or bf16
            - attention_config: 
                attention mask
            - prefill_compress: 
                if True:
                    we will compress the prompt part
                if False:
                    we do not compress the prompt part
            - max_comp_size:
                How many tokens can a thought be compressed into
                    used for delta_attn
            - n_inst:
                instrution length
                    used for delta_attn
            - n_continue:
                used for delta_attn
                    if we compress in thought level, n_continue will be set to 1
                    if we compress in token level, n_continue will be set to 0
        """
        self.dtype = dtype
        self.device:str = device
        self.max_length:int = max_length
        self.attention_config:Dict = attention_config
        self.prefill_compress:bool = prefill_compress

        self.min_dtype = torch.finfo(dtype).min

        # this is used for global attention.
        # this is always a lower triangular matrix
        self.base_attn = torch.triu(
            torch.full(
                (max_length, max_length), fill_value=self.min_dtype, dtype=self.dtype, device=self.device
            ),
            diagonal=1
        )
        # this is used for local attention.
        # as a matter of fact, if we use kv cache, the attention mask shape is [n_new_token, all_token]
        self.delta_attn = torch.full(
            (max_length, max_length), fill_value=0., dtype=self.dtype, device=self.device
        )
        # this is used for global attention during inference.
        self.cur_attn = torch.triu(
            torch.full(
                (max_length, max_length), fill_value=self.min_dtype, dtype=self.dtype, device=self.device
            ),
            diagonal=1
        )

        # mask value
        self.mask_value = self.min_dtype
        self.show_value = 0.
        
        # Indicates which row you are currently on to start writing.
        self.last_idx = 0

        # diagonal attention, this is used for args.diagonal being True.
        self.diagonal_attn = torch.full((max_comp_size, max_comp_size), self.mask_value)
        self.diagonal_attn.fill_diagonal_(self.show_value)
        
        # used for global attention
        self.global_indicator_list:List[List[int]] = list()

    def create_prompt_attention(
        self,
        length,
        indicator_list:List[List[int]],
    ) -> torch.Tensor:
        """
        indicator_list[0]:
            [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
        """
        self.reset()
        self.last_idx = length
        self.global_indicator_list.extend(indicator_list)

        prefill_compress = self.prefill_compress
        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        if prefill_compress == False or len(indicator_list) == 0:
            # if we do not compress the prompt, we directly return a lower triangular matrix
            return self.cur_attn[0:length, 0:length].unsqueeze(dim=0).unsqueeze(dim=0)

        assert prefill_compress == True
        for indicator in indicator_list:
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            if text_end + n_inst != comp_start:
                assert n_inst == 0
            assert n_inst == 0, "now we only support n_inst == 0"

            # 1. the current content is invisible for later content.
            self.cur_attn[
                comp_end:length, 
                text_start:text_end
            ] = self.mask_value

            # 2. if we set the see_current, it means that the previous content is not visible during compression
            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.mask_value
            
            # 3. attention config
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

        return self.cur_attn[0:length, 0:length].unsqueeze(dim=0).unsqueeze(dim=0)

    def late_ajust_prompt_attention(
        self,
        length:int,
        indicator_list:List[List[int]]
    ):
        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        assert self.prefill_compress == False
        for indicator in indicator_list:
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            if text_end + n_inst != comp_start:
                assert n_inst == 0
            assert n_inst == 0, "now we only support n_inst == 0"

            # 1. the current content is invisible for later content.
            self.cur_attn[
                comp_end:length,
                text_start:text_end
            ] = self.mask_value

            # 2. if we set the see_current, it means that the previous content is not visible during compression
            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.max_value
            
            # 3. attention config
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

    # we recommend using this function for debug.
    def update_attention_global(
        self,
        new_length:int,
        indicator:List[int],
    ):

        diagonal = self.attention_config['diagonal']
        see_current = self.attention_config['see_current']
        bi_attention = self.attention_config['bi_attention']

        # 1. copy
        self.cur_attn[
            self.last_idx:self.last_idx+new_length,
            0:self.last_idx
        ] = self.cur_attn[self.last_idx-1, 0:self.last_idx]
        
        # 2. if current is in compression mode
        if indicator != None:
            self.global_indicator_list.append(indicator)
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator

            self.cur_attn[
                comp_end:self.last_idx+new_length,
                text_start:text_end
            ] = self.mask_value

            if see_current:
                self.cur_attn[
                    comp_start:comp_end,
                    0:text_start
                ] = self.mask_value
            
            if bi_attention:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end,
                ] = self.show_value
            
            if diagonal:
                self.cur_attn[
                    comp_start:comp_end,
                    comp_start:comp_end
                ] = self.diagonal_attn[
                    0:comp_end-comp_start,
                    0:comp_end-comp_start,
                ]

        self.last_idx += new_length

        # 3. truncate and return 
        self.delta_attn[:] = self.show_value
        remove_size = 0 
        if indicator != None:
            for _indicator in self.global_indicator_list[0:-1]:
                text_start, text_end, n_inst, comp_start, comp_end, n_cont = _indicator
                remove_size += (text_end-text_start)
            text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator
            # we need add an offset bias.
            text_start -= remove_size
            text_end -= remove_size
            comp_start -= remove_size
            comp_end -= remove_size
            n_prefix = new_length - (comp_end - comp_start + n_cont)
            save_length = self.last_idx - remove_size
            self.delta_attn[
                0:new_length, 
                text_start:save_length
            ] = \
                self.cur_attn[
                    self.last_idx - new_length: self.last_idx, 
                    text_start+remove_size:self.last_idx
                ]
            self.delta_attn[0:new_length, self.last_idx - remove_size-new_length:self.last_idx - remove_size] = \
                self.base_attn[0:new_length, 0:new_length]
            return self.delta_attn[0:new_length, 0:self.last_idx-remove_size].unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            for _indicator in self.global_indicator_list:
                text_start, text_end, n_inst, comp_start, comp_end, n_cont = _indicator
                remove_size += (text_end-text_start)
            self.delta_attn[0:new_length, self.last_idx-remove_size-new_length:self.last_idx-remove_size] = \
                self.base_attn[0:new_length, 0:new_length]
            return self.delta_attn[0:new_length, 0:self.last_idx-remove_size].unsqueeze(dim=0).unsqueeze(dim=0)

    # This is a preferred method.
    def update_attention_local(
        self,
        origin_length:int,
        new_length:int,
        indicator:List[int],
    ) -> torch.Tensor:
        """
        args:
            - origin_length
            - new_length
        """

        # initialization
        self.delta_attn[0:new_length, 0:origin_length+new_length] = self.show_value
        self.delta_attn[0:new_length, origin_length:origin_length+new_length] = self.base_attn[0:new_length, 0:new_length]

        if indicator != None:
            text_start, text_end, n_inst, comp_start_c, comp_end_c, n_cont = indicator
            n_prefix = new_length - (comp_end_c - comp_start_c + n_cont)
            assert n_prefix >= 0
            comp_start_r = n_prefix
            comp_end_r = n_prefix + (comp_end_c - comp_start_c)

            # attention config
            diagonal = self.attention_config['diagonal']
            see_current = self.attention_config['see_current']
            bi_attention = self.attention_config['bi_attention']

            # mask
            self.delta_attn[
                comp_end_r:,
                text_start:text_end
            ] = self.mask_value

            if see_current:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    0:text_start
                ] = self.mask_value

            if bi_attention:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.show_value
            
            if diagonal:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.diagonal_attn[
                    0:comp_end_r-comp_start_r,
                    0:comp_end_r-comp_start_r,
                ]
        
        return self.delta_attn[0:new_length, 0:origin_length+new_length].unsqueeze(dim=0).unsqueeze(dim=0)

    def update_attention_local_for_mtp_register(
        self,
        origin_length:int,
        new_input_ids:List[int],
        register_token_id:int,
        indicator:List[int]=None,
    ) -> torch.Tensor:
        """
        local attention for [content + register...]:
        - register query can attend all historical non-register tokens (origin part)
        - register query cannot attend other registers in current step
        - register query can attend itself
        """
        new_length = len(new_input_ids)
        self.delta_attn[0:new_length, 0:origin_length+new_length] = self.show_value
        self.delta_attn[0:new_length, origin_length:origin_length+new_length] = self.base_attn[0:new_length, 0:new_length]

        if indicator is not None:
            text_start, text_end, n_inst, comp_start_c, comp_end_c, n_cont = indicator
            n_prefix = new_length - (comp_end_c - comp_start_c + n_cont)
            assert n_prefix >= 0
            comp_start_r = n_prefix
            comp_end_r = n_prefix + (comp_end_c - comp_start_c)

            diagonal = self.attention_config['diagonal']
            see_current = self.attention_config['see_current']
            bi_attention = self.attention_config['bi_attention']

            self.delta_attn[
                comp_end_r:,
                text_start:text_end
            ] = self.mask_value

            if see_current:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    0:text_start
                ] = self.mask_value

            if bi_attention:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.show_value
            
            if diagonal:
                self.delta_attn[
                    comp_start_r:comp_end_r,
                    comp_start_c:comp_end_c
                ] = self.diagonal_attn[
                    0:comp_end_r-comp_start_r,
                    0:comp_end_r-comp_start_r,
                ]

        is_register = [tok == register_token_id for tok in new_input_ids]
        for q_idx in range(new_length):
            if not is_register[q_idx]:
                continue
            for k_idx in range(new_length):
                if is_register[k_idx] and k_idx != q_idx:
                    self.delta_attn[q_idx, origin_length + k_idx] = self.mask_value

        return self.delta_attn[0:new_length, 0:origin_length+new_length].unsqueeze(dim=0).unsqueeze(dim=0)
        
    def reset(self):
        self.cur_attn[:, :] = self.base_attn[:, :]
        self.last_idx = 0
        self.global_indicator_list.clear()
        self.delta_attn[:] = self.show_value

class KVUtils:
    """
    KV Cache Manager
    """

    def __init__(self):
        self.past_key_values: DynamicCache = DynamicCache()

    def get_cache(self) -> DynamicCache:
        return self.past_key_values

    def set_cache(self, past_key_values:DynamicCache):
        self.past_key_values = past_key_values

    @torch.no_grad()
    def reduce_cache(self, start:int, end:int):
        assert end <= self.past_key_values._seen_tokens
        assert self.past_key_values._seen_tokens == self.past_key_values.key_cache[0].shape[2]

        # 1. reduce the value of seen_tokens
        self.past_key_values._seen_tokens -= (end-start)

        # 2. reduce the key_cache and value_cache
        bsz, n_head, q_length, head_dim = self.past_key_values.key_cache[0].shape
        new_q_length = q_length - (end-start)
        assert self.past_key_values._seen_tokens == new_q_length
        for layer_id in range(len(self.past_key_values.key_cache)):
            if new_q_length > end:
                # overlap
                # start:end
                for i in range(new_q_length-start):
                    self.past_key_values.key_cache[layer_id][:, :, start+i, :] = self.past_key_values.key_cache[layer_id][:, :, end+i, :]
                    self.past_key_values.value_cache[layer_id][:, :, start+i, :] = self.past_key_values.value_cache[layer_id][:, :, end+i, :]
            else:
                self.past_key_values.key_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.key_cache[layer_id][:, :, end:, :]
                self.past_key_values.value_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.value_cache[layer_id][:, :, end:, :]

            self.past_key_values.key_cache[layer_id] = \
                self.past_key_values.key_cache[layer_id][:, :, 0:new_q_length, :]
            self.past_key_values.value_cache[layer_id] = \
                self.past_key_values.value_cache[layer_id][:, :, 0:new_q_length, :]

    def __del__(self):
        del self.past_key_values.value_cache
        del self.past_key_values.key_cache
        del self.past_key_values
        torch.cuda.empty_cache()
        time.sleep(1)

class TokenUtils:

    """
    Generated Token Manager
    """

    def __init__(self, max_length:int, device:str, rolling_rope:bool):
        self.rolling_rope:bool = rolling_rope

        # self.input_ids[..., 0:self._seen_tokens]
        self.max_length:int = max_length
        self.input_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        self._seen_tokens:int = 0

        # complete token sequence
        self._whole_input_ids:List[int] = list()
        # dynamic token sequence (thoughts to be discarded.)
        self._current_input_ids:List[int] = list()

        # compression token will not be included
        self.show_prompt_input_ids:List[int] = list()
        self.show_output_input_ids:List[int] = list()

        self.position_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        # used when rolling_rope==True
        self.arange_ids:torch.Tensor = torch.arange(max_length, device=device)

        self._whole_position_ids:List[int] = list()
        self._current_position_ids:List[int] = list()

        # peak token
        self.max_token = 0

    def get_input_ids(self) -> torch.Tensor:
        return self.input_ids[..., 0:self._seen_tokens]

    def get_input_ids(self, start:int, end:int) -> torch.Tensor:
        if start >= 0 and end >= 0:
            return self.input_ids[..., start:end]
        else:
            if start < 0:
                start = self._seen_tokens + start
            if end < 0:
                end = self._seen_tokens + end
            return self.input_ids[..., start:end]
    
    def get_input_ids(self, idx:int) -> torch.Tensor:
        if idx >= 0:
            return self.input_ids[..., idx:idx+1]
        else:
            idx = self._seen_tokens + idx
            return self.input_ids[..., idx:idx+1]

    def get_position_ids(self) -> torch.Tensor:
        return self.position_ids[..., 0:self._seen_tokens]

    def set_input_id(self, idx:int):
        self.input_ids[..., self._seen_tokens] = idx
        self._whole_input_ids.append(idx)
        self._current_input_ids.append(idx)

        if self.rolling_rope:
            new_pos = len(self._current_position_ids)
        else:
            new_pos = self._whole_position_ids[-1] + 1
    
        self.position_ids[..., self._seen_tokens] = new_pos
        self._current_position_ids.append(new_pos)
        self._whole_position_ids.append(new_pos)

        self._seen_tokens += 1
        self.max_token = max(self.max_token, self._seen_tokens)

    def set_input_ids(self, input_ids:List[int], return_tensors:bool=False):
        assert isinstance(input_ids, list)
        _start = self._seen_tokens
        for i in range(len(input_ids)):
            self.input_ids[..., self._seen_tokens + i] = input_ids[i]
            if not self.rolling_rope:
                if len(self._whole_position_ids) == 0:
                    self.position_ids[..., self._seen_tokens + i] = 0
                    self._current_position_ids.append(0)
                    self._whole_position_ids.append(0)
                else:
                    self.position_ids[..., self._seen_tokens + i] = self.position_ids[0,self._seen_tokens + i - 1] + 1
                    self._current_position_ids.append(self._whole_position_ids[-1] + 1)
                    self._whole_position_ids.append(self._whole_position_ids[-1] + 1)
        _end = _start + len(input_ids)
        
        if self.rolling_rope:
            self.position_ids[..., 0:self._seen_tokens+len(input_ids)] = self.arange_ids[0:self._seen_tokens]
            self._current_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            self._whole_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            # assert False
        self._seen_tokens += len(input_ids)
        self._whole_input_ids.extend(input_ids)
        self._current_input_ids.extend(input_ids)
        self.max_token = max(self.max_token, self._seen_tokens)
        if return_tensors:
            # 这里切片的原因是前面用的都是kvcache，所以不需要重新考虑position_ids，只需解决当前position_ids
            return self.input_ids[..., _start:_end], self.position_ids[..., _start:_end]
 
    def reduce_input_ids(self, start:int, end:int):
        origin_length = self._seen_tokens
        self._seen_tokens -= (end-start)
        if self._seen_tokens > end:
            for i in range(self._seen_tokens-start):
                self.input_ids[..., start+i] = self.input_ids[..., end+i]
                self._current_input_ids[start+i] = self._current_input_ids[end+i]
        else:
            self.input_ids[..., start:self._seen_tokens] = self.input_ids[..., end:origin_length]
            self._current_input_ids[start:self._seen_tokens] = self._current_input_ids[end:origin_length]
        
        self._current_input_ids = self._current_input_ids[0:self._seen_tokens]

        if not self.rolling_rope:
            if self._seen_tokens > end:
                for i in range(self._seen_tokens-start):
                    self.position_ids[..., start+i] = self.position_ids[..., end+i]
                    self._current_position_ids[start+i] = self._current_position_ids[end+i]
            else:
                self.position_ids[..., start:self._seen_tokens] = self.position_ids[..., end:origin_length]
                self._current_position_ids[start:self._seen_tokens] = self._current_position_ids[end:origin_length]
        else:
            self._current_position_ids[start:self._seen_tokens] = [start+i for i in range(self._seen_tokens - start)]
            self.position_ids[..., 0:self._seen_tokens] = self.arange_ids[0:self._seen_tokens]
        self._current_position_ids = self._current_position_ids[0:self._seen_tokens]

    def reset(self):
        self._seen_tokens = 0
        self.max_token = 0
        self._whole_input_ids.clear()
        self._current_input_ids.clear()
        self._whole_position_ids.clear()
        self._current_position_ids.clear()
        self.show_prompt_input_ids.clear()
        self.show_output_input_ids.clear()

    def use_epl_for_compression(self, position_ids, indicator):
        # print("Use EPL for compression!")
        device = position_ids.device
        cot_start, cot_end, step, compression_count = indicator
        compressed_positions = []
        for k in range(compression_count):
            # k * step: 当前分段的起始
            # step / 2: 当前分段的中心偏移量
            # base_pos: 全局起始偏移
            # int(...): 向下取整得到整数索引
            center_offset = int(k * step + step / 2)
            
            # 计算最终位置
            pos = cot_start + center_offset
            compressed_positions.append(pos)

        # 额外拼接 <|splitter|> <|continue|> position id，cot_end是原本 <|o_1|> position id，现在赋值给 <|continue|>
        final_list = [cot_end - 1] + compressed_positions + [cot_end]
        position_ids_for_epl = torch.tensor([final_list], device=device, dtype=torch.long)
        return position_ids_for_epl

# ========== CORE CODE ==========
@torch.no_grad()
def _prefill_wo_prompt_compression(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt:str,
    system_prompt_list: List[str],
    question:str,
    question_list:List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    repetition_penalty:float=1.0,
) -> int:
    """
    prompt will not be compressed
    """
    assert compress_prompt is False

    past_key_values:DynamicCache = kv_utils.get_cache()

    # 1. concatenate prompt
    prompt:str = tokenizer.bos_token + comp_config.template_cfg['complete'].format(
        system=system_prompt, question=question
    )
    # 2. tokenize
    input_ids = tokenizer.tokenizer(
        prompt, return_tensors=None, add_special_tokens=False
    )['input_ids']
    token_utils.show_prompt_input_ids.extend(input_ids)
    token_utils.set_input_ids(input_ids)
    if DEBUG:
        DebugUtils.show_global_attention(
            tokenizer=tokenizer,
            attention_mask=attention_mask.squeeze().cpu().tolist(), 
            input_ids=input_ids,
            position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
            block=BLOCK,
            start_offset=None,
            end_offset=None,
            file_name="debug_global.png",
        )

    # 3. model.forward()
    model_output = model(
        input_ids=torch.as_tensor(
            [input_ids], device="cuda"
        ),
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
        output_hidden_states=True
    )
    # 4. get the generated token id
    predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
        model_output=model_output, idx=-1,token_utils=token_utils,repetition_penalty=repetition_penalty,tokenizer=tokenizer
    )

    return predicted_token_id, model_output.hidden_states[-1]
    
@torch.no_grad()
def _prefill_w_prompt_compression(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt:str,
    system_prompt_list: List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    repetition_penalty:float=1.0,
) -> int:
    """
    prompt will be compressed
    
    All the code here is not reused to ensure clarity and readability as much as possible.
    """
    assert compress_prompt == True

    input_ids:List[int] = list()
    indicator_list:List[List[int]] = list()

    prefix_prompt = comp_config.template_cfg['prefix']
    middle_prompt = comp_config.template_cfg['middle']
    suffix_prompt = comp_config.template_cfg['suffix']

    past_key_values = kv_utils.get_cache()

    # 1. set the input_ids and indicator_list
    if comp_config.prompt_comp_level == 'token' and comp_config.prompt_save_template == True:
        prefix_input_ids:List[int] = [tokenizer.bos_token_id] + tokenizer.tokenizer(
            prefix_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        middle_input_ids:List[int] = tokenizer.tokenizer(
            middle_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        suffix_input_ids:List[int] = tokenizer.tokenizer(
            suffix_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        system_prompt_input_ids:List[int] = tokenizer.tokenizer(
            system_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        question_input_ids:List[int] = tokenizer.tokenizer(
            question, return_tensors=None, add_special_tokens=False
        )['input_ids']

        step:int = comp_config.prompt_comp_step

        input_ids.extend(prefix_input_ids)
        token_utils.show_prompt_input_ids.extend(prefix_input_ids)
        for i in range(0, len(system_prompt_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(system_prompt_input_ids[i:i+step])
            token_utils.show_prompt_input_ids.extend(system_prompt_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        
        input_ids.extend(middle_input_ids)
        token_utils.show_prompt_input_ids.extend(middle_input_ids)
        for i in range(0, len(question_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(question_input_ids[i:i+step])
            token_utils.show_prompt_input_ids.extend(question_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )

        input_ids.extend(suffix_input_ids)
        token_utils.show_prompt_input_ids.extend(suffix_input_ids)
    elif comp_config.prompt_comp_level == 'token' and comp_config.prompt_save_template == False:
        prompt:str = tokenizer.bos_token + \
            prefix_prompt + system_prompt + \
                middle_prompt + question + \
                    suffix_prompt
        _input_ids:List[int] = tokenizer.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        step = comp_config.prompt_comp_step
        token_utils.show_prompt_input_ids.extend(_input_ids)

        for i in range(0, len(_input_ids), step):
            text_start = len(input_ids)
            input_ids.extend(_input_ids[i:i+step])
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        input_ids.append(comp_config.continue_token_id)
        token_utils.show_prompt_input_ids.append(comp_config.continue_token_id)
    elif comp_config.prompt_comp_level == 'sentence' and comp_config.prompt_save_template == True:
        prefix_input_ids:List[int] = [tokenizer.bos_token_id] + tokenizer.tokenizer(
            prefix_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        middle_input_ids:List[int] = tokenizer.tokenizer(
            middle_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']
        suffix_input_ids:List[int] = tokenizer.tokenizer(
            suffix_prompt, return_tensors=None, add_special_tokens=False
        )['input_ids']

        input_ids.extend(prefix_input_ids)
        token_utils.show_prompt_input_ids.extend(prefix_input_ids)
        for sent in system_prompt_list:
            text_start = len(input_ids)
            sent_input_ids = tokenizer.tokenizer(
                sent, return_tensors=None, add_special_tokens=False
            )['input_ids']
            input_ids.extend(sent_input_ids)
            token_utils.show_prompt_input_ids.extend(sent_input_ids)
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )
        
        input_ids.extend(middle_input_ids)
        token_utils.show_prompt_input_ids.extend(middle_input_ids)
        for sent in question_list:
            text_start = len(input_ids)
            sent_input_ids = tokenizer.tokenizer(
                sent, return_tensors=None, add_special_tokens=False
            )['input_ids']
            input_ids.extend(sent_input_ids)
            token_utils.show_prompt_input_ids.extend(sent_input_ids)
            text_end = len(input_ids)
            n_inst = 0
            comp_start = len(input_ids)
            input_ids.extend(comp_config.get_prompt_comp_token_id())
            comp_end = len(input_ids)
            n_cont = 0
            indicator_list.append(
                [text_start, text_end, n_inst, comp_start, comp_end, n_cont]
            )

        input_ids.extend(suffix_input_ids)
        token_utils.show_prompt_input_ids.extend(suffix_input_ids)
    elif comp_config.prompt_comp_level == 'sentence' and comp_config.prompt_save_template == False:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # print(input_ids)
    token_utils.set_input_ids(input_ids)

    # 2. generate attention_mask
    # [1, 1, length, length]
    attention_mask:torch.Tensor = attn_utils.create_prompt_attention(
        length=len(input_ids),
        indicator_list=indicator_list,
    )
    if DEBUG:
        # print(token_utils.get_position_ids().squeeze().cpu().tolist())
        DebugUtils.show_global_attention(
            tokenizer=tokenizer,
            attention_mask=attention_mask.squeeze().cpu().tolist(), 
            input_ids=input_ids,
            position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
            block=BLOCK,
            start_offset=None,
            end_offset=None,
            file_name="debug_global.png",
        )

    # 3. forward
    model_output = model(
        input_ids=torch.as_tensor(
            [input_ids], device="cuda"
        ),
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    if not prefill_compress:
        attn_utils.late_ajust_prompt_attention(
            length=len(input_ids),
            indicator_list=indicator_list,
        )
        if DEBUG:
            DebugUtils.show_global_attention(
                tokenizer=tokenizer,
                attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                # attention_mask=attention_mask.squeeze().cpu().tolist(), 
                input_ids=input_ids,
                position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                block=BLOCK,
                start_offset=None,
                end_offset=None,
                file_name="debug_global.png",
            )

    # 4. generate the new token id
    predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
        model_output=model_output, idx=-1,token_utils=token_utils,repetition_penalty=repetition_penalty,tokenizer=tokenizer
    )

    # 5. Update kv_Cache
    for indicator in indicator_list[::-1]:
        text_start, text_end, n_inst, comp_start, comp_end, n_cont = indicator
        start = text_start
        end = text_end
        assert n_inst == 0
        kv_utils.reduce_cache(start=start, end=end)
        token_utils.reduce_input_ids(start=start, end=end)

    return predicted_token_id

@torch.no_grad()
def prefill(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt:str,
    system_prompt_list: List[str],
    attention_config:Dict,
    prefill_compress:bool,
    compress_prompt:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    repetition_penalty:float=1.0,
) -> int:
    """
    There are several possible scenarios here:
        - Do not compress the prompt
        - Compress the prompt
            - Prefill stage mutually visible
            - Prefill stage mutually invisible
    """

    if not compress_prompt:
        return _prefill_wo_prompt_compression(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            system_prompt=system_prompt,
            system_prompt_list=system_prompt_list,
            question=question,
            question_list=question_list,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            compress_prompt=compress_prompt,
            repetition_penalty=repetition_penalty
        )
    else:
        return _prefill_w_prompt_compression(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            question=question,
            question_list=question_list,
            system_prompt=system_prompt,
            system_prompt_list=system_prompt_list,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            compress_prompt=compress_prompt,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            repetition_penalty=repetition_penalty
        )

@torch.no_grad()
def _token_level_generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    predicted_token_id:int,
    update_attention_method:str="global",
    repetition_penalty:float=1.0,
) -> Tuple[str, str]:
    
    assert update_attention_method in ["global", "local"]

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id
    explicit_token_cnt = 1
    output_comp_step = comp_config.output_comp_step

    global_start:int = len(token_utils._whole_input_ids)
    local_start:int = len(token_utils._current_input_ids)

    
    assert local_start == kv_utils.get_cache()._seen_tokens, \
        f"{local_start} == {kv_utils.get_cache()._seen_tokens}"
    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
        new_input_ids = [predicted_token_id]
        token_utils.show_output_input_ids.append(predicted_token_id)
        IS_COMP_MODE:bool = False

        # 1. construct attention_mask
        if explicit_token_cnt == output_comp_step:
            IS_COMP_MODE = True
            new_input_ids.extend(
                comp_config.get_output_comp_token_id()
            )
            new_input_ids.append(
                comp_config.continue_token_id
            )
            explicit_token_cnt = 1
            new_length = len(new_input_ids)
            if token_utils.max_length < token_utils._seen_tokens + new_length:
                # 超长时直接结束，避免后续 attention 切片维度不匹配
                break
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                indicator = [
                    global_start,
                    origin_length + 1,  # Because the last token has not been included yet.
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_global(
                    new_length=new_length,
                    indicator=indicator,
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                indicator = [
                    local_start,
                    origin_length + 1,
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id()),
                    1,
                ]
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=new_length,
                    indicator=indicator
                )
        else:
            explicit_token_cnt += 1
            if token_utils.max_length < token_utils._seen_tokens + 1:
                break
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                attention_mask = attn_utils.update_attention_global(
                    new_length=1,
                    indicator=None
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=1,
                    indicator=None
                )

        # The line below marks the end of the mask during the reduce process, 
        # primarily for the purpose of reduction.
        _local_mask_end = len(token_utils._current_input_ids) + 1
        # 2. position_ids and input_ids
        input_ids, position_ids = token_utils.set_input_ids(
            new_input_ids, return_tensors=True
        ) 
        if DEBUG:
            if update_attention_method == 'global':
                DebugUtils.show_global_attention(
                    tokenizer=tokenizer,
                    attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                    # attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._whole_input_ids),
                    position_ids=deepcopy(token_utils._whole_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_global.png",
                )
            else:
                DebugUtils.show_local_attention(
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._current_input_ids),
                    position_ids=deepcopy(token_utils._current_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_local_1.png",
                )

        # 3. generate new token
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
            position_ids=position_ids,
        )

        # 4. update kv cache
        if IS_COMP_MODE:
            start = local_start
            end = _local_mask_end

            kv_utils.reduce_cache(start=start, end=end)
            token_utils.reduce_input_ids(start=start, end=end)

            # update
            global_start:int = len(token_utils._whole_input_ids)
            local_start:int = len(token_utils._current_input_ids)

        # 5. get the new predicted_tokens
        predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
            model_output=model_output, idx=-1,token_utils=token_utils,repetition_penalty=repetition_penalty,tokenizer=tokenizer
        )
        new_token_counters += 1
    
    token_utils.show_output_input_ids.append(predicted_token_id)
    # return tokenizer.decode(token_utils._whole_input_ids)
    return tokenizer.decode(token_utils.show_prompt_input_ids), tokenizer.decode(token_utils.show_output_input_ids)

@torch.no_grad()
def _sentence_level_generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    predicted_token_id:int,
    update_attention_method:str="global",
    use_EPL:bool=False,
    repetition_penalty:float=1.0,
) -> Tuple[str,str]:
    assert update_attention_method in ["global", "local"]

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id
    # eos_token_id = None
    output_comp_step = comp_config.output_comp_step

    global_start:int = len(token_utils._whole_input_ids) # 整个输入完整的序列，包括cot+压缩token？
    local_start:int = len(token_utils._current_input_ids) # 应该是当前输入的序列，意味着去掉了cot，但是有压缩token

    use_compression_all_count = 0
    debug_count = 0
    cot_start = global_start
    cot_end = 0
    van_cot_start = global_start
    delete_delay_steps = max(0, int(comp_config.output_cfg.get("delete_delay_steps", 0)))
    pending_delete_ranges: List[Tuple[int, int]] = []

    def _shift_ranges_after_delete(ranges: List[Tuple[int, int]], start: int, end: int) -> List[Tuple[int, int]]:
        shift = end - start
        new_ranges: List[Tuple[int, int]] = []
        for s, e in ranges:
            if e <= start:
                new_ranges.append((s, e))
            elif s >= end:
                new_ranges.append((s - shift, e - shift))
            else:
                continue
        return new_ranges

    assert local_start == kv_utils.get_cache()._seen_tokens, \
        f"{local_start} == {kv_utils.get_cache()._seen_tokens}"
    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
        new_input_ids = [predicted_token_id]
        IS_COMP_MODE:bool = False
        token_utils.show_output_input_ids.append(predicted_token_id)
        # print(tokenizer.decode(token_utils.show_output_input_ids))
        # 1. construct attention_mask
        if predicted_token_id == comp_config.split_token_id:
            IS_COMP_MODE = True

            # print(tokenizer.decode(token_utils._current_input_ids))

            last_pos_raw = int(token_utils.get_position_ids()[0, -1].item())
            if use_EPL:
                last_pos_epl = last_pos_raw - use_compression_all_count
                cot_length = last_pos_epl + 2 - cot_start
            else:
                cot_length = last_pos_raw + 2 - van_cot_start
            
            new_input_ids.extend(
                comp_config.get_output_comp_token_id(cot_length=cot_length)
            )
            new_input_ids.append(
                comp_config.continue_token_id
            )
            new_length = len(new_input_ids)
            if token_utils.max_length < token_utils._seen_tokens + new_length:
                # 先检查长度，再构造 attention，避免局部切片越界导致 shape mismatch
                break
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                indicator = [
                    global_start,
                    origin_length + 1,  # the last token has not been included yet.
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id(cot_length=cot_length)),
                    1,
                ]
                attention_mask = attn_utils.update_attention_global(
                    new_length=new_length,
                    indicator=indicator,
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                indicator = [
                    local_start,
                    origin_length + 1,
                    0,
                    origin_length + 1,
                    origin_length + 1 + len(comp_config.get_output_comp_token_id(cot_length=cot_length)),
                    1,
                ]
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=new_length,
                    indicator=indicator
                )
        else:
            if token_utils.max_length < token_utils._seen_tokens + 1:
                break
            if update_attention_method == 'global':
                origin_length = len(token_utils._whole_input_ids)
                attention_mask = attn_utils.update_attention_global(
                    new_length=1,
                    indicator=None
                )
            else:
                origin_length = len(token_utils._current_input_ids)
                attention_mask = attn_utils.update_attention_local(
                    origin_length=origin_length,
                    new_length=1,
                    indicator=None
                )

        # The line below marks the end of the mask during the reduce process, 
        # primarily for the purpose of reduction.
        _local_mask_end = len(token_utils._current_input_ids) + 1
        # 2. position_ids and input_ids
        # ......The code is beautifully repeated......
        input_ids, position_ids = token_utils.set_input_ids(new_input_ids, return_tensors=True)
        if IS_COMP_MODE:
            van_cot_start = int(position_ids[0][-1].item()) + 1
        if use_EPL:
            position_ids = position_ids - use_compression_all_count

        if use_EPL and IS_COMP_MODE:
            # 这里 position_ids 是 '<|splitter|><|o_0|><|o_1|><|o_2|><|o_3|><|o_4|><|o_5|><|o_6|><|o_7|><|o_8|><|continue|>' 对应的正常位置编码
            # position_ids[0][0] 是 <|splitter|> position id
            # cot_start 是 cot first token position id，cot_end 是 <|o_0|> position id
            # cot_end - cot_start 是算上<|splitter|>的 cot 长度，也就是 n_abandoned
            # 训练时 <|splitter|> 也是算在 n_abandoned 之内的
            cot_end = int(position_ids[0][0].item()) + 1
            step = (cot_end - cot_start) / len(comp_config.get_output_comp_token_id(cot_length=(cot_end - cot_start)))
            indicator = [
                    cot_start, # cot first token position id
                    cot_end, # <|o_1|> position id
                    step, # 压缩步长
                    len(comp_config.get_output_comp_token_id(cot_length=(cot_end - cot_start))) # 压缩token数量
                ]
            # 更新cot位置
            # 这里算上 <|continue|>，对应下一段 cot 的 first token position id
            position_ids = token_utils.use_epl_for_compression(position_ids, indicator)
            use_compression_all_count += len(comp_config.get_output_comp_token_id(cot_length=(cot_end - cot_start)))
            cot_start = cot_end + 1
        if DEBUG:
            if update_attention_method == 'global':
                DebugUtils.show_global_attention(
                    tokenizer=tokenizer,
                    attention_mask=attn_utils.cur_attn[0:attn_utils.last_idx, 0:attn_utils.last_idx].cpu().tolist(),
                    # attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._whole_input_ids),
                    position_ids=deepcopy(token_utils._whole_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_global.png",
                )
            else:
                DebugUtils.show_local_attention(
                    tokenizer=tokenizer,
                    attention_mask=attention_mask.squeeze(dim=0).squeeze(dim=0).cpu().tolist(), 
                    input_ids=deepcopy(token_utils._current_input_ids),
                    position_ids=deepcopy(token_utils._current_position_ids),
                    # position_ids=token_utils.get_position_ids().squeeze().cpu().tolist(),
                    block=BLOCK,
                    start_offset=None,
                    end_offset=None,
                    file_name="debug_local_1.png",
                )

        # 3. generate new token(本来对这里有疑问的，为什么attention_mask是全0？因为当前只传入了一个token，前面的是kvcache，所以前面正常应该都能看到？
        # 突然看到非压缩时indicator都是None，貌似合理了)
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
            position_ids=position_ids,
        )
        # 4. update kv cache
        if IS_COMP_MODE:
            pending_delete_ranges.append((local_start, _local_mask_end))

            # 先把下一段 thought 的起点移动到当前末尾，再按延迟窗口执行实际删除。
            global_start = len(token_utils._whole_input_ids)
            local_start = len(token_utils._current_input_ids)

            while len(pending_delete_ranges) > delete_delay_steps:
                start, end = pending_delete_ranges.pop(0)
                kv_utils.reduce_cache(start=start, end=end)
                token_utils.reduce_input_ids(start=start, end=end)
                # 虚拟左移
                pending_delete_ranges = _shift_ranges_after_delete(pending_delete_ranges, start, end)
                local_start = len(token_utils._current_input_ids)

        # 5. get new predicted_tokens 151665
        predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
            model_output=model_output, idx=-1,token_utils=token_utils,repetition_penalty=repetition_penalty,tokenizer=tokenizer
        )
        debug_count += 1
        new_token_counters += 1

    token_utils.show_output_input_ids.append(predicted_token_id)
    return tokenizer.decode(token_utils.show_prompt_input_ids), tokenizer.decode(token_utils.show_output_input_ids)



# mtp register generate最快实现版本
# 无法和普通sentence_level_generate做到完全一致，因为：
# 1.数值精度
# 2.多token和单token存在计算差异计算 
# 导致某些位置（例如 to 和 for）logit差异过小而选择了不同的token
@torch.no_grad()
def _sentence_level_mtp_register_generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    attn_utils:AttentionUtils,
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    predicted_token_id:int,
    last_hidden_state: torch.Tensor,
    update_attention_method:str="global",
    use_EPL:bool=False,
    repetition_penalty:float=1.0
) -> Tuple[str,str]:
    assert update_attention_method in ["global", "local"]

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id
    global_start:int = len(token_utils._whole_input_ids)
    local_start:int = len(token_utils._current_input_ids)

    use_compression_all_count = 0
    cot_start = global_start
    # 当 verify 阶段提前确认下一个 control token 为 split 时，记录其位置用于下一轮 cot_length 计算
    pending_split_pos = None
    # van_cot_start = global_start

    # register token count 默认为 0，因为训练时 mtp register offset = random.randint(0, 2)
    if comp_config.mtp_cfg:
        register_token_count = int(comp_config.mtp_cfg.get("max_offset", 0))
        register_token_count = max(0, register_token_count)
    else:
        register_token_count = 0

    # 在首次进入 split 分支前也需要一个可用 position_ids 基准
    position_ids = token_utils.get_position_ids()[:, -1:]

    assert local_start == kv_utils.get_cache()._seen_tokens, \
        f"{local_start} == {kv_utils.get_cache()._seen_tokens}"


    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
            IS_COMP_MODE:bool = False
            token_utils.show_output_input_ids.append(predicted_token_id)
            step_input_ids = [predicted_token_id]

            # 1. construct attention_mask / step tokens
            if predicted_token_id == comp_config.split_token_id:
                IS_COMP_MODE = True

                if use_EPL:
                    if pending_split_pos is not None:
                        # pending_split_pos 是 <|splitter|> position id
                        cot_length = pending_split_pos + 1 - cot_start
                        pending_split_pos = None
                    else:
                        last_pos_raw = int(token_utils.get_position_ids()[0, -1].item())
                        last_pos_epl = last_pos_raw - use_compression_all_count
                        cot_length = last_pos_epl + 2 - cot_start
                else:
                    last_pos_raw = int(token_utils.get_position_ids()[0, -1].item())
                    cot_length = last_pos_raw + 2 - cot_start

                step_input_ids.extend(
                    comp_config.get_output_comp_token_id(cot_length=cot_length)
                )
                step_input_ids.append(
                    comp_config.continue_token_id
                )
                if register_token_count > 0:
                    step_input_ids.extend([comp_config.register_token_id] * register_token_count)
                # 须在构造 attention mask 之前检查，否则 delta_attn 列切片可能被截断导致 shape 报错
                if token_utils.max_length < len(step_input_ids) + token_utils._seen_tokens:
                    break
                new_length = len(step_input_ids)
                if update_attention_method == 'global':
                    origin_length = len(token_utils._whole_input_ids)
                    indicator = [
                        global_start,
                        origin_length + 1,  # the last token has not been included yet.
                        0,
                        origin_length + 1,
                        origin_length + 1 + len(comp_config.get_output_comp_token_id(cot_length=cot_length)),
                        1,
                    ]
                    attention_mask = attn_utils.update_attention_global(
                        new_length=new_length,
                        indicator=indicator,
                    )
                else:
                    origin_length = len(token_utils._current_input_ids)
                    indicator = [
                        local_start,
                        origin_length + 1,
                        0,
                        origin_length + 1,
                        origin_length + 1 + len(comp_config.get_output_comp_token_id(cot_length=cot_length)),
                        1,
                    ]
                    attention_mask = attn_utils.update_attention_local_for_mtp_register(
                        origin_length=origin_length,
                        new_input_ids=step_input_ids,
                        register_token_id=comp_config.register_token_id,
                        indicator=indicator,
                    )
            else:
                # register token 作为辅助预测 token，推理时仅临时插入，随后从 cache 中删除
                if register_token_count > 0:
                    step_input_ids.extend([comp_config.register_token_id] * register_token_count)

                if token_utils.max_length < len(step_input_ids) + token_utils._seen_tokens:
                    break
                new_length = len(step_input_ids)
                if update_attention_method == 'global':
                    origin_length = len(token_utils._whole_input_ids)
                    attention_mask = attn_utils.update_attention_global(
                        new_length=new_length,
                        indicator=None
                    )
                else:
                    origin_length = len(token_utils._current_input_ids)
                    attention_mask = attn_utils.update_attention_local_for_mtp_register(
                        origin_length=origin_length,
                        new_input_ids=step_input_ids,
                        register_token_id=comp_config.register_token_id,
                        indicator=None,
                    )

            # The line below marks the end of the mask during the reduce process, 
            # primarily for the purpose of reduction.
            _local_mask_end = len(token_utils._current_input_ids) + 1
            # 2. position_ids and input_ids
            input_ids, position_ids = token_utils.set_input_ids(step_input_ids, return_tensors=True)
            # if IS_COMP_MODE:
            #     van_cot_start = int(position_ids[0][-1].item()) + 1
            if use_EPL:
                position_ids = position_ids - use_compression_all_count

            if use_EPL and IS_COMP_MODE:
                # 这里 position_ids 是 '<|splitter|><|o_0|><|o_1|><|o_2|><|o_3|><|o_4|><|o_5|><|o_6|><|o_7|><|o_8|><|continue|><|register_1|><|register_2|>...' 对应的正常位置编码
                # cot_start 是 cot first token position id，cot_end 是 <|o_0|> position id
                # cot_end - cot_start 是算上<|splitter|>的 cot 长度，也就是 n_abandoned
                # 训练时 <|splitter|> 也是算在 n_abandoned 之内的

                # step1: 计算'<|splitter|><|o_0|><|o_1|><|o_2|><|o_3|><|o_4|><|o_5|><|o_6|><|o_7|><|o_8|><|continue|>'的位置编码
                # position_ids[0][0] 是 <|splitter|> position id
                cur_split_end = int(position_ids[0][0].item()) + 1
                step = (cur_split_end - cot_start) / len(comp_config.get_output_comp_token_id(cot_length=(cur_split_end - cot_start)))
                indicator = [
                        cot_start, # cot first token position id
                        cur_split_end, # <|o_1|> position id
                        step, # 压缩步长
                        len(comp_config.get_output_comp_token_id(cot_length=(cur_split_end - cot_start))) # 压缩token数量
                    ]
                # 更新cot位置
                # 这里算上 <|continue|>，对应下一段 cot 的 first token position id
                position_ids = token_utils.use_epl_for_compression(position_ids, indicator)
                use_compression_all_count += len(comp_config.get_output_comp_token_id(cot_length=(cur_split_end - cot_start)))
                cot_start = cur_split_end + 1
                
                # step2: 这里补齐register token位置编码
                if register_token_count > 0:
                    last_pos = int(position_ids[0, -1].item())
                    register_position_ids = torch.arange(
                        last_pos + 1,
                        last_pos + 1 + register_token_count,
                        device=position_ids.device,
                        dtype=position_ids.dtype,
                    ).unsqueeze(0)
                    position_ids = torch.cat([position_ids, register_position_ids], dim=1)
            elif IS_COMP_MODE:
                # 非 EPL 下使用原始坐标更新 cot_start
                cur_split_end = int(position_ids[0][0].item()) + 1
                cot_start = cur_split_end + 1

            assert input_ids.shape[1] == position_ids.shape[1], \
                f"shape mismatch: input_ids={input_ids.shape}, position_ids={position_ids.shape}"

            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=kv_utils.get_cache(),
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
            )

            if IS_COMP_MODE:
                # compression 分支先删除被压缩的原文，再删除临时 register
                start = local_start
                end = _local_mask_end
                kv_utils.reduce_cache(start=start, end=end)
                token_utils.reduce_input_ids(start=start, end=end)
                if register_token_count > 0:
                    reg_start = kv_utils.get_cache()._seen_tokens - register_token_count
                    reg_end = kv_utils.get_cache()._seen_tokens
                    kv_utils.reduce_cache(start=reg_start, end=reg_end)
                    token_utils.reduce_input_ids(start=reg_start, end=reg_end)
                global_start = len(token_utils._whole_input_ids)
                local_start = len(token_utils._current_input_ids)
            else:
                # 非压缩分支只删除临时 register
                if register_token_count > 0:
                    reg_start = _local_mask_end
                    reg_end = _local_mask_end + register_token_count
                    kv_utils.reduce_cache(start=reg_start, end=reg_end)
                    token_utils.reduce_input_ids(start=reg_start, end=reg_end)

            # 基于 [anchor + register...] 的 logits 产出 draft tokens（压缩/非压缩共用）
            draft_len = register_token_count + 1
            draft_tokens: List[int] = []
            for offset in range(draft_len):
                idx = -(draft_len - offset)
                draft_tokens.append(
                    InferenceUtils.get_predicted_token_ids(
                        model_output=model_output,
                        idx=idx,
                        token_utils=token_utils,
                        repetition_penalty=repetition_penalty,
                        tokenizer=tokenizer,
                        # 模拟逐 token 草稿生成：当前位置前的 draft 前缀也参与 repetition 集合
                        extra_generated_ids=draft_tokens,
                    )
                )

            # 若首个 draft 已是控制 token，直接进入下一轮分支处理
            if draft_tokens[0] == comp_config.split_token_id or draft_tokens[0] == eos_token_id:
                if draft_tokens[0] == comp_config.split_token_id:
                    tail_raw = int(token_utils.get_position_ids()[0, -1].item())
                    pending_split_pos = tail_raw - use_compression_all_count + 1 if use_EPL else tail_raw + 1
                predicted_token_id = draft_tokens[0]
                accepted_count = 1
            # 无 register 时退化为普通单步预测
            elif len(draft_tokens) == 1:
                predicted_token_id = draft_tokens[0]
                accepted_count = 1
            else:
                # 验证阶段：喂入 draft 序列，验证后续 token 的一致性，并获取“下一个 token”候选
                verify_new_length = len(draft_tokens)
                if token_utils.max_length < verify_new_length + token_utils._seen_tokens:
                    predicted_token_id = draft_tokens[0]
                    accepted_count = 1
                else:
                    if update_attention_method == 'global':
                        verify_attention_mask = attn_utils.update_attention_global(
                            new_length=verify_new_length,
                            indicator=None
                        )
                    else:
                        verify_attention_mask = attn_utils.update_attention_local(
                            origin_length=len(token_utils._current_input_ids),
                            new_length=verify_new_length,
                            indicator=None
                        )

                    verify_input_ids, verify_position_ids = token_utils.set_input_ids(
                        draft_tokens, return_tensors=True
                    )
                    if use_EPL:
                        verify_position_ids = verify_position_ids - use_compression_all_count

                    verify_output = model(
                        input_ids=verify_input_ids,
                        attention_mask=verify_attention_mask,
                        past_key_values=kv_utils.get_cache(),
                        use_cache=True,
                        return_dict=True,
                        position_ids=verify_position_ids,
                    )

                    verify_preds: List[int] = []
                    for pos in range(verify_new_length):
                        idx = -(verify_new_length - pos)
                        verify_preds.append(
                            InferenceUtils.get_predicted_token_ids(
                                model_output=verify_output,
                                idx=idx,
                                token_utils=token_utils,
                                repetition_penalty=repetition_penalty,
                                tokenizer=tokenizer,
                                # 模拟逐 token 解码：当前位置之前的 draft 前缀也纳入 repetition 集合
                                extra_generated_ids=draft_tokens[:pos+1],
                            )
                        )

                    # 固定接受第一个token；验证 d2..dk 是否匹配
                    accepted_len = 1
                    for i in range(verify_new_length - 1):
                        if verify_preds[i] == draft_tokens[i + 1]:
                            accepted_len += 1
                        else:
                            break

                    # 移除未通过验证的尾部 token
                    if accepted_len < verify_new_length:
                        trim = verify_new_length - accepted_len
                        trim_start = kv_utils.get_cache()._seen_tokens - trim
                        trim_end = kv_utils.get_cache()._seen_tokens
                        kv_utils.reduce_cache(start=trim_start, end=trim_end)
                        token_utils.reduce_input_ids(start=trim_start, end=trim_end)

                    # 额外吞并已验证通过的 token（除了当前已写入的第一个 token）
                    # 注意：若遇到 split/eos，不在本轮直接吞并，交给下一轮走原有分支逻辑（含 indicator）
                    control_token = None
                    extra_accepted: List[int] = []
                    for tok in draft_tokens[1:accepted_len]:
                        if tok == comp_config.split_token_id or tok == eos_token_id:
                            control_token = tok
                            break
                        extra_accepted.append(tok)

                    # 若遇到控制token，需要把“多接受但不该吞并”的尾部从 cache 里回滚掉
                    effective_accepted_len = 1 + len(extra_accepted)
                    if effective_accepted_len < accepted_len:
                        trim = accepted_len - effective_accepted_len
                        trim_start = kv_utils.get_cache()._seen_tokens - trim
                        trim_end = kv_utils.get_cache()._seen_tokens
                        kv_utils.reduce_cache(start=trim_start, end=trim_end)
                        token_utils.reduce_input_ids(start=trim_start, end=trim_end)
                        accepted_len = effective_accepted_len
                    
                    token_utils.show_output_input_ids.append(draft_tokens[0])
                    for tok in extra_accepted:
                        token_utils.show_output_input_ids.append(tok)

                    accepted_count = accepted_len
                    if control_token is not None:
                        if control_token == comp_config.split_token_id:
                            tail_raw = int(token_utils.get_position_ids()[0, -1].item())
                            pending_split_pos = tail_raw - use_compression_all_count + 1 if use_EPL else tail_raw + 1
                        predicted_token_id = control_token
                    else:
                        # 下一轮从“最后一个已接受 token 的下一 token 预测”继续
                        predicted_token_id = verify_preds[accepted_len - 1]

            new_token_counters += accepted_count

    token_utils.show_output_input_ids.append(predicted_token_id)
    # return tokenizer.decode(token_utils._whole_input_ids)
    return tokenizer.decode(token_utils.show_prompt_input_ids), tokenizer.decode(token_utils.show_output_input_ids)




@torch.no_grad()
def generate(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    comp_config: Config,
    question:str,
    question_list:List[str],
    system_prompt: str,
    system_prompt_list: List[str],
    max_new_tokens: int,
    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    compress_prompt:bool,
    attn_utils: AttentionUtils,
    token_utils: TokenUtils,
    update_attention_method:str,
    use_EPL:bool=False,
    repetition_penalty:float=1.0,
    spec_decode:bool=False,
) -> Tuple[str,str]:

    assert update_attention_method in ['global', 'local'], update_attention_method
    kv_utils = KVUtils()

    # 1. prefill
    predicted_token_id, last_hidden_state = prefill(
        model=model,
        tokenizer=tokenizer,
        comp_config=comp_config,
        question=question,
        question_list=question_list,
        system_prompt=system_prompt,
        system_prompt_list=system_prompt_list,
        attention_config=attention_config,
        prefill_compress=prefill_compress,
        compress_prompt=compress_prompt,
        attn_utils=attn_utils,
        kv_utils=kv_utils,
        token_utils=token_utils,
        repetition_penalty=repetition_penalty
    )

    # 2. auto-regressive generation
    if comp_config.output_comp_level == 'token':
        prompt, output = _token_level_generate(
            model=model,
            tokenizer=tokenizer,
            comp_config=comp_config,
            max_new_tokens=max_new_tokens,
            attention_config=attention_config,
            prefill_compress=prefill_compress,
            exclude_continue=exclude_continue,
            attn_utils=attn_utils,
            kv_utils=kv_utils,
            token_utils=token_utils,
            predicted_token_id=predicted_token_id,
            update_attention_method=update_attention_method,
            repetition_penalty=repetition_penalty
        )
    elif comp_config.output_comp_level == 'sentence':
        if comp_config.mtp_cfg and spec_decode:
            # mtp register generation
            prompt, output = _sentence_level_mtp_register_generate(
                model=model,
                tokenizer=tokenizer,
                comp_config=comp_config,
                max_new_tokens=max_new_tokens,
                attention_config=attention_config,
                prefill_compress=prefill_compress,
                exclude_continue=exclude_continue,
                attn_utils=attn_utils,
                kv_utils=kv_utils,
                token_utils=token_utils,
                predicted_token_id=predicted_token_id,
                last_hidden_state=last_hidden_state,
                update_attention_method=update_attention_method,
                use_EPL=use_EPL,
                repetition_penalty=repetition_penalty
            )
        else:
            prompt, output = _sentence_level_generate(
                model=model,
                tokenizer=tokenizer,
                comp_config=comp_config,
                max_new_tokens=max_new_tokens,
                attention_config=attention_config,
                prefill_compress=prefill_compress,
                exclude_continue=exclude_continue,
                attn_utils=attn_utils,
                kv_utils=kv_utils,
                token_utils=token_utils,
                predicted_token_id=predicted_token_id,
                update_attention_method=update_attention_method,
                use_EPL=use_EPL,
                repetition_penalty=repetition_penalty,
            )
            
    
    del kv_utils
    return prompt, output

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_tag', type=str)
    parser.add_argument('--model_short_tag', type=str, default=None)
    parser.add_argument('--ckpt', type=int)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--compress_config', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--output_tag', type=str)
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama'])
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--bos_token', type=str)
    parser.add_argument('--eos_token', type=str)

    # specialized argument
    # ==============================
    parser.add_argument('--rolling_rope', type=str2bool)
    parser.add_argument('--diagonal', type=str2bool)
    parser.add_argument('--bi_directional', type=str2bool)
    parser.add_argument('--see_current', type=str2bool)
    parser.add_argument('--exclude_continue', type=str2bool)
    parser.add_argument('--output_compress_instruction', type=str)
    parser.add_argument('--prefill_compress', type=str2bool, default=True)
    parser.add_argument('--compress_prompt', type=str2bool, default=True)
    parser.add_argument('--update_attention_method', type=str, choices=['global', 'local'])
    # ==============================

    parser.add_argument('--split_size', type=int)
    # start from 1
    parser.add_argument('--index', type=int)        
    parser.add_argument('--use_EPL', type=str2bool, default=False)
    parser.add_argument('--spec_decode', type=str2bool, default=False)
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',  # 允许多个值
        default=['mmlu', 'gsm8k', 'gpqa', 'bbh'],  # 默认全部
        choices=['mmlu', 'gsm8k', 'gpqa', 'bbh'],
        help='Datasets to evaluate. Can specify multiple: --datasets mmlu gsm8k'
    )

    args = parser.parse_args()
    return args

def get_model_and_tokenizer(
    args, 
    comp_config:Config
) -> Tuple[
    Union[Qwen2ForCausalLM, LlamaForCausalLM],
    Tokenizer
]:
    if args.model_path == None or args.model_path == "":
        model_path = f"output/{args.model_tag}/checkpoint-{args.ckpt}"
    else:
        model_path = args.model_path
    print(f"load model from `{model_path}` ...")
    special_token_list:List[str] = list()
    tokenizer: Tokenizer = Tokenizer(
        tokenizer_path=args.tokenizer_path if args.tokenizer_path != None else model_path,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )

    for token in comp_config.special_token_name_list:
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)

    if args.model_type.lower() == 'qwen':
        model = Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif args.model_type.lower() == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    comp_config.convert2id(tokenizer)
    
    return model, tokenizer

@torch.no_grad()
def eval_dataset(
    model:Union[Qwen2ForCausalLM, LlamaForCausalLM],
    tokenizer:Tokenizer,
    reader:Reader,
    comp_config:Config,
    output_file:str,
    max_new_tokens:int,
    repetition_penalty:float,
    max_prompt_len:int,
    device:str,
    dtype,
    max_comp_size:int,

    attention_config:Dict,
    prefill_compress:bool,
    exclude_continue:bool,
    compress_prompt:bool,

    update_attention_method:str,
    rolling_rope:bool,

    dataset_name:str,
    split_size:int=None,
    index:int=None,
    use_EPL:bool=False,
    spec_decode:bool=False,
):

    if split_size != None and index != None:
        assert index > 0
        assert index <= split_size
        step = len(reader) // split_size
        start = (index-1) * step
        end = index * step
        if index == split_size:
            end = len(reader)
    elif split_size == None and index == None:
        start = 0
        end = len(reader)
    else:
        assert False

    print(f"Starting test for `{dataset_name}`. Total size is {len(reader)}. Now, {index}/{split_size}: {start}-{end}")

    pbar = tqdm(total=end-start)
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            pass

    # ===== CONTINUE =====
    data_copy_list = list()
    _id_list = list()
    if os.path.isfile(output_file):
        with jsonlines.open(output_file, 'r') as f:
            for item in f:
                _id_list.append(item['idx'])
                data_copy_list.append(item)
        if len(_id_list) != 0:
            assert _id_list[0] == start
            start = _id_list[-1] + 1
    # ====================

    total = 0
    acc = 0

    attn_utils = AttentionUtils(
        max_length=max_new_tokens + max_prompt_len,
        device=device,
        dtype=dtype,
        attention_config=attention_config,
        prefill_compress=prefill_compress,
        max_comp_size=max_comp_size,
        n_inst=0,
        n_continue=1
    )
    token_utils = TokenUtils(
        max_length=max_new_tokens + max_prompt_len,
        device=device,
        rolling_rope=rolling_rope
    )

    with jsonlines.open(output_file, 'w') as writer:
        # ===== CONTINUE =====
        for item in data_copy_list:
            writer.write(item)
            total += 1
            if item['acc_state'] == True:
                acc += 1
            model_answer = item['model_answer']
            gt_answer = item['gt_answer']
            acc_state = item['acc_state']
            input_len = item['input_len']
            output_len = item['output_len']
            max_token = item['max_token']
            comp_pattern = item['comp_pattern']

            pbar.set_description(f"model:`{model_answer}`; gt:`{gt_answer}`; correct:{acc_state}; acc: {acc}/{total}={round(acc/total, 5)}; input: {input_len}; output: {output_len}; `{comp_pattern}`; max_token: {max_token}")
            pbar.update(1)
        # ====================
        
        for i in range(start, end):
            total += 1
            question:str = reader.get_prompt(idx=i)
            question_list:List[str] = reader.get_prompt_list(idx=i)
            system_prompt:str = reader.get_system_prompt()
            system_prompt_list:List[str] = reader.get_system_prompt_list()

            start_time = time.time()
            prompt, output = generate(
                question=question,
                question_list=question_list,
                system_prompt=system_prompt,
                system_prompt_list=system_prompt_list,

                model=model,
                tokenizer=tokenizer,
                comp_config=comp_config,
                attention_config=attention_config,
                max_new_tokens=max_new_tokens,
                prefill_compress=prefill_compress,
                exclude_continue=exclude_continue,
                compress_prompt=compress_prompt,

                attn_utils=attn_utils,
                token_utils=token_utils,
                update_attention_method=update_attention_method,
                use_EPL=use_EPL,
                repetition_penalty=repetition_penalty,
                spec_decode=spec_decode,
            )
            end_time = time.time()
            input_len:int = len(token_utils.show_prompt_input_ids)
            output_len:int = len(token_utils.show_output_input_ids)
            
            model_answer:str = reader.extract_answer(output)
            gt_answer:str = reader.get_answer(i)
            acc_state, comp_pattern = reader.compare_answer(model_answer, gt_answer, i)
            if acc_state == True:
                acc += 1
                
            writer.write(dict(
                idx=i,
                model_answer=model_answer,
                gt_answer=gt_answer,
                acc_state=acc_state,
                output=output,
                prompt=prompt,
                input_len=input_len,
                output_len=output_len,
                infer_time=end_time-start_time,
                comp_pattern=comp_pattern,
                max_token=token_utils.max_token
            ))
            pbar.set_description(f"model:`{model_answer}`; gt:`{gt_answer}`; correct:{acc_state}; acc: {acc}/{total}={round(acc/total, 5)}; input: {input_len}; output: {output_len}; `{comp_pattern}`; max_token: {token_utils.max_token}")
            token_utils.reset()
            attn_utils.reset()
            pbar.update(1)
            torch.cuda.empty_cache()
    pbar.close()

def main():
    device = 'cuda'
    max_comp_size = 15
    max_prompt_len = 1000
    max_comp_size = 300
    max_prompt_len = 1100
    dtype = torch.bfloat16

    args = get_parser()
    if args.model_short_tag == None:
        args.model_short_tag = args.model_tag
    print(args)

    comp_config = Config.from_file(args.compress_config)
    attention_config = dict(
        diagonal=args.diagonal,
        bi_attention=args.bi_directional,
        see_current=args.see_current,
        prefill_compress=args.prefill_compress,
        exclude_continue=args.exclude_continue
    )


    model, tokenizer = get_model_and_tokenizer(
        args, comp_config
    )

    all_tasks = {
        "mmlu": MMLUReader(),
        "gsm8k": GSM8KReader(),
        "gpqa": GPQAReader(),
        "bbh": BBHReader(),
    }

    task_list = [(all_tasks[name], name) for name in args.datasets if name in all_tasks]
    
    # 打印要评估的数据集
    print(f"\n{'='*60}")
    print(f"Evaluating datasets: {', '.join(args.datasets)}")
    print(f"{'='*60}\n")

    for reader, name in task_list:
        eval_dataset(
            model=model,
            tokenizer=tokenizer,
            reader=reader,
            comp_config=comp_config,
            output_file=f"{args.output_tag}/{name}/{args.index}_{name}.jsonl",
            # output_file=f"inference_results/{args.output_tag}/{name}_{args.model_tag}_{args.ckpt}.jsonl",
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            max_prompt_len=max_prompt_len,
            device=device,
            dtype=dtype,
            max_comp_size=max_comp_size,

            attention_config=attention_config,
            prefill_compress=args.prefill_compress,
            exclude_continue=args.exclude_continue,
            compress_prompt=args.compress_prompt,

            update_attention_method=args.update_attention_method,
            rolling_rope=args.rolling_rope,

            dataset_name=name,
            split_size=args.split_size,
            index=args.index,
            use_EPL=args.use_EPL,
            spec_decode=args.spec_decode
        )

if __name__ == '__main__':
    main()
