import dataclasses
from functools import partial
from acdc.docstring.utils import AllDataThings
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict, Callable
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
from acdc.acdc_utils import (
    MatchNLLMetric,
    make_nd_dict,
    shuffle_tensor,
)
from acdc.loadmodel import load_model 
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from transformer_lens import HookedTransformer
from acdc.acdc_utils import kl_divergence, negative_log_probs

def get_model(device,model_name):

    return load_model(model_name,device)

def get_validation_data(num_examples=None, seq_len=None, device=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_good_induction_candidates(num_examples=None, seq_len=None, device=None):
    """Not needed?"""
    good_induction_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
    )
    good_induction_candidates = torch.load(good_induction_candidates_fname, map_location=device)

    if num_examples is None:
        return good_induction_candidates
    else:
        return good_induction_candidates[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None, device=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]

'''
def get_all_induction_things(num_examples, seq_len, device, model_name = "gpt2" ,data_seed=42, metric="kl_div", return_one_element=True) -> AllDataThings:
    tl_model = get_model(device=device,model_name=model_name)

    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device) # None so we get all
    assert validation_data_orig.shape == mask_orig.shape

    assert seq_len <= validation_data_orig.shape[1]-1

    validation_slice = slice(0, num_examples)
    validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
    validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
    validation_mask = mask_orig[validation_slice, :seq_len].contiguous()

    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()

    test_slice = slice(num_examples, num_examples*2)
    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1:seq_len+1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()

    # data_seed+1: different shuffling
    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()

        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()

    if metric == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_val_logprobs,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
            return_one_element=return_one_element,
        )
    elif metric == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
    elif metric == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels, base_model_logprobs=base_val_logprobs, mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
    else:
        raise ValueError(f"Unknown metric {metric}")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels, base_model_logprobs=base_test_logprobs, mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
    }
    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )
'''
def get_all_induction_things(num_examples, seq_len, device, model_name="gpt2", data_seed=42, metric="kl_div", return_one_element=True) -> AllDataThings:
    try:
        print("加载模型中...")
        tl_model = get_model(device=device, model_name=model_name)
        
        # 获取模型的词汇表大小和最大序列长度
        if hasattr(tl_model, 'cfg'):
            # 使用 d_vocab 而不是 vocab_size
            vocab_size = tl_model.cfg.d_vocab
            max_seq_len = tl_model.cfg.n_ctx if hasattr(tl_model.cfg, 'n_ctx') else 1024
        else:
            vocab_size = tl_model.config.vocab_size if hasattr(tl_model.config, 'vocab_size') else 50257
            max_seq_len = tl_model.config.max_position_embeddings if hasattr(tl_model.config, 'max_position_embeddings') else 1024
        
        print(f"模型词汇表大小: {vocab_size}")
        print(f"模型最大序列长度: {max_seq_len}")
        
        print("加载数据中...")
        validation_data_orig = get_validation_data(device=device)
        mask_orig = get_mask_repeat_candidates(num_examples=None, device=device)  # None so we get all
        
        # 打印诊断信息
        print(f"validation_data_orig.shape: {validation_data_orig.shape}")
        print(f"validation_data_orig.dtype: {validation_data_orig.dtype}")
        print(f"validation_data_orig.min(): {validation_data_orig.min().item()}")
        print(f"validation_data_orig.max(): {validation_data_orig.max().item()}")
        
        # 检查数据中的最大索引值
        max_idx = validation_data_orig.max().item()
        if max_idx >= vocab_size:
            print(f"警告: 数据中包含超出词汇表范围的索引 (最大值: {max_idx}, 词汇表大小: {vocab_size})")
            print("正在裁剪超出范围的索引...")
            validation_data_orig = torch.clamp(validation_data_orig, 0, vocab_size-1)
            print(f"裁剪后的最大索引值: {validation_data_orig.max().item()}")
        
        assert validation_data_orig.shape == mask_orig.shape
        
        # 确保 seq_len 不超过模型最大长度和数据长度
        original_seq_len = seq_len
        seq_len = min(seq_len, max_seq_len - 1, validation_data_orig.shape[1] - 1)
        if seq_len != original_seq_len:
            print(f"警告: 序列长度已从 {original_seq_len} 调整为 {seq_len}")
        
        assert seq_len <= validation_data_orig.shape[1]-1
        
        print("处理训练和验证数据...")
        validation_slice = slice(0, num_examples)
        validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
        validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
        validation_mask = mask_orig[validation_slice, :seq_len].contiguous()
        
        # 再次检查处理后的数据
        print(f"validation_data.shape: {validation_data.shape}")
        print(f"validation_data.min(): {validation_data.min().item()}")
        print(f"validation_data.max(): {validation_data.max().item()}")
        
        validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()
        
        test_slice = slice(num_examples, num_examples*2)
        test_data = validation_data_orig[test_slice, :seq_len].contiguous()
        test_labels = validation_data_orig[test_slice, 1:seq_len+1].contiguous()
        test_mask = mask_orig[test_slice, :seq_len].contiguous()
        
        # data_seed+1: different shuffling
        test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()
        
        print("开始模型推理...")
        try:
            with torch.no_grad():
                # 检查是否需要批处理
                total_elements = validation_data.numel()
                # 如果数据量大，使用批处理
                if total_elements > 100000:  # 可以根据您的GPU内存调整阈值
                    print(f"数据量较大 ({total_elements} 元素)，使用批处理...")
                    batch_size = 8  # 可以根据您的GPU内存调整
                    
                    # 处理验证数据
                    all_val_logprobs = []
                    num_batches = (validation_data.size(0) + batch_size - 1) // batch_size
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, validation_data.size(0))
                        batch = validation_data[start_idx:end_idx]
                        
                        print(f"处理验证批次 {i+1}/{num_batches}, 形状: {batch.shape}")
                        outputs = tl_model(batch)
                        batch_logprobs = F.log_softmax(outputs, dim=-1).detach()
                        all_val_logprobs.append(batch_logprobs)
                    
                    base_val_logprobs = torch.cat(all_val_logprobs, dim=0)
                    
                    # 处理测试数据
                    all_test_logprobs = []
                    num_batches = (test_data.size(0) + batch_size - 1) // batch_size
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, test_data.size(0))
                        batch = test_data[start_idx:end_idx]
                        
                        print(f"处理测试批次 {i+1}/{num_batches}, 形状: {batch.shape}")
                        outputs = tl_model(batch)
                        batch_logprobs = F.log_softmax(outputs, dim=-1).detach()
                        all_test_logprobs.append(batch_logprobs)
                    
                    base_test_logprobs = torch.cat(all_test_logprobs, dim=0)
                else:
                    print("直接处理整个数据集...")
                    # 原始代码，直接处理整个数据集
                    base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()
                    base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()
                
                print("模型推理完成!")
                
        except RuntimeError as e:
            if "index out of bounds" in str(e) or "device-side assert triggered" in str(e):
                print(f"错误: {str(e)}")
                print("可能的原因:")
                print("1. 输入数据中包含超出词汇表范围的索引")
                print("2. 序列长度超出模型支持的最大长度")
                print("3. GPU内存不足")
                
                # 尝试修复问题 - 裁剪索引并减小批次大小
                print("尝试修复: 裁剪索引值并使用更小的批次大小...")
                validation_data = torch.clamp(validation_data, 0, vocab_size-1)
                test_data = torch.clamp(test_data, 0, vocab_size-1)
                
                # 使用更小的批次大小
                batch_size = 1
                
                # 重试
                with torch.no_grad():
                    all_val_logprobs = []
                    num_batches = (validation_data.size(0) + batch_size - 1) // batch_size
                    
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, validation_data.size(0))
                        batch = validation_data[start_idx:end_idx]
                        
                        outputs = tl_model(batch)
                        batch_logprobs = F.log_softmax(outputs, dim=-1).detach()
                        all_val_logprobs.append(batch_logprobs)
                    
                    base_val_logprobs = torch.cat(all_val_logprobs, dim=0)
                    
                    all_test_logprobs = []
                    num_batches = (test_data.size(0) + batch_size - 1) // batch_size
                    
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, test_data.size(0))
                        batch = test_data[start_idx:end_idx]
                        
                        outputs = tl_model(batch)
                        batch_logprobs = F.log_softmax(outputs, dim=-1).detach()
                        all_test_logprobs.append(batch_logprobs)
                    
                    base_test_logprobs = torch.cat(all_test_logprobs, dim=0)
                    print("修复后模型推理成功!")
            else:
                # 重新抛出其他类型的错误
                raise
        
        print("设置评估指标...")
        if metric == "kl_div":
            validation_metric = partial(
                kl_divergence,
                base_model_logprobs=base_val_logprobs,
                mask_repeat_candidates=validation_mask,
                last_seq_element_only=False,
                return_one_element=return_one_element,
            )
        elif metric == "nll":
            validation_metric = partial(
                negative_log_probs,
                labels=validation_labels,
                mask_repeat_candidates=validation_mask,
                last_seq_element_only=False,
            )
        elif metric == "match_nll":
            validation_metric = MatchNLLMetric(
                labels=validation_labels, base_model_logprobs=base_val_logprobs, mask_repeat_candidates=validation_mask,
                last_seq_element_only=False,
            )
        else:
            raise ValueError(f"Unknown metric {metric}")
        
        test_metrics = {
            "kl_div": partial(
                kl_divergence,
                base_model_logprobs=base_test_logprobs,
                mask_repeat_candidates=test_mask,
                last_seq_element_only=False,
            ),
            "nll": partial(
                negative_log_probs,
                labels=test_labels,
                mask_repeat_candidates=test_mask,
                last_seq_element_only=False,
            ),
            "match_nll": MatchNLLMetric(
                labels=test_labels, base_model_logprobs=base_test_logprobs, mask_repeat_candidates=test_mask,
                last_seq_element_only=False,
            ),
        }
        
        print("函数执行完成，返回结果")
        return AllDataThings(
            tl_model=tl_model,
            validation_metric=validation_metric,
            validation_data=validation_data,
            validation_labels=validation_labels,
            validation_mask=validation_mask,
            validation_patch_data=validation_patch_data,
            test_metrics=test_metrics,
            test_data=test_data,
            test_labels=test_labels,
            test_mask=test_mask,
            test_patch_data=test_patch_data,
        )
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_logprobs, kl_take_mean=True):
    """Returns each instance of induction as its own batch idx"""

    end_positions = []
    batch_size, seq_len = toks_int_values.shape
    new_tensors = []

    toks_int_values_other_batch_list = []
    new_base_model_logprobs_list = []

    for i in range(batch_size):
        for j in range(seq_len - 1): # -1 because we don't know what follows the last token so can't calculate losses
            if mask_rep[i, j]:
                end_positions.append(j)
                new_tensors.append(toks_int_values[i].cpu().clone())
                toks_int_values_other_batch_list.append(toks_int_values_other[i].cpu().clone())
                new_base_model_logprobs_list.append(base_model_logprobs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device).clone()
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device).clone()
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_logprobs = torch.stack(new_base_model_logprobs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device).clone()
    metric = partial(
        kl_divergence, 
        base_model_logprobs=new_base_model_logprobs, 
        end_positions=end_positions_tensor, 
        mask_repeat_candidates=None, # !!! 
        last_seq_element_only=False, 
        return_one_element=False
    )
    
    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric
