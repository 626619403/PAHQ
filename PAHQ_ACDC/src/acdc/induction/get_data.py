from transformers import PreTrainedTokenizerBase
import torch
import os
import sys
sys.path.append("/root/autodl-tmp/acdc_q")
import pickle
from typing import Dict, Any, Optional, Tuple
import dataclasses
from functools import partial
#from acdc.docstring.utils import AllDataThings
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
@dataclasses.dataclass(frozen=False)
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: Optional[torch.Tensor]
    validation_mask: Optional[torch.Tensor]
    validation_patch_data: torch.Tensor
    test_metrics: dict[str, Any]
    test_data: torch.Tensor
    test_labels: Optional[torch.Tensor]
    test_mask: Optional[torch.Tensor]
    test_patch_data: torch.Tensor
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
    
def convert_and_save_data_to_text(
    model_name: str,
    output_dir: str,
    device: str = "cuda",
    num_examples: Optional[int] = None,
    seq_len: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convert validation data, masks, and good_induction_candidates to text and save locally
    
    Parameters:
        model_name: Model name, used to load the model and tokenizer
        output_dir: Output directory
        device: Device to use
        num_examples: Number of examples to process, None means all
        seq_len: Sequence length, None means use original length
        
    Returns:
        Dictionary containing the saved file path
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = load_model(model_name, device)
    
    # Ensure model has tokenizer
    assert model.tokenizer is not None, "Model does not have a tokenizer!"
    
    # Load original data
    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(device=device)
    
    try:
        good_induction_candidates = get_good_induction_candidates(device=device)
        has_good_induction = True
    except:
        has_good_induction = False
        print("good_induction_candidates not found, skipping processing")
    
    # Apply sample and sequence length limits
    if num_examples is not None:
        validation_data = validation_data_orig[:num_examples]
        mask = mask_orig[:num_examples]
        if has_good_induction:
            good_induction = good_induction_candidates[:num_examples]
    else:
        validation_data = validation_data_orig
        mask = mask_orig
        if has_good_induction:
            good_induction = good_induction_candidates
    
    if seq_len is not None:
        validation_data = validation_data[:, :seq_len]
        mask = mask[:, :seq_len]
        if has_good_induction and hasattr(good_induction, 'shape') and len(good_induction.shape) > 1:
            good_induction = good_induction[:, :seq_len]
    
    # Convert tokens to text
    print(f"Converting {validation_data.shape[0]} samples to text...")
    text_data = []
    for i in range(validation_data.shape[0]):
        text = model.to_string(validation_data[i])
        text_data.append(text)
    
    # Save data
    data_dict = {
        "text_data": text_data,
        "mask": mask.cpu(),
        "original_tokens": validation_data.cpu()
    }
    
    if has_good_induction:
        data_dict["good_induction"] = good_induction.cpu()
    
    # Save to file
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"Data saved to {output_path}")
    return {"output_path": output_path}

def load_and_process_local_data(
    data_path: str,
    new_model_name: str,
    device: str = "cuda",
    num_examples: Optional[int] = None,
    seq_len: Optional[int] = None,
    metric: str = "kl_div",
    return_one_element: bool = True
) -> AllDataThings:
    """
    Load locally stored text data, process it with a new model, and return an AllDataThings object
    
    Parameters:
        data_path: Path to local data file
        new_model_name: Name of the new model
        device: Device to use
        num_examples: Number of examples to process, None means all
        seq_len: Sequence length, None means use original length
        metric: Evaluation metric to use
        return_one_element: Whether to return a single element
        
    Returns:
        AllDataThings object containing processed data and model
    """
    # Load new model
    tl_model = load_model(model_name=new_model_name, device=device)
    
    # Ensure model has tokenizer
    assert tl_model.tokenizer is not None, "Model does not have a tokenizer!"
    
    # Load local data
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    text_data = data_dict["text_data"]
    mask_orig = data_dict["mask"]
    
    # Apply sample and sequence length limits
    if num_examples is not None:
        text_data = text_data[:num_examples*2]  # *2 to split into validation and test sets
        mask_orig = mask_orig[:num_examples*2]
    
    # Process text data with new model's tokenizer
    print("Processing text data with new model's tokenizer...")
    all_tokens = []
    for text in text_data:
        tokens = tl_model.to_tokens(text, prepend_bos=False)
        all_tokens.append(tokens[0])  # Remove batch dimension
    
    # Stack all tokens into a tensor
    max_len = max(len(t) for t in all_tokens)
    if seq_len is not None:
        max_len = min(max_len, seq_len)
    
    padded_tokens = []
    for tokens in all_tokens:
        if len(tokens) >= max_len:
            padded_tokens.append(tokens[:max_len])
        else:
            # Right padding
            padding = torch.full((max_len - len(tokens),), tl_model.tokenizer.pad_token_id, 
                                dtype=tokens.dtype, device=tokens.device)
            padded_tokens.append(torch.cat([tokens, padding]))
    
    all_tokens_tensor = torch.stack(padded_tokens).to(device)
    
    # Adjust mask to match new token length
    if seq_len is not None and mask_orig.shape[1] > seq_len:
        mask = mask_orig[:, :seq_len].to(device)
    else:
        mask = mask_orig.to(device)
    
    # Split into validation and test sets
    half_point = all_tokens_tensor.shape[0] // 2
    validation_data = all_tokens_tensor[:half_point]
    validation_mask = mask[:half_point]
    
    test_data = all_tokens_tensor[half_point:]
    test_mask = mask[half_point:]
    
    # Create labels - ensure input and label sequence lengths match
    # For consistency, we truncate the input by one position, rather than shifting the labels left
    validation_input = validation_data[:, :-1].contiguous()  # Input without the last token
    validation_labels = validation_data[:, 1:].contiguous()  # Labels without the first token
    
    test_input = test_data[:, :-1].contiguous()
    test_labels = test_data[:, 1:].contiguous()
    
    # Adjust masks accordingly
    if validation_mask.shape[1] > validation_input.shape[1]:
        validation_mask = validation_mask[:, :validation_input.shape[1]]
    if test_mask.shape[1] > test_input.shape[1]:
        test_mask = test_mask[:, :test_input.shape[1]]
    
    # Create patch data (randomly shuffled)
    data_seed = 42
    validation_patch_data = shuffle_tensor(validation_input, seed=data_seed).contiguous()
    test_patch_data = shuffle_tensor(test_input, seed=data_seed+1).contiguous()
    
    # Calculate base model logprobs - using adjusted inputs
    with torch.no_grad():
        # Validation set
        base_val_logits = tl_model(validation_input)
        # Ensure logits and input sequence lengths match
        if base_val_logits.shape[1] != validation_input.shape[1]:
            base_val_logits = base_val_logits[:, :validation_input.shape[1]]
        base_val_logprobs = F.log_softmax(base_val_logits, dim=-1).detach()
        
        # Test set
        base_test_logits = tl_model(test_input)
        # Ensure logits and input sequence lengths match
        if base_test_logits.shape[1] != test_input.shape[1]:
            base_test_logits = base_test_logits[:, :test_input.shape[1]]
        base_test_logprobs = F.log_softmax(base_test_logits, dim=-1).detach()
    
    # Print shape information for debugging
    print(f"Validation input shape: {validation_input.shape}, Validation labels shape: {validation_labels.shape}")
    print(f"Validation logprobs shape: {base_val_logprobs.shape}, Validation mask shape: {validation_mask.shape}")
    print(f"Test input shape: {test_input.shape}, Test labels shape: {test_labels.shape}")
    print(f"Test logprobs shape: {base_test_logprobs.shape}, Test mask shape: {test_mask.shape}")
    
    # Configure evaluation metrics
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
        # Add safety check to ensure batch sizes match
        if base_val_logprobs.shape[0] != validation_labels.shape[0]:
            min_batch = min(base_val_logprobs.shape[0], validation_labels.shape[0])
            print(f"Warning: Validation set batch sizes don't match! Adjusting to {min_batch}")
            base_val_logprobs = base_val_logprobs[:min_batch]
            validation_labels = validation_labels[:min_batch]
            validation_mask = validation_mask[:min_batch] if validation_mask is not None else None
        
        validation_metric = MatchNLLMetric(
            labels=validation_labels, 
            base_model_logprobs=base_val_logprobs, 
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
    else:
        raise ValueError(f"Unknown evaluation metric {metric}")
    
    # Configure test metrics - add the same safety check
    if base_test_logprobs.shape[0] != test_labels.shape[0]:
        min_batch = min(base_test_logprobs.shape[0], test_labels.shape[0])
        print(f"Warning: Test set batch sizes don't match! Adjusting to {min_batch}")
        base_test_logprobs = base_test_logprobs[:min_batch]
        test_labels = test_labels[:min_batch]
        test_mask = test_mask[:min_batch] if test_mask is not None else None
    
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
            labels=test_labels, 
            base_model_logprobs=base_test_logprobs, 
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
    }
    
    print(f"Data processing complete, validation set size: {validation_input.shape}, test set size: {test_input.shape}")
    
    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_input,  # Use adjusted input
        validation_labels=validation_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_input,  # Use adjusted input
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )

def main():
    
    new_model_name = "attn-only-4l"  
    all_data_things = load_and_process_local_data(
        data_path="./acdc/induction/data/redwood_attn_2l_data.pkl",
        new_model_name=new_model_name,
        device="cuda",
        num_examples=1, 
        seq_len=128,  
        metric="kl_div"
    )

    print(f"Model: {all_data_things.tl_model.cfg.model_name}")
    print(f"Validation data shape: {all_data_things.validation_data.shape}")
    print(f"Test data shape: {all_data_things.test_data.shape}")
    
    # Run simple inference test
    with torch.no_grad():
        logits = all_data_things.tl_model(all_data_things.validation_data[:1])
        print(f"Inference output shape: {logits.shape}")


if __name__ == "__main__":
    main()
