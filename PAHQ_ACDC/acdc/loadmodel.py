import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from functools import partial

# Import transformer_lens
from transformer_lens import HookedTransformer

@dataclass(frozen=False)
class ModelData:
    """Container class for storing model and related data"""
    model: HookedTransformer
    validation_metric: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    validation_data: Optional[torch.Tensor] = None
    validation_labels: Optional[torch.Tensor] = None
    validation_mask: Optional[torch.Tensor] = None
    validation_patch_data: Optional[torch.Tensor] = None
    test_metrics: Optional[Dict[str, Any]] = None
    test_data: Optional[torch.Tensor] = None
    test_labels: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    test_patch_data: Optional[torch.Tensor] = None

class ModelLoader:
    """Utility class for loading and configuring various pretrained models"""
    
    @staticmethod
    def load_model(model_name: str, device: str = "cuda") -> HookedTransformer:
        """
        Load a Transformer model based on the model name
        
        Args:
            model_name: Model name
            device: Device, defaults to "cuda"
            
        Returns:
            Configured HookedTransformer model
            
        Raises:
            ValueError: If the model name is not recognized
            ImportError: If model loading fails
        """
        model_name = model_name.lower()
        
        # Attention-only model group
        if model_name in ["attn-only-4l", "attn-only-2l", "attn-only-3l"]:
            try:
                model = HookedTransformer.from_pretrained(model_name)
            except Exception as e:
                raise ImportError(f"Failed to load {model_name}. Please verify model name and transformer_lens version. Error: {e}")
            # Configure attention hooks
            model = model.to(device)
            model.set_use_attn_result(True)
            model.set_use_split_qkv_input(True)
            return model
        
        # GPT-2 style model group
        elif model_name in ["gpt2", "gpt2-medium", "distilgpt2"]:
            try:
                model = HookedTransformer.from_pretrained(model_name)
            except Exception as e:
                raise ImportError(f"Failed to load {model_name}. Please verify model name. Error: {e}")
            # Configure hooks for GPT-2 style models
            model = model.to(device)
            model.set_use_attn_result(True)
            model.set_use_split_qkv_input(True)
            if "use_hook_mlp_in" in model.cfg.to_dict():
                model.set_use_hook_mlp_in(True)
            return model
        
        # Redwood and similar research model group
        elif model_name in ["redwood_attn_2l", "redwood_attn_2l-ext", "solu_2l", "pythia-70m"]:
            try:
                # For research models, specific configuration parameters may be needed
                model = HookedTransformer.from_pretrained(model_name, center_writing_weights=False, 
                                                        center_unembed=False, fold_ln=False, device=device)
            except Exception as e:
                raise ImportError(f"Failed to load {model_name}. Please verify model name. Error: {e}")
            # Configure hooks
            model.set_use_attn_result(True)
            model.set_use_split_qkv_input(True)
            #if "use_hook_mlp_in" in model.cfg.to_dict():
                #model.set_use_hook_mlp_in(True)
            return model
        
        else:
            raise ValueError(f"Unknown model name: {model_name}. Please choose from supported models.")

    @staticmethod
    def create_dummy_data(model: HookedTransformer, batch_size: int = 8, seq_len: int = 16) -> ModelData:
        """
        Create random test data for the model
        
        Args:
            model: Loaded model
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            ModelData object containing the model and random data
        """
        device = model.device
        d_vocab = getattr(model.cfg, "d_vocab", 50257)
        dummy_input = torch.randint(low=0, high=d_vocab, size=(batch_size, seq_len), device=device)
        
        with torch.no_grad():
            logits = model(dummy_input)
            base_logprobs = F.log_softmax(logits[:, -1], dim=-1)

        # Construct a simple negative log probability metric
        def simple_nll(logits: torch.Tensor, labels: torch.Tensor, return_one_element: bool = True):
            batch_indices = torch.arange(len(labels)).to(labels.device)
            token_logprobs = logits[batch_indices, labels]
            loss = -token_logprobs
            if return_one_element:
                return loss.mean()
            return loss

        validation_metric = partial(simple_nll, labels=dummy_input[:, -1], return_one_element=True)
        test_metrics = {"nll": validation_metric}

        return ModelData(
            model=model,
            validation_metric=validation_metric,
            validation_data=dummy_input,
            validation_labels=dummy_input[:, -1],
            validation_mask=None,
            validation_patch_data=dummy_input,
            test_metrics=test_metrics,
            test_data=dummy_input,
            test_labels=dummy_input[:, -1],
            test_mask=None,
            test_patch_data=dummy_input,
        )

# Convenience functions for loading specific models directly
def load_attn_only_4l(device="cuda") -> HookedTransformer:
    """Load attn-only-4l model"""
    return ModelLoader.load_model("attn-only-4l", device)

def load_attn_only_2l(device="cuda") -> HookedTransformer:
    """Load attn-only-2l model"""
    return ModelLoader.load_model("attn-only-2l", device)

def load_attention_only_6l(device="cuda") -> HookedTransformer:
    """Load attention-only-6l model"""
    return ModelLoader.load_model("attention-only-6l", device)

def load_gpt2_small(device="cuda") -> HookedTransformer:
    """Load gpt2 model"""
    return ModelLoader.load_model("gpt2", device)

def load_distilgpt2(device="cuda") -> HookedTransformer:
    """Load distilgpt2 model"""
    return ModelLoader.load_model("distilgpt2", device)

def load_tiny_stories(device="cuda") -> HookedTransformer:
    """Load tiny_stories model"""
    return ModelLoader.load_model("tiny_stories", device)

def load_redwood_attn_2l(device="cuda") -> HookedTransformer:
    """Load redwood_attn_2l model"""
    return ModelLoader.load_model("redwood_attn_2l", device)

def load_solu_2l(device="cuda") -> HookedTransformer:
    """Load solu_2l model"""
    return ModelLoader.load_model("solu_2l", device)

def load_pythia_70m(device="cuda") -> HookedTransformer:
    """Load pythia-70m model"""
    return ModelLoader.load_model("pythia-70m", device)

def load_model(model_name: str, device: str = "cuda") -> HookedTransformer:
    """
    General model loading function
    
    Args:
        model_name: Model name
        device: Device name
        
    Returns:
        Loaded and configured model
    """
    return ModelLoader.load_model(model_name, device)

def load_model_with_data(model_name: str, device: str = "cuda", batch_size: int = 8, seq_len: int = 16) -> ModelData:
    """
    Load model and create test data
    
    Args:
        model_name: Model name
        device: Device name
        batch_size: Batch size
        seq_len: Sequence length
        
    Returns:
        ModelData object containing model and data
    """
    model = load_model(model_name, device)
    return ModelLoader.create_dummy_data(model, batch_size, seq_len)
