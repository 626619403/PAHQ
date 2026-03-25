"""
Design sketch / early prototype — NOT runnable production code.

This file documents the original conceptual design for PAHQ's weight decomposition
approach.  It was preserved here (from PAHQ_ACDC/reference.py) because it clarifies
the intent behind `weight_arrd`, even though the file contains two confirmed bugs:

Bug 1 — quantize_weight_on_cpu:
  `target_h` is not in scope (missing function parameter).
  `weight_high` is used before assignment on the right-hand side of its own definition.
  The correct version is in pahq/rtn_quantizer.py::weight_split_by_head().

Bug 2 — mix_percision_matmul:
  `_mix_percision_matmul` computes a result but never returns it (implicit None).
  x_low1 and x_low2 are set to None, then returned silently.
  The corrected version is in pahq/rtn_quantizer.py::mixed_precision_matmul().

The `weight_arrd` function below is correct and is reproduced unchanged.
"""

import torch
from torch.nn.parameter import Parameter


def weight_arrd(weight, h=32, target_h=0):
    """Split a 2-D weight matrix [d1, d2] into three parts along the head dimension.

    Splits weight into:
      - weight_low1: heads 0 .. target_h-1
      - weight_high: head target_h (to be kept at high precision)
      - weight_low2: heads target_h+1 .. h-1

    Returns three reshaped 2-D tensors ready for mixed-precision matmul.
    """
    d1, d2 = weight.shape
    weight = weight.reshape(d1, d2 // h, h)
    weight_low1 = weight[:, :, :target_h]
    weight_high = weight[:, :, target_h]
    weight_low2 = weight[:, :, target_h + 1:]
    return (
        weight_low1.reshape(d1, d2 // h * target_h),
        weight_high.reshape(d1, d2 // h),
        weight_low2.reshape(d1, d2 - d2 // h * (target_h + 1)),
    )


# ---- BUGGY CODE BELOW — kept for historical reference only ----
# The corrected implementations are in pahq/rtn_quantizer.py.

def _BUGGY_quantize_weight_on_cpu(weight, weight_cache):  # noqa: N802
    # BUG: target_h not in scope; weight_high used before assignment
    weight_high = weight_cache[:, :, target_h].to(weight_high.device)  # type: ignore # noqa
    return weight_low1, weight_high, weight_low2  # type: ignore # noqa


def _BUGGY_mix_percision_matmul(x, weight_low1, weight_high, weight_low2):  # noqa: N802
    def _inner(x, weight):
        x = x.to(weight.dtype)
        x = torch.matmul(x, weight)
        # BUG: missing `return x` — implicitly returns None

    x_low1 = _inner(x[:, :, :weight_low1.shape[-1]], weight_low1)  # None
    x_high = torch.matmul(x[:, :, weight_low1.shape[-1]:-weight_low2.shape[-1]], weight_high)
    x_low2 = _inner(x[:, :, -weight_low2.shape[-1]:], weight_low2)  # None
    return x_low1, x_high, x_low2  # returns (None, tensor, None)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h=32, d=512):
        super().__init__()
        self.weight_qkv = Parameter(torch.empty((d, d*3), device=torch.device("cpu")))
        self.out = torch.nn.Linear(d, d)
        self.h = h
        self.d = d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape[:2]
        weight_qkv_device = self.weight_qkv.to(x.device)
        weight_qkv_device_low1, weight_qkv_device_high, weight_qkv_device_low2 = \
            quantize_weight_on_cuda(weight_qkv_device.t())
        qkv_low1, qkv_high, qkv_low2 = mix_percision_matmul(
            x, weight_qkv_device_low1, weight_qkv_device_high, weight_qkv_device_low2)
        # q, k, v = q.reshape(B, N, self.h, -1), k.reshape(B, N, self.h, -1), v.reshape(B, N, self.h, -1)

        q, k, v = q / (self.d ** 0.5), k, v
        qk = torch.einsum('bqd,bkd->bqk', q, k)
        s = torch.nn.functional.softmax(qk, dim=-1)
        y = torch.einsum('bqk,bkd->bqd', s, v)
        return self.out(y)