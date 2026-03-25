"""Round-to-Nearest (RTN) quantization module (``pahq.quantizer``).

This module provides:
- RTNQuantizer: symmetric per-tensor RTN quantization at arbitrary bit widths.
- weight_split_by_head: splits a weight matrix by attention head index (corrected
  version of the buggy reference_sketch.py::weight_arrd / mix_percision_matmul).
- mixed_precision_matmul: computes matmul with head-split weights.

The RTNQuantizer is used to construct the RTN-Q baseline described in the PAHQ
paper (Section 3 / Table 1): all non-selected attention-head weights are quantized
to INT8 (or INT4) while the selected head stays at FP32.
"""

from __future__ import annotations

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Core quantizer
# ---------------------------------------------------------------------------

class RTNQuantizer:
    """Symmetric, per-tensor, round-to-nearest quantizer.

    Parameters
    ----------
    bits : int
        Target bit-width (e.g. 8 for INT8, 4 for INT4).

    Examples
    --------
    >>> q = RTNQuantizer(bits=8)
    >>> w = torch.randn(4, 64)
    >>> w_q = q.quantize(w)     # still float32, but values snapped to INT8 grid
    >>> w_dq = q.dequantize(w_q, w)  # reconstruct (not needed separately, convenience)
    """

    def __init__(self, bits: int = 8) -> None:
        if bits < 1 or bits > 32:
            raise ValueError(f"bits must be between 1 and 32, got {bits}")
        self.bits = bits
        self._n_levels = 2 ** (bits - 1) - 1  # e.g. 127 for INT8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize ``weight`` and return a dequantized float32 tensor.

        The quantization is performed in-place on a clone so the original tensor
        is not modified.  Returns float32 so downstream code does not need to
        handle integer dtypes.
        """
        scale = self._compute_scale(weight)
        q = torch.round(weight / scale).clamp(-self._n_levels, self._n_levels)
        return (q * scale).to(torch.float32)

    def quantize_to_int(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (int_weight, scale) pair for explicit integer storage.

        The original weight can be reconstructed as ``int_weight * scale``.
        """
        scale = self._compute_scale(weight)
        q = torch.round(weight / scale).clamp(-self._n_levels, self._n_levels).to(torch.int8 if self.bits <= 8 else torch.int16)
        return q, scale

    def dequantize(self, int_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Convert (int_weight, scale) back to float32."""
        return (int_weight.to(torch.float32) * scale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_scale(self, weight: torch.Tensor) -> torch.Tensor:
        """Per-tensor symmetric scale: scale = max(|w|) / n_levels."""
        max_val = weight.abs().max()
        # Avoid division by zero for zero-weight tensors
        scale = max_val / self._n_levels if max_val > 0 else torch.ones(1, device=weight.device, dtype=weight.dtype)
        return scale


# ---------------------------------------------------------------------------
# Head-splitting utilities (corrected from docs/reference_sketch.py)
# ---------------------------------------------------------------------------

def weight_split_by_head(
    weight: torch.Tensor,
    n_heads: int,
    target_head: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a 2-D weight [d_in, d_out] along the head dimension.

    Assumes the output dimension ``d_out = d_head * n_heads`` is evenly divisible.

    Parameters
    ----------
    weight : Tensor [d_in, d_out]
    n_heads : int
        Total number of attention heads.
    target_head : int
        Index of the high-precision head to extract.

    Returns
    -------
    weight_low1 : Tensor [d_in, d_head * target_head]
        Weights for heads 0 .. target_head - 1.
    weight_high : Tensor [d_in, d_head]
        Weights for the high-precision target_head.
    weight_low2 : Tensor [d_in, d_head * (n_heads - target_head - 1)]
        Weights for heads target_head+1 .. n_heads - 1.

    Notes
    -----
    This is the corrected version of the buggy ``weight_arrd`` / ``quantize_weight_on_cpu``
    in docs/reference_sketch.py.  The reference sketch had ``target_h`` out of scope and
    ``weight_high`` used before assignment.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    d_in, d_out = weight.shape
    if d_out % n_heads != 0:
        raise ValueError(f"d_out={d_out} is not divisible by n_heads={n_heads}")
    d_head = d_out // n_heads

    # Reshape to [d_in, n_heads, d_head] for clean indexing
    w = weight.reshape(d_in, n_heads, d_head)
    weight_low1 = w[:, :target_head, :].reshape(d_in, d_head * target_head)
    weight_high = w[:, target_head, :]                     # [d_in, d_head]
    weight_low2 = w[:, target_head + 1:, :].reshape(d_in, d_head * (n_heads - target_head - 1))

    return weight_low1, weight_high, weight_low2


def mixed_precision_matmul(
    x: torch.Tensor,
    weight_low1: torch.Tensor,
    weight_high: torch.Tensor,
    weight_low2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute x @ [weight_low1, weight_high, weight_low2] separately.

    Parameters
    ----------
    x : Tensor [..., d_in]
    weight_low1 : Tensor [d_in, k1]  (low-precision)
    weight_high : Tensor [d_in, d_head]  (high-precision)
    weight_low2 : Tensor [d_in, k2]  (low-precision)

    Returns
    -------
    out_low1, out_high, out_low2 : Tensors corresponding to the three weight parts.

    Notes
    -----
    Corrected from docs/reference_sketch.py::mix_percision_matmul which had a
    missing ``return`` inside the inner function, causing None outputs.
    """
    out_low1 = torch.matmul(x.to(weight_low1.dtype), weight_low1)
    out_high = torch.matmul(x.to(weight_high.dtype), weight_high)
    out_low2 = torch.matmul(x.to(weight_low2.dtype), weight_low2)
    return out_low1, out_high, out_low2
