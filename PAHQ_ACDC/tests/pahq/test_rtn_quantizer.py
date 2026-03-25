"""Unit tests for pahq.rtn_quantizer."""

import pytest
import torch

from pahq.rtn_quantizer import (
    RTNQuantizer,
    mixed_precision_matmul,
    weight_split_by_head,
)


# ---------------------------------------------------------------------------
# RTNQuantizer
# ---------------------------------------------------------------------------

class TestRTNQuantizerInit:
    def test_valid_bits(self):
        for b in [1, 4, 8, 16, 32]:
            q = RTNQuantizer(bits=b)
            assert q.bits == b

    def test_invalid_bits_low(self):
        with pytest.raises(ValueError):
            RTNQuantizer(bits=0)

    def test_invalid_bits_high(self):
        with pytest.raises(ValueError):
            RTNQuantizer(bits=33)

    def test_n_levels_int8(self):
        q = RTNQuantizer(bits=8)
        assert q._n_levels == 127  # 2^7 - 1

    def test_n_levels_int4(self):
        q = RTNQuantizer(bits=4)
        assert q._n_levels == 7  # 2^3 - 1


class TestRTNQuantizerQuantize:
    def test_output_dtype_is_float32(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 8)
        out = q.quantize(w)
        assert out.dtype == torch.float32

    def test_values_clipped_to_range(self):
        q = RTNQuantizer(bits=8)
        # Extreme values should be clipped to [-127, 127] * scale
        w = torch.tensor([[-1000.0, 1000.0]])
        out = q.quantize(w)
        scale = w.abs().max() / 127
        max_possible = 127 * scale
        assert out.abs().max() <= max_possible + 1e-5

    def test_zero_weight_no_nan(self):
        q = RTNQuantizer(bits=8)
        w = torch.zeros(3, 3)
        out = q.quantize(w)
        assert not torch.isnan(out).any()
        assert torch.all(out == 0)

    def test_original_tensor_not_modified(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 8)
        w_copy = w.clone()
        q.quantize(w)
        assert torch.allclose(w, w_copy)

    def test_quantization_error_decreases_with_more_bits(self):
        w = torch.randn(16, 16) * 5.0
        errors = []
        for bits in [2, 4, 8]:
            q = RTNQuantizer(bits=bits)
            w_q = q.quantize(w)
            errors.append((w - w_q).pow(2).mean().item())
        # Higher bits → lower MSE
        assert errors[0] > errors[1] > errors[2]


class TestRTNQuantizerQuantizeToInt:
    def test_returns_int8_for_8bit(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 4)
        q_int, scale = q.quantize_to_int(w)
        assert q_int.dtype == torch.int8

    def test_scale_is_positive(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 4)
        _, scale = q.quantize_to_int(w)
        assert scale > 0

    def test_round_trip(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 4)
        q_int, scale = q.quantize_to_int(w)
        w_reconstructed = q.dequantize(q_int, scale)
        # Reconstruction error should be small relative to weight range
        rel_err = (w - w_reconstructed).abs().mean() / w.abs().mean()
        assert rel_err < 0.05  # <5% relative error for INT8

    def test_dequantize_output_dtype_float32(self):
        q = RTNQuantizer(bits=8)
        w = torch.randn(4, 4)
        q_int, scale = q.quantize_to_int(w)
        out = q.dequantize(q_int, scale)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# weight_split_by_head
# ---------------------------------------------------------------------------

class TestWeightSplitByHead:
    def test_output_shapes(self):
        d_in, n_heads = 8, 4
        d_out = 16  # d_head = 4
        w = torch.randn(d_in, d_out)
        low1, high, low2 = weight_split_by_head(w, n_heads=n_heads, target_head=1)
        d_head = d_out // n_heads
        assert low1.shape == (d_in, d_head * 1)   # heads 0..0
        assert high.shape == (d_in, d_head)         # head 1
        assert low2.shape == (d_in, d_head * 2)     # heads 2..3

    def test_target_head_0(self):
        d_in, n_heads = 8, 4
        d_out = 16
        w = torch.randn(d_in, d_out)
        low1, high, low2 = weight_split_by_head(w, n_heads=n_heads, target_head=0)
        d_head = d_out // n_heads
        assert low1.shape == (d_in, 0)              # no heads before 0
        assert high.shape == (d_in, d_head)
        assert low2.shape == (d_in, d_head * 3)

    def test_target_head_last(self):
        d_in, n_heads = 8, 4
        d_out = 16
        w = torch.randn(d_in, d_out)
        low1, high, low2 = weight_split_by_head(w, n_heads=n_heads, target_head=3)
        d_head = d_out // n_heads
        assert low1.shape == (d_in, d_head * 3)
        assert high.shape == (d_in, d_head)
        assert low2.shape == (d_in, 0)

    def test_concatenation_recovers_original(self):
        d_in, n_heads = 8, 4
        d_out = 16
        d_head = d_out // n_heads
        w = torch.arange(float(d_in * d_out)).reshape(d_in, d_out)
        # Split and reassemble along output dimension
        w_reshaped = w.reshape(d_in, n_heads, d_head)
        for target in range(n_heads):
            low1, high, low2 = weight_split_by_head(w, n_heads=n_heads, target_head=target)
            # Reassemble: low1 = heads 0..target-1, high = head target, low2 = target+1..end
            high_2d = high  # already [d_in, d_head]
            parts = []
            if low1.shape[1] > 0:
                parts.append(low1)
            parts.append(high_2d)
            if low2.shape[1] > 0:
                parts.append(low2)
            reconstructed = torch.cat(parts, dim=1)
            assert reconstructed.shape == w.shape
            assert torch.allclose(reconstructed, w), f"Mismatch at target_head={target}"

    def test_requires_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            weight_split_by_head(torch.randn(2, 4, 4), n_heads=2, target_head=0)

    def test_requires_divisible_d_out(self):
        with pytest.raises(ValueError, match="divisible"):
            weight_split_by_head(torch.randn(4, 7), n_heads=3, target_head=0)


# ---------------------------------------------------------------------------
# mixed_precision_matmul
# ---------------------------------------------------------------------------

class TestMixedPrecisionMatmul:
    def setup_method(self):
        torch.manual_seed(42)
        self.d_in = 8
        self.n_heads = 4
        self.d_head = 2
        self.d_out = self.n_heads * self.d_head
        self.batch, self.seq = 2, 3

        self.x = torch.randn(self.batch, self.seq, self.d_in)
        w = torch.randn(self.d_in, self.d_out)
        self.low1, self.high, self.low2 = weight_split_by_head(
            w, n_heads=self.n_heads, target_head=1
        )

    def test_output_shapes(self):
        out_low1, out_high, out_low2 = mixed_precision_matmul(
            self.x, self.low1, self.high, self.low2
        )
        assert out_low1.shape == (self.batch, self.seq, self.low1.shape[1])
        assert out_high.shape == (self.batch, self.seq, self.high.shape[1])
        assert out_low2.shape == (self.batch, self.seq, self.low2.shape[1])

    def test_no_none_outputs(self):
        """Regression: the original buggy version returned None for low1/low2."""
        out_low1, out_high, out_low2 = mixed_precision_matmul(
            self.x, self.low1, self.high, self.low2
        )
        assert out_low1 is not None
        assert out_high is not None
        assert out_low2 is not None

    def test_concatenation_matches_full_matmul(self):
        """Verify that concatenating the three outputs equals x @ w_full."""
        w = torch.cat([self.low1, self.high, self.low2], dim=1)
        expected = torch.matmul(self.x, w)

        out_low1, out_high, out_low2 = mixed_precision_matmul(
            self.x, self.low1, self.high, self.low2
        )
        recovered = torch.cat([out_low1, out_high, out_low2], dim=-1)
        assert torch.allclose(recovered, expected, atol=1e-5)

    def test_empty_low1(self):
        """Target head = 0 → low1 is empty [d_in, 0]."""
        d_in, n_heads, d_head = 8, 4, 2
        w = torch.randn(d_in, n_heads * d_head)
        low1, high, low2 = weight_split_by_head(w, n_heads=n_heads, target_head=0)
        x = torch.randn(2, 3, d_in)
        out_low1, out_high, out_low2 = mixed_precision_matmul(x, low1, high, low2)
        assert out_low1.shape[-1] == 0
        assert out_high.shape[-1] == d_head
