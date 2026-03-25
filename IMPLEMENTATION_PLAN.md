# PAHQ Codebase: Implementation Plan & Technical Report

## 1. Paper Summary

**PAHQ: Per Attention Head Quantization** (EACL 2026)
Accelerates ACDC (Automated Circuit Discovery) by maintaining FP32 for the single attention head under evaluation while quantizing all other heads to FP8 (E4M3). A three-stream CUDA scheduler (S_load, S_low, S_high) overlaps CPU→GPU weight transfer with FP8 computation to mask latency.

- **80% runtime reduction**, **30% memory reduction** vs. unaccelerated ACDC
- **>40% AUC-ROC improvement** vs. RTN-Q (naive 8-bit quantization)
- Training-free, plug-and-play with any edge-based circuit discovery method

---

## 2. Codebase Audit: Issues Found

### 2.1 Critical Bugs

| File | Issue | Fix |
|------|-------|-----|
| `PAHQ_ACDC/reference.py` | `quantize_weight_on_cpu` uses `target_h` not in scope; `_mix_percision_matmul` never returns | Moved to `docs/reference_sketch.py` (design sketch, not runnable); correct version in `pahq/rtn_quantizer.py` |
| `transformer_lens/GEMM.py` | Unconditional `import triton` at module level; crashes on CPU | Wrapped in `try/except ImportError` with CPU fallback |
| `components.py` L518-519 | `torch.cuda.Stream()` created inside `forward()` on every call; expensive | Moved to `Attention.__init__()` with GPU guard |

### 2.2 Missing Implementations (vs. Paper)

| What | Paper Reference | Status Before | Fix |
|------|----------------|---------------|-----|
| EAP baseline | Table 1 | Not in codebase | `pahq/eap.py` |
| Hanna 2024 faithfulness | Table 5 (Appendix) | Not implemented | `pahq/faithfulness.py` |
| Local AUC-ROC computation | All tables | wandb-dependent only | `pahq/roc_local.py` |
| RTN-Q as standalone module | Table 1 | Scattered, not reusable | `pahq/rtn_quantizer.py` |
| W_O FP32 for selected head | Section 4.2 ("output projection ... FP32") | Uses full `self.W_O` | Fixed in `components.py` |

### 2.3 Code Quality Issues

| File | Issue |
|------|-------|
| `notebooks/roc_plot_generator.py` L29-34 | `sys.path.append("~/autodl-tmp/acdc_q/")` - hardcoded server path |
| `acdc/main.py` L156 | `default="/root/autodl-tmp/wandb"` - hardcoded server path |
| `docstring/untitled.py` | Misnamed, no `__main__` guard |
| `cmapy.py` (root) | Matplotlib compatibility utility misplaced at repo root |
| `GEMM.py`, `components.py` | Mixed Chinese/English comments in production code |
| `acdc/ioi/utils.py` | Commented-out code, Chinese inline comments |

---

## 3. New Directory Structure

```
PAHQ/
├── IMPLEMENTATION_PLAN.md       (this file)
├── README.md
├── requirements.txt
├── docs/
│   └── reference_sketch.py      (moved from PAHQ_ACDC/reference.py, with bug notes)
├── transformer_lens/
│   └── transformer_lens/
│       ├── components.py        (PAHQ Attention - fixed)
│       ├── GEMM.py              (Triton kernels - CPU guard added)
│       └── ...
└── PAHQ_ACDC/
    ├── acdc/                    (core ACDC framework - minimal changes)
    │   ├── main.py              (fixed wandb-dir default)
    │   ├── TLACDCExperiment.py
    │   ├── acdc_utils.py
    │   ├── docstring/
    │   │   ├── utils.py
    │   │   ├── prompts.py
    │   │   └── generate_dataset.py   (renamed from untitled.py)
    │   └── ...
    ├── pahq/                    (NEW: PAHQ-specific additions)
    │   ├── __init__.py
    │   ├── eap.py               (EAP baseline: gradient-based edge attribution)
    │   ├── faithfulness.py      (Hanna et al. 2024 faithfulness metric)
    │   ├── roc_local.py         (local AUC-ROC, no wandb dependency)
    │   └── rtn_quantizer.py     (RTN-Q: round-to-nearest quantization module)
    ├── experiments/
    │   ├── ablations/
    │   │   ├── stream_ablation.py     (scheduler on/off combinations)
    │   │   └── precision_ablation.py  (4/8/16-bit comparison)
    │   └── run_pahq.sh          (env-agnostic experiment runner)
    ├── notebooks/
    │   └── roc_plot_generator.py    (fixed: no hardcoded paths)
    └── tests/
        ├── acdc/                (existing ACDC tests)
        └── pahq/                (NEW: CPU-only unit tests)
            ├── conftest.py
            ├── test_fp8_manager.py
            ├── test_rtn_quantizer.py
            ├── test_eap.py
            ├── test_faithfulness.py
            └── test_roc_local.py
```

---

## 4. Implementation Details

### 4.1 EAP Baseline (`pahq/eap.py`)

Edge Attribution Patching (Syed et al. 2023) estimates edge importance via a first-order Taylor approximation:

```
score(e: u→v) = (x_u^clean - x_u^corrupt) · ∂L/∂x_v |_{x_v = x_v^corrupt}
```

This requires:
1. Forward pass on clean input → record activations at all sender hook points
2. Forward pass on corrupted input → record activations + gradients of metric w.r.t. all receiver hook points
3. Multiply element-wise: `score = (clean_act - corrupt_act) * grad`

Key implementation notes:
- Requires `torch.enable_grad()` context (ACDC defaults to `no_grad`)
- Hook names must match ACDC's `TorchIndex`-keyed correspondence exactly
- Returns `Dict[Tuple, float]` matching `Subgraph` type from `TLACDCExperiment.py`

### 4.2 Faithfulness Metric (`pahq/faithfulness.py`)

Hanna et al. 2024 faithfulness formula:

```
F(C) = (metric(C) - metric(∅)) / (metric(G) - metric(∅))
```

where `metric(C)` = task metric evaluated with only circuit C edges active.

Implementation via `TLACDCExperiment.call_metric_with_corr()` which already handles setting up hooks for a given correspondence. The three evaluations (empty, circuit, full) each call `exp.call_metric_with_corr` with appropriate correspondences.

### 4.3 Local AUC-ROC (`pahq/roc_local.py`)

The existing `roc_plot_generator.py:get_points()` function correctly computes TPR/FPR from a list of `(corr, score_d)` pairs. This logic is extracted into a standalone function.

AUC computation: `numpy.trapz(y=tpr_list, x=fpr_list)` after sorting by FPR. The "pessimistic" AUC from the paper uses step segments between Pareto frontier points (already implemented in `get_points` via the `decreasing` parameter).

### 4.4 RTN-Q Module (`pahq/rtn_quantizer.py`)

Round-to-nearest quantization for comparison baseline:

```python
scale = max(|w|) / (2^(bits-1) - 1)     # per-tensor symmetric
q = clip(round(w / scale), -(2^(bits-1)), 2^(bits-1)-1)
dq = q * scale                            # dequantized
```

Two use cases:
1. **Static**: quantize all weights before ACDC runs (matches paper's RTN-Q baseline)
2. **Dynamic**: quantize non-selected heads on each ACDC step (experimental)

---

## 5. Proposed Additional Experiments

### 5.1 FP8 Format Comparison: E4M3 vs E5M2

**Motivation**: The paper claims E4M3 (3-bit mantissa, higher precision, lower range) is better than naive RTN for ACDC because mantissa precision matters more than range. E5M2 (2-bit mantissa, higher range) would test this claim directly.

**Protocol**:
- Run PAHQ-ACDC with `FP8Manager` using `torch.float8_e5m2` instead of `torch.float8_e4m3fn`
- Compare faithfulness (AUC-ROC) across IOI, Docstring, Greater-Than
- Expected: E4M3 ≥ E5M2 in faithfulness, since mantissa loss is the dominant failure mode

**Implementation**: Add `format: str = "e4m3"` parameter to `FP8Manager.__init__()` in `components.py`.

### 5.2 Activation Magnitude Distribution Analysis

**Motivation**: The mantissa loss claim depends on activation magnitudes fitting within FP8 E4M3 range [−448, 448]. This should be empirically verified.

**Protocol**:
- Load gpt2/attn-4l/redwood-2l
- For each attention layer: collect W_Q, W_K, W_V statistics (max, mean, std, percentiles)
- Compare vs FP8 E4M3 dynamic range; plot histogram of quantization errors
- Expected: most weights within E4M3 range; quantization error < 1% for >99th percentile

**Output**: Appendix figure showing weight distributions + quantization error CDF.

### 5.3 MLP Quantization Sensitivity

**Motivation**: PAHQ quantizes only attention weights. The paper uses BF16 for MLP layers (mentioned in Section 4.1) but doesn't ablate this choice.

**Protocol**:
- Run PAHQ-ACDC with MLP weights at FP32 (baseline), BF16, INT8
- Compare faithfulness and runtime
- Expected: BF16 MLP has negligible faithfulness loss, FP8 MLP may degrade

**Implementation**: Add `quantize_mlp: bool = False` flag; when enabled, temporarily cast `MLP.W_in`/`W_out` to bfloat16 in `MLP.forward()`.

### 5.4 Cross-Threshold Circuit Stability

**Motivation**: PAHQ introduces quantization noise that may cause circuit inconsistency across ACDC thresholds.

**Protocol**:
- For each method (ACDC, PAHQ) and threshold pair (τ₁, τ₂) in {0.001, 0.005, 0.01, 0.05, 0.1}:
  - Compute circuit C(τ₁) and C(τ₂) as edge sets
  - Compute Jaccard similarity J = |C(τ₁) ∩ C(τ₂)| / |C(τ₁) ∪ C(τ₂)|
- Report mean Jaccard for ACDC vs. PAHQ
- Expected: PAHQ has slightly lower Jaccard (due to quantization noise) but within acceptable range

### 5.5 PAHQ-EAP Hybrid (New Method Proposal)

**Motivation**: Could PAHQ's high-precision activation computation improve EAP's gradient estimates?

**Concept**: Replace EAP's standard forward/backward pass with PAHQ's mixed-precision forward. The gradient `∂L/∂x_v` is computed at full precision for the selected edge, while others use FP8.

**Expected benefit**: More accurate gradient estimates for EAP, reducing its approximation error on nonlinear interactions (the main weakness cited in the paper).

---

## 6. Unit Test Strategy

All tests in `tests/pahq/` must pass on CPU without GPU. Approach:
- Mock tensors: batch=2, seq_len=4, n_heads=2, d_model=8, d_head=4
- Skip GPU-specific tests with `pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")`
- Skip FP8 dtype tests if `torch < 2.1` or CPU doesn't support `torch.float8_e4m3fn`

Test coverage targets:
- `FP8Manager`: shape preservation, dtype output, round-trip bounds
- `RTNQuantizer`: quantization error bounds, range clamping, weight splitting
- `eap_scores()`: dict output, finite values, zero score for zero corruption
- `compute_faithfulness()`: boundary conditions (0 for empty, 1 for full circuit)
- `compute_auc()`: known AUC values for simple (0,0)→(1,1) and perfect classifier cases

---

## 7. Implementation Status

| Item | Status | File |
|------|--------|------|
| GEMM.py CPU guard (triton import) | ✅ Done | `transformer_lens/transformer_lens/GEMM.py` |
| components.py stream init bug | ✅ Done | `transformer_lens/transformer_lens/components.py` |
| main.py hardcoded wandb path | ✅ Done | `PAHQ_ACDC/acdc/main.py` |
| roc_plot_generator.py hardcoded paths | ✅ Done | `PAHQ_ACDC/notebooks/roc_plot_generator.py` |
| docs/reference_sketch.py (bugs documented) | ✅ Done | `docs/reference_sketch.py` |
| pahq/rtn_quantizer.py | ✅ Done | `PAHQ_ACDC/pahq/rtn_quantizer.py` |
| pahq/eap.py | ✅ Done | `PAHQ_ACDC/pahq/eap.py` |
| pahq/faithfulness.py | ✅ Done | `PAHQ_ACDC/pahq/faithfulness.py` |
| pahq/roc_local.py | ✅ Done | `PAHQ_ACDC/pahq/roc_local.py` |
| tests/pahq/conftest.py | ✅ Done | `PAHQ_ACDC/tests/pahq/conftest.py` |
| tests/pahq/test_rtn_quantizer.py | ✅ Done | `PAHQ_ACDC/tests/pahq/test_rtn_quantizer.py` |
| tests/pahq/test_eap.py | ✅ Done | `PAHQ_ACDC/tests/pahq/test_eap.py` |
| tests/pahq/test_faithfulness.py | ✅ Done | `PAHQ_ACDC/tests/pahq/test_faithfulness.py` |
| tests/pahq/test_roc_local.py | ✅ Done | `PAHQ_ACDC/tests/pahq/test_roc_local.py` |
| docstring/generate_dataset.py (renamed) | ✅ Done | `PAHQ_ACDC/acdc/docstring/generate_dataset.py` |
| tests/pahq/test_fp8_manager.py | ⬜ Pending | (requires GPU for meaningful tests) |
| experiments/ablations/stream_ablation.py | ⬜ Pending | |
| experiments/ablations/precision_ablation.py | ⬜ Pending | |
| Clean Chinese comments from production code | ⬜ Pending | `GEMM.py`, `components.py` |
