"""Unit tests for pahq.eap.

The EAP algorithm requires a real HookedTransformer and gradient-enabled
forward passes.  These tests use a tiny 1-layer, 1-head transformer so they
run quickly on CPU.  GPU-specific behaviour is not tested here.
"""

import pytest
import torch

from pahq.eap import _apply_threshold, compute_eap_scores, run_eap


# ---------------------------------------------------------------------------
# _apply_threshold (pure logic, no model needed)
# ---------------------------------------------------------------------------

class MockEdgeType:
    """Mirror of EdgeType values without importing acdc."""
    ADDITION = 0
    DIRECT_COMPUTATION = 1
    PLACEHOLDER = 2


class MockEdge:
    def __init__(self, edge_type_val, present=True):
        # Store as an int-comparable value to avoid acdc import
        self._type_val = edge_type_val
        self.present = present
        self.effect_size = None

    @property
    def edge_type(self):
        return self

    @property
    def value(self):
        return self._type_val

    def __eq__(self, other):
        if hasattr(other, "value"):
            return self._type_val == other.value
        return NotImplemented


class MockTorchIndex:
    def __init__(self, idx):
        self.hashable_tuple = (idx,)
        self.as_index = (slice(None),)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple


def _make_mock_corr(scores_and_types: dict):
    """Build a minimal correspondence-like object for _apply_threshold tests.

    scores_and_types : dict mapping key → (score, edge_type_val)
    """
    from types import SimpleNamespace

    edges_flat = {}
    for key, (_, etype) in scores_and_types.items():
        child_name, child_idx, parent_name, parent_idx = key
        if child_name not in edges_flat:
            edges_flat[child_name] = {}
        ci = MockTorchIndex(child_idx)
        if ci not in edges_flat[child_name]:
            edges_flat[child_name][ci] = {}
        if parent_name not in edges_flat[child_name][ci]:
            edges_flat[child_name][ci][parent_name] = {}
        pi = MockTorchIndex(parent_idx)
        edges_flat[child_name][ci][parent_name][pi] = MockEdge(etype)

    class FakeCorr:
        def __init__(self):
            self.edges = edges_flat

    return FakeCorr()


class TestApplyThreshold:
    def _build_scores_and_corr(self):
        # 3 ADDITION edges with different scores + 1 PLACEHOLDER
        keys_types = {
            ("n1", 0, "n0", 0): (1.5, MockEdgeType.ADDITION),
            ("n1", 0, "n0", 1): (0.3, MockEdgeType.ADDITION),
            ("n1", 0, "n0", 2): (0.05, MockEdgeType.DIRECT_COMPUTATION),
            ("n2", 0, "n1", 0): (0.0, MockEdgeType.PLACEHOLDER),
        }
        scores = {}
        for key, (score, _) in keys_types.items():
            child_name, child_idx, parent_name, parent_idx = key
            ci = MockTorchIndex(child_idx)
            pi = MockTorchIndex(parent_idx)
            scores[(child_name, ci, parent_name, pi)] = score

        corr = _make_mock_corr(keys_types)
        return corr, scores

    def test_placeholder_always_present(self):
        corr, scores = self._build_scores_and_corr()

        # Replace scores with correct key types
        from types import SimpleNamespace
        # Build scores with MockTorchIndex keys
        proper_scores = {}
        for (cn, ci_int, pn, pi_int), sc in [
            ("n1", 0, "n0", 0), ("n1", 0, "n0", 1),
            ("n1", 0, "n0", 2), ("n2", 0, "n1", 0),
        ]:
            pass  # already done above

        _apply_threshold(corr, scores, threshold=1.0, abs_value=True)
        # PLACEHOLDER edge should always be present
        placeholder = corr.edges["n2"][MockTorchIndex(0)]["n1"][MockTorchIndex(0)]
        assert placeholder.present is True

    def test_high_threshold_removes_low_score_edges(self):
        corr, scores = self._build_scores_and_corr()
        _apply_threshold(corr, scores, threshold=1.0, abs_value=True)

        # score=0.3 < 1.0 → absent
        edge_low = corr.edges["n1"][MockTorchIndex(0)]["n0"][MockTorchIndex(1)]
        assert edge_low.present is False

        # score=1.5 >= 1.0 → present
        edge_high = corr.edges["n1"][MockTorchIndex(0)]["n0"][MockTorchIndex(0)]
        assert edge_high.present is True

    def test_zero_threshold_keeps_all_non_placeholder(self):
        corr, scores = self._build_scores_and_corr()
        _apply_threshold(corr, scores, threshold=0.0, abs_value=True)

        # score=0.05 >= 0.0 → present
        edge = corr.edges["n1"][MockTorchIndex(0)]["n0"][MockTorchIndex(2)]
        assert edge.present is True

    def test_effect_size_stored(self):
        corr, scores = self._build_scores_and_corr()
        _apply_threshold(corr, scores, threshold=0.0, abs_value=True)

        edge = corr.edges["n1"][MockTorchIndex(0)]["n0"][MockTorchIndex(0)]
        assert edge.effect_size == 1.5

    def test_abs_value_false_uses_raw_score(self):
        corr, scores = self._build_scores_and_corr()
        # Make score negative
        for k in scores:
            scores[k] = -abs(scores[k])
        # With abs_value=False and threshold=0.5, negative scores are always < 0.5
        _apply_threshold(corr, scores, threshold=0.5, abs_value=False)
        # Every non-placeholder edge should be absent
        for ci_int in range(3):
            edge = corr.edges["n1"][MockTorchIndex(0)]["n0"][MockTorchIndex(ci_int)]
            assert edge.present is False


# ---------------------------------------------------------------------------
# Integration smoke test (CPU, tiny model)
# ---------------------------------------------------------------------------

try:
    from transformer_lens import HookedTransformer
    _TL_AVAILABLE = True
except Exception:
    _TL_AVAILABLE = False


@pytest.mark.skipif(not _TL_AVAILABLE, reason="transformer_lens not available")
class TestEapIntegration:
    """Smoke tests that the EAP pipeline runs end-to-end on CPU."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        torch.autograd.set_grad_enabled(True)
        yield
        torch.autograd.set_grad_enabled(False)

    def test_compute_eap_scores_returns_dict(self):
        """Minimal check: compute_eap_scores doesn't crash and returns a dict."""
        from acdc.induction.utils import get_all_induction_things
        from acdc.TLACDCExperiment import TLACDCExperiment

        things = get_all_induction_things(
            num_examples=2,
            seq_len=10,
            device="cpu",
            metric="kl_div",
            model_name="attn-only-2l",
        )
        tl_model = things.tl_model

        exp = TLACDCExperiment(
            model=tl_model,
            ds=things.validation_data,
            ref_ds=things.validation_patch_data,
            threshold=0.5,
            metric=things.validation_metric,
            test_metric=things.test_metrics.get("kl_div", things.validation_metric),
            add_sender_hooks=True,
            add_receiver_hooks=False,
        )

        torch.autograd.set_grad_enabled(True)
        scores = compute_eap_scores(
            model=tl_model,
            corr=exp.corr,
            clean_data=things.validation_data[:2],
            corrupted_data=things.validation_patch_data[:2],
            metric=things.validation_metric,
        )
        assert isinstance(scores, dict)
        assert len(scores) > 0
