"""Unit tests for pahq.faithfulness.

These tests use mock TLACDCExperiment objects to avoid requiring a loaded
language model.  All tests run on CPU.
"""

import math
import pytest
import torch
from unittest.mock import MagicMock, patch

from pahq.faithfulness import FaithfulnessEvaluator, faithfulness


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockEdge:
    def __init__(self, present=True):
        self.present = present


class MockCorr:
    def __init__(self, n_edges=5):
        self._n_edges = n_edges
        self._edges = {i: MockEdge(True) for i in range(n_edges)}

    def all_edges(self):
        return self._edges

    def count_no_edges(self):
        return sum(1 for e in self._edges.values() if e.present)


def _make_mock_exp(metric_values: dict):
    """Build a mock TLACDCExperiment.

    Parameters
    ----------
    metric_values : dict
        Maps (full|empty|circuit) → float metric value.
    """
    exp = MagicMock()
    exp.ds = torch.zeros(2, 4, dtype=torch.long)
    exp.corr = MockCorr(n_edges=10)

    call_count = [0]

    def fake_metric(output):
        val = list(metric_values.values())[call_count[0] % len(metric_values)]
        call_count[0] += 1
        return torch.tensor(val)

    exp.metric = fake_metric

    # setup_model_hooks is called in _eval_corr; just a no-op here
    exp.setup_model_hooks = MagicMock()
    exp.model = MagicMock()
    exp.model.reset_hooks = MagicMock()
    exp.model.return_value = torch.zeros(2, 4, 50)

    return exp, call_count


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator
# ---------------------------------------------------------------------------

class TestFaithfulnessEvaluator:
    def _build_evaluator_with_scores(self, full_score, empty_score, circuit_score):
        """Return an evaluator that returns known scores for each call."""
        ev = MagicMock(spec=FaithfulnessEvaluator)
        ev.score_full_model = MagicMock(return_value=full_score)
        ev.score_empty_model = MagicMock(return_value=empty_score)
        ev._eval_corr = MagicMock(return_value=circuit_score)
        ev._score_full = None
        ev._score_empty = None
        # Manually call the real evaluate logic
        ev.evaluate = lambda corr: FaithfulnessEvaluator.evaluate(ev, corr)
        return ev

    def test_perfect_circuit_gives_one(self):
        # m(C) = m(G) → F = 1.0
        ev = self._build_evaluator_with_scores(
            full_score=0.1, empty_score=1.0, circuit_score=0.1
        )
        score = ev.evaluate(MockCorr())
        assert abs(score - 1.0) < 1e-6

    def test_empty_circuit_gives_zero(self):
        # m(C) = m(∅) → F = 0.0
        ev = self._build_evaluator_with_scores(
            full_score=0.1, empty_score=1.0, circuit_score=1.0
        )
        score = ev.evaluate(MockCorr())
        assert abs(score - 0.0) < 1e-6

    def test_intermediate_circuit(self):
        # m(G)=0, m(∅)=1, m(C)=0.4 → F = (0.4-1)/(0-1) = 0.6
        ev = self._build_evaluator_with_scores(
            full_score=0.0, empty_score=1.0, circuit_score=0.4
        )
        score = ev.evaluate(MockCorr())
        assert abs(score - 0.6) < 1e-6

    def test_degenerate_denominator_returns_nan(self):
        # m(G) == m(∅) → NaN
        ev = self._build_evaluator_with_scores(
            full_score=0.5, empty_score=0.5, circuit_score=0.3
        )
        score = ev.evaluate(MockCorr())
        assert math.isnan(score)

    def test_circuit_worse_than_empty_gives_negative(self):
        # m(C) > m(∅) → F < 0
        ev = self._build_evaluator_with_scores(
            full_score=0.0, empty_score=1.0, circuit_score=1.5
        )
        score = ev.evaluate(MockCorr())
        assert score < 0.0

    def test_circuit_better_than_full_gives_greater_than_one(self):
        # m(C) < m(G) → F > 1 (circuit is somehow better than full model)
        ev = self._build_evaluator_with_scores(
            full_score=0.1, empty_score=1.0, circuit_score=0.05
        )
        score = ev.evaluate(MockCorr())
        assert score > 1.0


class TestFaithfulnessMakeCorr:
    """Test that _make_full_corr and _make_empty_corr produce correct corrs."""

    def _make_evaluator_with_corr(self, corr):
        exp = MagicMock()
        exp.corr = corr
        ev = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        ev.exp = exp
        ev.metric_fn = MagicMock(return_value=torch.tensor(0.0))
        ev.data = torch.zeros(2, 4, dtype=torch.long)
        ev._score_full = None
        ev._score_empty = None
        return ev

    def test_make_full_corr_all_edges_present(self):
        corr = MockCorr(n_edges=5)
        # Set some to False
        for k in list(corr._edges)[:3]:
            corr._edges[k].present = False
        ev = self._make_evaluator_with_corr(corr)
        full = ev._make_full_corr()
        assert all(e.present for e in full.all_edges().values())

    def test_make_empty_corr_no_edges_present(self):
        corr = MockCorr(n_edges=5)
        ev = self._make_evaluator_with_corr(corr)
        empty = ev._make_empty_corr()
        assert all(not e.present for e in empty.all_edges().values())

    def test_make_full_does_not_mutate_original(self):
        corr = MockCorr(n_edges=5)
        for k in list(corr._edges)[:3]:
            corr._edges[k].present = False
        ev = self._make_evaluator_with_corr(corr)
        _ = ev._make_full_corr()
        # Original still has 3 edges False
        assert sum(1 for e in corr._edges.values() if not e.present) == 3


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestFaithfulnessFunction:
    def test_returns_float(self):
        ev = MagicMock(spec=FaithfulnessEvaluator)
        ev.evaluate = MagicMock(return_value=0.75)
        with patch("pahq.faithfulness.FaithfulnessEvaluator", return_value=ev):
            result = faithfulness(
                exp=MagicMock(),
                corr=MockCorr(),
            )
        assert isinstance(result, float)
        assert result == 0.75
