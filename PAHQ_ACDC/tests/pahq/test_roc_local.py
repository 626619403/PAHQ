"""Unit tests for pahq.roc_local."""

import math
import pytest
import numpy as np

from pahq.roc_local import RocEvaluator, default_thresholds


# ---------------------------------------------------------------------------
# Helpers: lightweight mock objects (no ACDC imports needed)
# ---------------------------------------------------------------------------

class MockEdge:
    def __init__(self, present=True):
        self.present = present


class MockCorr:
    """Minimal stand-in for TLACDCCorrespondence for ROC tests.

    Edges are stored as a flat dict {key: MockEdge}.
    We implement only the methods used by RocEvaluator.
    """

    def __init__(self, edges: dict):
        self._edges = edges  # {key: MockEdge}

    def count_no_edges(self):
        return sum(1 for e in self._edges.values() if e.present)

    def all_edges(self):
        return self._edges


def _make_stats_fn(tp, fp, fn, tn):
    """Return a mock ``get_edge_stats`` result."""
    return {
        "true positive": tp,
        "false positive": fp,
        "false negative": fn,
        "true negative": tn,
        "all": tp + fp + fn + tn,
        "ground truth": tp + fn,
        "recovered": tp + fp,
    }


# ---------------------------------------------------------------------------
# default_thresholds
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_length(self):
        assert len(default_thresholds(21)) == 21

    def test_custom_length(self):
        assert len(default_thresholds(5)) == 5

    def test_monotone_increasing(self):
        thrs = default_thresholds()
        assert all(thrs[i] < thrs[i + 1] for i in range(len(thrs) - 1))

    def test_range(self):
        thrs = default_thresholds()
        assert abs(thrs[0] - 0.001) < 1e-6
        assert abs(thrs[-1] - 10 ** 0.5) < 0.01


# ---------------------------------------------------------------------------
# RocEvaluator.auc
# ---------------------------------------------------------------------------

class TestRocAuc:
    def test_perfect_classifier(self):
        # TPR=1 everywhere except fpr=0 → AUC = 1
        points = [
            {"edge_fpr": 0.0, "edge_tpr": 0.0},
            {"edge_fpr": 0.0, "edge_tpr": 1.0},
            {"edge_fpr": 1.0, "edge_tpr": 1.0},
        ]
        auc = RocEvaluator.auc(points)
        assert abs(auc - 1.0) < 1e-6

    def test_random_classifier(self):
        # Diagonal line → AUC = 0.5
        points = [
            {"edge_fpr": 0.0, "edge_tpr": 0.0},
            {"edge_fpr": 0.5, "edge_tpr": 0.5},
            {"edge_fpr": 1.0, "edge_tpr": 1.0},
        ]
        auc = RocEvaluator.auc(points)
        assert abs(auc - 0.5) < 1e-6

    def test_worst_classifier(self):
        # TPR=0 everywhere until fpr=1 → AUC = 0
        points = [
            {"edge_fpr": 0.0, "edge_tpr": 0.0},
            {"edge_fpr": 1.0, "edge_tpr": 0.0},
            {"edge_fpr": 1.0, "edge_tpr": 1.0},
        ]
        auc = RocEvaluator.auc(points)
        assert abs(auc - 0.0) < 1e-6

    def test_auc_bounded(self):
        # Generate random monotone-ish curve and check 0 ≤ AUC ≤ 1
        rng = np.random.default_rng(0)
        fprs = np.sort(rng.uniform(0, 1, 10))
        tprs = np.sort(rng.uniform(0, 1, 10))
        points = [{"edge_fpr": f, "edge_tpr": t} for f, t in zip(fprs, tprs)]
        auc = RocEvaluator.auc(points)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# RocEvaluator.threshold_sweep (uses mock get_edge_stats via monkeypatch)
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    """Tests for the static threshold_sweep method.

    We can't easily construct real TLACDCCorrespondence objects in unit tests,
    so we test the AUC + anchor-point logic at a higher level through
    get_points-style checks.
    """

    def test_default_thresholds_length_and_monotone(self):
        thrs = default_thresholds(10)
        assert len(thrs) == 10
        for i in range(len(thrs) - 1):
            assert thrs[i] < thrs[i + 1]

    def test_auc_increases_with_better_scores(self):
        """Check that a 'better' score dict yields higher AUC."""
        # Perfect separation: all true positives score >> threshold
        perfect_points = [
            {"edge_fpr": 0.0, "edge_tpr": 0.0},
            {"edge_fpr": 0.0, "edge_tpr": 1.0},
            {"edge_fpr": 1.0, "edge_tpr": 1.0},
        ]
        random_points = [
            {"edge_fpr": 0.0, "edge_tpr": 0.0},
            {"edge_fpr": 0.5, "edge_tpr": 0.5},
            {"edge_fpr": 1.0, "edge_tpr": 1.0},
        ]
        assert RocEvaluator.auc(perfect_points) > RocEvaluator.auc(random_points)


# ---------------------------------------------------------------------------
# Integration: sweep + auc with mock correspondences
# ---------------------------------------------------------------------------

class TestSweepIntegration:
    """Lightweight integration test using mock correspondences.

    We monkey-patch ``acdc.acdc_utils.get_edge_stats`` to avoid needing a
    real model / correspondence.
    """

    def _make_mock_corr(self, n_present):
        edges = {f"e{i}": MockEdge(i < n_present) for i in range(10)}
        return MockCorr(edges)

    def test_sweep_produces_anchor_points(self, monkeypatch):
        import pahq.roc_local as roc_mod

        call_count = [0]

        def fake_get_edge_stats(ground_truth, recovered):
            # Return stats where all recovered edges are TPs
            n = recovered.count_no_edges()
            call_count[0] += 1
            return {
                "true positive": n,
                "false positive": 0,
                "false negative": 5 - n,
                "true negative": 5,
                "all": 10,
                "ground truth": 5,
                "recovered": n,
            }

        monkeypatch.setattr(roc_mod, "get_edge_stats", fake_get_edge_stats)

        gt = self._make_mock_corr(5)
        full = self._make_mock_corr(10)
        evaluator = RocEvaluator(ground_truth=gt, full_corr=full)

        pairs = [
            (self._make_mock_corr(2), {"score": 0.3}),
            (self._make_mock_corr(5), {"score": 0.1}),
            (self._make_mock_corr(8), {"score": 0.7}),
        ]
        points = evaluator.sweep(pairs, decreasing=True)

        # Should have: initial anchor + 3 pairs + terminal anchor = 5
        assert len(points) == 5
        assert points[0]["edge_fpr"] == 0.0
        assert points[0]["edge_tpr"] == 0.0
        assert points[-1]["edge_fpr"] == 1.0
        assert points[-1]["edge_tpr"] == 1.0

    def test_auc_of_perfect_sweep_is_one(self, monkeypatch):
        import pahq.roc_local as roc_mod

        def fake_get_edge_stats(ground_truth, recovered):
            n = recovered.count_no_edges()
            tp = min(n, 5)
            fp = max(0, n - 5)
            return {
                "true positive": tp,
                "false positive": fp,
                "false negative": 5 - tp,
                "true negative": 5 - fp,
                "all": 10,
                "ground truth": 5,
                "recovered": n,
            }

        monkeypatch.setattr(roc_mod, "get_edge_stats", fake_get_edge_stats)

        gt = self._make_mock_corr(5)
        full = self._make_mock_corr(10)
        evaluator = RocEvaluator(ground_truth=gt, full_corr=full)

        # Perfect ordering: first recover all 5 TPs, then FPs
        pairs = [
            (self._make_mock_corr(1), {"score": 5.0}),
            (self._make_mock_corr(2), {"score": 4.0}),
            (self._make_mock_corr(3), {"score": 3.0}),
            (self._make_mock_corr(4), {"score": 2.0}),
            (self._make_mock_corr(5), {"score": 1.0}),
        ]
        points = evaluator.sweep(pairs, decreasing=True)
        auc = RocEvaluator.auc(points)
        assert auc >= 0.9  # near 1.0 (anchors prevent exact 1.0 in this setup)
