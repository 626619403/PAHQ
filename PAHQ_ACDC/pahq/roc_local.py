"""Local AUC-ROC computation — no wandb dependency.

This module provides utilities to:
1. Sweep a threshold range over a scored correspondence (or a list of
   (corr, score) pairs) and compute TPR / FPR at each threshold.
2. Compute the area under the ROC curve (AUC) via the trapezoidal rule.
3. Optionally plot the ROC curve with matplotlib.

The design mirrors the logic in ``notebooks/roc_plot_generator.py`` but is
self-contained, works offline, and is unit-testable.

Typical usage
-------------
>>> from pahq.roc_local import RocEvaluator
>>> ev = RocEvaluator(ground_truth_corr, max_subgraph_corr)
>>> points = ev.sweep(corrs_and_scores)   # list[(corr, {"score": float})]
>>> auc = ev.auc(points)
>>> ev.plot(points)  # optional matplotlib plot
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.acdc_utils import get_edge_stats, get_node_stats

# EdgeKey = (child_name, child_index, parent_name, parent_index)
EdgeKey = Tuple

# A point on the ROC curve contains at minimum "edge_tpr" and "edge_fpr".
RocPoint = Dict[str, float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RocEvaluator:
    """Sweep a list of (correspondence, score_dict) pairs and compute ROC.

    Parameters
    ----------
    ground_truth : TLACDCCorrespondence
        The canonical / reference circuit.  Edges present here are *true
        positives* in the ground truth.
    full_corr : TLACDCCorrespondence
        The complete graph (all edges present).  Determines the denominator
        for FPR.
    """

    def __init__(
        self,
        ground_truth: TLACDCCorrespondence,
        full_corr: TLACDCCorrespondence,
    ) -> None:
        self.ground_truth = ground_truth
        self.full_corr = full_corr
        self._n_edges_total = full_corr.count_no_edges()
        self._n_edges_gt = ground_truth.count_no_edges()

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        corrs_and_scores: Sequence[Tuple[TLACDCCorrespondence, Dict]],
        decreasing: bool = True,
    ) -> List[RocPoint]:
        """Compute ROC points for an ordered list of (corr, score_dict) pairs.

        Parameters
        ----------
        corrs_and_scores : list of (TLACDCCorrespondence, dict)
            Each element is a (correspondence, score_dict) pair.  The
            ``score_dict`` must contain a ``"score"`` key used for ordering.
        decreasing : bool
            If True, higher score → more edges present (ACDC threshold
            convention: decrease threshold to include more edges).

        Returns
        -------
        List of dicts with keys:
          ``edge_tpr``, ``edge_fpr``, ``edge_precision``,
          ``n_edges``, ``score``.
        """
        # Anchor points
        points: List[RocPoint] = [
            {"edge_tpr": 0.0, "edge_fpr": 0.0, "edge_precision": 1.0,
             "n_edges": 0, "score": math.inf if decreasing else -math.inf},
        ]

        sorted_pairs = sorted(
            corrs_and_scores,
            key=lambda x: x[1]["score"],
            reverse=decreasing,
        )

        for corr, score_d in sorted_pairs:
            stats = get_edge_stats(
                ground_truth=self.ground_truth,
                recovered=corr,
            )
            tp = stats["true positive"]
            fp = stats["false positive"]
            fn = stats["false negative"]
            tn = stats["true negative"]

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0

            point: RocPoint = {
                "edge_tpr": tpr,
                "edge_fpr": fpr,
                "edge_precision": prec,
                "n_edges": corr.count_no_edges(),
                "score": score_d["score"],
            }
            point.update({k: v for k, v in score_d.items() if k != "score"})
            points.append(point)

        # Terminal anchor
        points.append(
            {"edge_tpr": 1.0, "edge_fpr": 1.0, "edge_precision": 0.0,
             "n_edges": self._n_edges_total,
             "score": -math.inf if decreasing else math.inf},
        )
        return points

    # ------------------------------------------------------------------
    # AUC
    # ------------------------------------------------------------------

    @staticmethod
    def auc(points: List[RocPoint]) -> float:
        """Compute AUC-ROC from a list of RocPoints via the trapezoidal rule.

        Parameters
        ----------
        points : list of dicts
            Output of :meth:`sweep`.  Must contain ``edge_fpr`` and
            ``edge_tpr`` keys.

        Returns
        -------
        float in [0, 1]
        """
        fprs = [p["edge_fpr"] for p in points]
        tprs = [p["edge_tpr"] for p in points]
        return float(np.trapz(tprs, fprs))

    # ------------------------------------------------------------------
    # Threshold sweep (for a single scored correspondence)
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_sweep(
        base_corr: TLACDCCorrespondence,
        edge_scores: Dict[EdgeKey, float],
        thresholds: Sequence[float],
        ground_truth: TLACDCCorrespondence,
        abs_value: bool = True,
    ) -> List[RocPoint]:
        """Build a sweep from a dict of per-edge scores + a threshold list.

        Useful when you have EAP scores and want to scan thresholds locally
        without re-running the model.

        Parameters
        ----------
        base_corr : TLACDCCorrespondence
            Full correspondence (serves as template).
        edge_scores : dict
            Maps edge key tuple → float attribution score.
        thresholds : sequence of float
            Thresholds to apply (descending order recommended).
        ground_truth : TLACDCCorrespondence
            Reference circuit for TPR/FPR computation.
        abs_value : bool
            Whether to apply |score| >= threshold.

        Returns
        -------
        List of RocPoints.
        """
        evaluator = RocEvaluator(ground_truth=ground_truth, full_corr=base_corr)
        corrs_and_scores: List[Tuple[TLACDCCorrespondence, Dict]] = []

        for thr in sorted(thresholds, reverse=True):
            corr = deepcopy(base_corr)
            for edge_key, score in edge_scores.items():
                (child_name, child_index, parent_name, parent_index) = edge_key
                try:
                    edge = corr.edges[child_name][child_index][parent_name][parent_index]
                except KeyError:
                    continue
                compare = abs(score) if abs_value else score
                edge.present = compare >= thr
            corrs_and_scores.append((corr, {"score": thr}))

        return evaluator.sweep(corrs_and_scores, decreasing=True)

    # ------------------------------------------------------------------
    # Plotting (optional — gracefully skips if matplotlib unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def plot(
        points: List[RocPoint],
        label: str = "ROC",
        ax=None,
        show: bool = False,
    ):
        """Plot the ROC curve.

        Parameters
        ----------
        points : list of RocPoints
        label : str
            Legend label.
        ax : matplotlib Axes, optional
            Plot into this axes.  If None, use ``plt.gca()``.
        show : bool
            Call ``plt.show()`` after plotting.

        Returns
        -------
        ax : matplotlib Axes, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if ax is None:
            ax = plt.gca()

        fprs = [p["edge_fpr"] for p in points]
        tprs = [p["edge_tpr"] for p in points]
        auc_val = RocEvaluator.auc(points)

        ax.plot(fprs, tprs, label=f"{label} (AUC={auc_val:.3f})", marker=".")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if show:
            plt.show()

        return ax


# ---------------------------------------------------------------------------
# Convenience: default log-spaced threshold grid (as in PAHQ paper)
# ---------------------------------------------------------------------------

def default_thresholds(n: int = 21) -> List[float]:
    """Return *n* logarithmically-spaced threshold values from 0.001 to 3.16.

    These are the 21 values used in the PAHQ paper for the AUC-ROC sweep.
    """
    return list(np.logspace(-3, 0.5, n).tolist())
