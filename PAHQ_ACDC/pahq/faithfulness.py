"""Hanna et al. (2024) faithfulness metric.

Reference:
    Hanna et al. 2024, "How to Evaluate Mechanistic Interpretability?"
    https://arxiv.org/abs/2402.08164

Definition
----------
Given:
  * G  — the full model graph (all edges present)
  * C  — a candidate circuit (a subgraph of G)
  * ∅  — the empty circuit (no edges present, output is corrupted-baseline)
  * m  — a task metric (lower is better, e.g. KL divergence)

The faithfulness of C is:

    F(C) = (m(C) - m(∅)) / (m(G) - m(∅))

A perfectly faithful circuit has F(C) = 1.0.
A circuit that is no better than the empty model has F(C) = 0.0.
Values outside [0, 1] are possible for circuits worse than empty or better
than the full model.

Usage
-----
>>> from pahq.faithfulness import FaithfulnessEvaluator
>>> ev = FaithfulnessEvaluator(exp)
>>> score = ev.evaluate(my_corr)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import torch

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCExperiment import TLACDCExperiment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FaithfulnessEvaluator:
    """Compute Hanna 2024 faithfulness for a given correspondence.

    Parameters
    ----------
    exp : TLACDCExperiment
        A fully-initialised ACDC experiment object.  The evaluator uses
        ``exp.model``, ``exp.metric``, and ``exp.ds`` (clean data).
    metric_fn : callable, optional
        Override the metric used.  Defaults to ``exp.metric``.
    data : Tensor, optional
        Override the data used.  Defaults to ``exp.ds``.
    """

    def __init__(
        self,
        exp: TLACDCExperiment,
        metric_fn: Optional[Callable] = None,
        data: Optional[torch.Tensor] = None,
    ) -> None:
        self.exp = exp
        self.metric_fn = metric_fn if metric_fn is not None else exp.metric
        self.data = data if data is not None else exp.ds

        # Pre-compute reference scores
        self._score_full: Optional[float] = None
        self._score_empty: Optional[float] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def score_full_model(self) -> float:
        """Compute and cache m(G) — metric on the full (all-edges) model."""
        if self._score_full is None:
            self._score_full = self._eval_corr(self._make_full_corr())
        return self._score_full

    def score_empty_model(self) -> float:
        """Compute and cache m(∅) — metric on the empty (no-edges) model."""
        if self._score_empty is None:
            self._score_empty = self._eval_corr(self._make_empty_corr())
        return self._score_empty

    def evaluate(self, corr: TLACDCCorrespondence) -> float:
        """Return the Hanna 2024 faithfulness score F(C) ∈ ℝ.

        Parameters
        ----------
        corr : TLACDCCorrespondence
            The candidate circuit to evaluate.

        Returns
        -------
        float
            Faithfulness score.  Typically in [0, 1]; values outside that
            range indicate a circuit worse than empty or better than full.
        """
        m_full = self.score_full_model()
        m_empty = self.score_empty_model()
        m_circuit = self._eval_corr(corr)

        denom = m_full - m_empty
        if abs(denom) < 1e-12:
            # Degenerate case: full model == empty model on this metric.
            return float("nan")

        return (m_circuit - m_empty) / denom

    def evaluate_batch(
        self,
        corrs: list,
    ) -> list:
        """Evaluate faithfulness for a list of correspondences.

        Returns a list of (corr, faithfulness_score) tuples.
        """
        return [(c, self.evaluate(c)) for c in corrs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_full_corr(self) -> TLACDCCorrespondence:
        """Return a copy of the base correspondence with all edges present."""
        full = deepcopy(self.exp.corr)
        for edge in full.all_edges().values():
            edge.present = True
        return full

    def _make_empty_corr(self) -> TLACDCCorrespondence:
        """Return a copy of the base correspondence with no edges present."""
        empty = deepcopy(self.exp.corr)
        for edge in empty.all_edges().values():
            edge.present = False
        return empty

    def _eval_corr(self, corr: TLACDCCorrespondence) -> float:
        """Run the model with hooks set by *corr* and compute the metric."""
        old_corr = self.exp.corr
        try:
            self.exp.corr = corr
            self.exp.model.reset_hooks()
            self.exp.setup_model_hooks(
                add_sender_hooks=True,
                add_receiver_hooks=True,
                doing_acdc_runs=False,
            )
            with torch.no_grad():
                out = self.exp.model(self.data)
            score = self.metric_fn(out)
            return float(score.item() if isinstance(score, torch.Tensor) else score)
        finally:
            self.exp.corr = old_corr
            self.exp.model.reset_hooks()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def faithfulness(
    exp: TLACDCExperiment,
    corr: TLACDCCorrespondence,
    metric_fn: Optional[Callable] = None,
    data: Optional[torch.Tensor] = None,
) -> float:
    """One-shot Hanna 2024 faithfulness score.

    Equivalent to ``FaithfulnessEvaluator(exp, ...).evaluate(corr)`` but
    computes full/empty baselines from scratch each call.  Use
    ``FaithfulnessEvaluator`` directly if evaluating multiple circuits.
    """
    ev = FaithfulnessEvaluator(exp, metric_fn=metric_fn, data=data)
    return ev.evaluate(corr)
