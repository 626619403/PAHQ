"""Edge Attribution Patching (EAP) baseline.

EAP approximates ACDC's per-edge importance score in a single forward+backward
pass, making it an O(1) alternative to ACDC's O(|E|) algorithm.

Reference:
    Syed et al. 2023, "Attribution Patching Outperforms Automated Circuit Discovery"
    https://arxiv.org/abs/2310.10348

Algorithm
---------
For each edge (u → v) in the computational graph:

    score(u → v)  ≈  E_x[∂L/∂act_v(x) · (act_v(x) - act_v(x'))]

where x is the *clean* input, x' is the *corrupted* input, and act_v is the
activation at node v.  This is a first-order Taylor expansion of the metric
change when the edge is ablated.

Implementation notes
--------------------
* We register forward hooks to cache both clean and corrupted activations.
* We register a backward hook to capture the gradient of the metric w.r.t.
  each activation.
* The product is summed over batch and position dimensions to give a scalar
  score per edge.
* Placeholder edges (EdgeType.PLACEHOLDER) are assigned score 0 and kept
  present (consistent with ACDC's treatment of them).
"""

from __future__ import annotations

import torch
from typing import Callable, Dict, List, Optional, Tuple

from acdc.TLACDCEdge import Edge, EdgeType, TorchIndex
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.HookedTransformer import HookedTransformer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eap(
    model: HookedTransformer,
    corr: TLACDCCorrespondence,
    clean_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
    threshold: float = 0.0,
    abs_value: bool = True,
) -> TLACDCCorrespondence:
    """Run EAP and return a pruned correspondence.

    Parameters
    ----------
    model : HookedTransformer
        The model to analyse.  Must have gradient tracking enabled
        (``torch.autograd.set_grad_enabled(True)``).
    corr : TLACDCCorrespondence
        The *full* (all-present) correspondence to prune.  This object is
        modified in-place and also returned.
    clean_data : Tensor [batch, seq]
        Clean input token IDs.
    corrupted_data : Tensor [batch, seq]
        Corrupted (patched) input token IDs.
    metric : callable
        Maps model output logits → scalar loss.  Must be differentiable.
    threshold : float
        Edges with |score| < threshold are set to ``present=False``.
    abs_value : bool
        Whether to threshold on |score| (True) or raw score (False).

    Returns
    -------
    corr : TLACDCCorrespondence
        The pruned correspondence (same object that was passed in).
    """
    scores = _compute_eap_scores(model, corr, clean_data, corrupted_data, metric)
    _apply_threshold(corr, scores, threshold, abs_value)
    return corr


def compute_eap_scores(
    model: HookedTransformer,
    corr: TLACDCCorrespondence,
    clean_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[Tuple, float]:
    """Compute EAP scores for every edge without pruning.

    Returns
    -------
    scores : dict
        Maps ``(child_name, child_index, parent_name, parent_index)`` tuples
        to float score values.
    """
    return _compute_eap_scores(model, corr, clean_data, corrupted_data, metric)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_eap_scores(
    model: HookedTransformer,
    corr: TLACDCCorrespondence,
    clean_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[Tuple, float]:
    """Core EAP computation.

    Returns a mapping from edge key tuples to float attribution scores.
    """
    # Collect every hook name that appears as a *receiver* (child) node in
    # the correspondence — these are the activations we need to differentiate.
    receiver_hook_names: List[str] = []
    for child_name in corr.edges:
        for child_index in corr.edges[child_name]:
            receiver_hook_names.append(child_name)

    receiver_hook_names = list(dict.fromkeys(receiver_hook_names))  # dedup, preserve order

    # --- Step 1: cache corrupted activations (no grad needed) ---------------
    corrupted_acts: Dict[str, torch.Tensor] = {}

    def _make_corrupted_hook(name: str):
        def hook(value: torch.Tensor, hook):
            corrupted_acts[name] = value.detach()
            return value
        return hook

    handles = []
    for hname in receiver_hook_names:
        handles.append(model.hook_dict[hname].hook(
            _make_corrupted_hook(hname)
        ))

    with torch.no_grad():
        model(corrupted_data)

    for h in handles:
        h.remove()
    handles.clear()

    # --- Step 2: forward pass on clean data, keep grads ---------------------
    clean_acts: Dict[str, torch.Tensor] = {}
    grad_acts: Dict[str, torch.Tensor] = {}

    def _make_clean_hook(name: str):
        def hook(value: torch.Tensor, hook):
            # retain so we can call .backward() later
            value = value.requires_grad_(True)
            value.retain_grad()
            clean_acts[name] = value
            return value
        return hook

    for hname in receiver_hook_names:
        handles.append(model.hook_dict[hname].hook(
            _make_clean_hook(hname)
        ))

    with torch.enable_grad():
        out = model(clean_data)
        loss = metric(out)
        loss.backward()

    # Collect gradients
    for hname in receiver_hook_names:
        act = clean_acts.get(hname)
        if act is not None and act.grad is not None:
            grad_acts[hname] = act.grad.detach()

    for h in handles:
        h.remove()
    handles.clear()

    # --- Step 3: compute per-edge scores ------------------------------------
    scores: Dict[Tuple, float] = {}

    for child_name, child_subtree in corr.edges.items():
        grad = grad_acts.get(child_name)
        clean_act = clean_acts.get(child_name)
        corrupted_act = corrupted_acts.get(child_name)

        for child_index, parent_subtree in child_subtree.items():
            ci = child_index.as_index  # e.g. (slice(None), slice(None), 3)

            if grad is not None and clean_act is not None and corrupted_act is not None:
                # Δ = clean - corrupted, sliced to this node's index
                delta = (clean_act.detach()[ci] - corrupted_act[ci])
                g = grad[ci]
                # Sum over all non-index dimensions (batch, seq) → scalar
                score_val = float((g * delta).sum().item())
            else:
                score_val = 0.0

            for parent_name, parent_index_subtree in parent_subtree.items():
                for parent_index, edge in parent_index_subtree.items():
                    key = (child_name, child_index, parent_name, parent_index)
                    if edge.edge_type == EdgeType.PLACEHOLDER:
                        scores[key] = 0.0
                    else:
                        scores[key] = score_val

    return scores


def _apply_threshold(
    corr: TLACDCCorrespondence,
    scores: Dict[Tuple, float],
    threshold: float,
    abs_value: bool,
) -> None:
    """Set ``edge.present`` based on threshold and store score in ``edge.effect_size``."""
    for (child_name, child_index, parent_name, parent_index), score in scores.items():
        edge: Edge = corr.edges[child_name][child_index][parent_name][parent_index]
        edge.effect_size = score

        if edge.edge_type == EdgeType.PLACEHOLDER:
            edge.present = True
            continue

        compare = abs(score) if abs_value else score
        edge.present = compare >= threshold
