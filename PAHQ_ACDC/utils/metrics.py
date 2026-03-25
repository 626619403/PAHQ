"""Shared metric utilities for PAHQ circuit discovery experiments.

These are thin wrappers / re-exports of the metrics implemented in
``acdc.acdc_utils``, plus a few helpers used across scripts.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F


def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
    base_model_probs_last_seq_element_only: bool = False,
    return_one_element: bool = True,
) -> torch.Tensor:
    """KL divergence between model logits and base model log-probabilities.

    Parameters
    ----------
    logits : Tensor [batch, seq, vocab]
    base_model_logprobs : Tensor [batch, seq, vocab] or [batch, vocab]
        Log-probabilities of the reference (full) model.
    mask_repeat_candidates : optional Tensor [batch, seq]
        Mask for induction task -- only compare at repeated positions.
    last_seq_element_only : bool
        Slice to last token before computing KL.
    return_one_element : bool
        Return scalar mean (True) or per-example tensor (False).
    """
    if last_seq_element_only:
        logits = logits[:, -1, :]
    if base_model_probs_last_seq_element_only:
        base_model_logprobs = base_model_logprobs[:, -1, :]

    logprobs = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

    if mask_repeat_candidates is not None:
        answer = kl[mask_repeat_candidates]
    elif not last_seq_element_only:
        answer = kl.view(-1)
    else:
        answer = kl

    return answer.mean() if return_one_element else answer


def negative_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    baseline: Union[float, torch.Tensor] = 0.0,
    last_seq_element_only: bool = True,
    return_one_element: bool = True,
) -> torch.Tensor:
    """Negative log-probability of ground-truth labels.

    Parameters
    ----------
    logits : Tensor [batch, seq, vocab]
    labels : Tensor [batch, vocab] -- one-hot or smooth, used as log-prob targets.
    """
    logprobs = F.log_softmax(logits, dim=-1)
    if last_seq_element_only:
        logprobs = logprobs[:, -1, :]

    nlls = -(logprobs * labels).sum(dim=-1) - baseline

    if mask_repeat_candidates is not None:
        answer = nlls[mask_repeat_candidates]
    elif not last_seq_element_only:
        answer = nlls.view(-1)
    else:
        answer = nlls

    return answer.mean() if return_one_element else answer
