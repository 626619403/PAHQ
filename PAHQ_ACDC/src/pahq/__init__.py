"""PAHQ-specific additions: baselines, metrics, and experiment utilities."""

from pahq.quantizer import RTNQuantizer, weight_split_by_head, mixed_precision_matmul
from pahq.faithfulness import FaithfulnessEvaluator, faithfulness
from pahq.roc import RocEvaluator, default_thresholds
from pahq.eap import run_eap, compute_eap_scores

__all__ = [
    "RTNQuantizer",
    "weight_split_by_head",
    "mixed_precision_matmul",
    "FaithfulnessEvaluator",
    "faithfulness",
    "RocEvaluator",
    "default_thresholds",
    "run_eap",
    "compute_eap_scores",
]
