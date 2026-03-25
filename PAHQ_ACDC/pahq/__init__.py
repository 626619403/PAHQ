"""PAHQ-specific additions: baselines, metrics, and experiment utilities."""

from pahq.rtn_quantizer import RTNQuantizer, weight_split_by_head, mixed_precision_matmul
from pahq.faithfulness import FaithfulnessEvaluator, faithfulness
from pahq.roc_local import RocEvaluator, default_thresholds
from pahq.eap import run_eap, compute_eap_scores

__all__ = [
    # RTN-Q quantizer
    "RTNQuantizer",
    "weight_split_by_head",
    "mixed_precision_matmul",
    # Faithfulness metric
    "FaithfulnessEvaluator",
    "faithfulness",
    # Local AUC-ROC
    "RocEvaluator",
    "default_thresholds",
    # EAP baseline
    "run_eap",
    "compute_eap_scores",
]
