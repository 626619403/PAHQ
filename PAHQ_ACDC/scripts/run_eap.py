"""Run Edge Attribution Patching (EAP) baseline circuit discovery.

EAP computes per-edge attribution scores in a single forward+backward pass
(O(1) cost vs ACDC's O(|E|)), making it a fast approximate alternative.

Usage:
    python scripts/run_eap.py --task ioi --threshold 0.05 --device cuda
    python scripts/run_eap.py --config configs/tasks/ioi.yaml --threshold 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
for _p in [str(_ROOT / "src"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import ct
from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import get_all_induction_things
from acdc.ioi.utils import get_all_ioi_things
from pahq.eap import compute_eap_scores, run_eap
from pahq.roc import RocEvaluator, default_thresholds


TASK_LOADERS = {
    "ioi": lambda a: get_all_ioi_things(num_examples=a.num_examples, device=a.device,
                                         metric_name=a.metric, model_name=a.model),
    "docstring": lambda a: get_all_docstring_things(num_examples=a.num_examples, seq_len=a.seq_len,
                                                     device=a.device, metric_name=a.metric,
                                                     correct_incorrect_wandb=False, model_name=a.model),
    "induction": lambda a: get_all_induction_things(num_examples=a.num_examples, seq_len=a.seq_len,
                                                     device=a.device, metric=a.metric, model_name=a.model),
    "greaterthan": lambda a: get_all_greaterthan_things(num_examples=a.num_examples, metric_name=a.metric,
                                                         device=a.device, model_name=a.model),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EAP circuit discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--task", type=str, choices=list(TASK_LOADERS))
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Attribution threshold for edge inclusion (0 = keep all).")
    p.add_argument("--num-examples", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--metric", type=str, default="kl_div")
    p.add_argument("--model", type=str, default="attn-only-4l")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compute-roc", action="store_true",
                   help="Sweep thresholds and compute AUC-ROC (requires ground-truth circuit).")
    p.add_argument("--out-dir", type=str, default=str(_ROOT / "results"))
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key in ["task", "num_examples", "seq_len", "metric", "model"]:
            if getattr(args, key, None) is None:
                setattr(args, key, cfg.get(key))

    if args.task is None:
        raise ValueError("--task is required")

    torch.manual_seed(args.seed)
    torch.autograd.set_grad_enabled(True)  # EAP requires gradients

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    things = TASK_LOADERS[args.task](args)
    model = things.tl_model.to(args.device)

    # Build full correspondence (template)
    exp = TLACDCExperiment(
        model=model,
        threshold=args.threshold,
        ds=things.validation_data,
        ref_ds=things.validation_patch_data,
        metric=things.validation_metric,
        test_metric=list(things.test_metrics.values())[0] if things.test_metrics else things.validation_metric,
        add_sender_hooks=True,
        add_receiver_hooks=False,
        verbose=False,
        using_wandb=False,
        wandb_entity_name="",
        wandb_project_name="",
        wandb_run_name="",
        wandb_group_name="",
    )

    print(f"Computing EAP scores for {args.task}...")
    scores = compute_eap_scores(
        model=model,
        corr=exp.corr,
        clean_data=things.validation_data,
        corrupted_data=things.validation_patch_data,
        metric=things.validation_metric,
    )

    # Apply threshold
    run_eap(model, exp.corr, things.validation_data, things.validation_patch_data,
            things.validation_metric, threshold=args.threshold)

    n_edges = exp.corr.count_no_edges()
    print(f"Edges after threshold={args.threshold}: {n_edges}")

    # Save
    run_name = f"eap_{args.task}_{args.threshold}"
    exp.save_edges(str(out_dir / f"{run_name}_edges.pkl"))
    print(f"Saved to {out_dir / f'{run_name}_edges.pkl'}")


if __name__ == "__main__":
    main()
