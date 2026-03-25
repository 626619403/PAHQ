"""Compute AUC-ROC curves locally (no wandb dependency).

Loads saved edge files from results/ and computes ROC vs. the ground-truth
circuit for a given task. Outputs a JSON file with per-threshold TPR/FPR
values and the overall AUC.

Usage:
    python scripts/run_roc.py --task ioi --edges-dir results/ --out results/roc_ioi.json
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
for _p in [str(_ROOT / "src"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.ioi.utils import get_all_ioi_things, get_ioi_true_edges
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
from acdc.induction.utils import get_all_induction_things
from acdc.docstring.utils import get_all_docstring_things, get_docstring_subgraph_true_edges
from pahq.roc import RocEvaluator, default_thresholds

TASK_LOADERS = {
    "ioi": lambda a: get_all_ioi_things(num_examples=a.num_examples, device=a.device,
                                         metric_name=a.metric, model_name=a.model),
    "docstring": lambda a: get_all_docstring_things(num_examples=a.num_examples, seq_len=41,
                                                     device=a.device, metric_name=a.metric,
                                                     correct_incorrect_wandb=False, model_name=a.model),
    "induction": lambda a: get_all_induction_things(num_examples=a.num_examples, seq_len=300,
                                                     device=a.device, metric=a.metric, model_name=a.model),
    "greaterthan": lambda a: get_all_greaterthan_things(num_examples=a.num_examples, metric_name=a.metric,
                                                         device=a.device, model_name=a.model),
}

TRUE_EDGES_FNS = {
    "ioi": get_ioi_true_edges,
    "greaterthan": get_greaterthan_true_edges,
    "docstring": get_docstring_subgraph_true_edges,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local AUC-ROC computation from saved edges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", type=str, required=True, choices=list(TASK_LOADERS))
    p.add_argument("--edges-dir", type=str, default=str(_ROOT / "results"),
                   help="Directory containing *_edges.pkl files.")
    p.add_argument("--out", type=str, default=None, help="Output JSON path.")
    p.add_argument("--num-examples", type=int, default=50)
    p.add_argument("--metric", type=str, default="kl_div")
    p.add_argument("--model", type=str, default="attn-only-4l")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--plot", action="store_true", help="Show matplotlib ROC plot.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.autograd.set_grad_enabled(False)

    print(f"Loading {args.task} task...")
    things = TASK_LOADERS[args.task](args)
    model = things.tl_model.to(args.device)

    exp = TLACDCExperiment(
        model=model,
        threshold=0.1,
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

    # Build ground-truth circuit
    gt_fn = TRUE_EDGES_FNS.get(args.task)
    if gt_fn is None:
        raise NotImplementedError(f"No ground-truth edges known for task '{args.task}'")

    # Load and score edge files
    edges_dir = Path(args.edges_dir)
    edge_files = sorted(edges_dir.glob(f"*{args.task}*_edges.pkl"))
    if not edge_files:
        print(f"No edge files found in {edges_dir} for task {args.task}")
        return

    corrs_and_scores = []
    for ef in edge_files:
        corr = deepcopy(exp.corr)
        try:
            exp.load_subgraph(str(ef))
        except Exception as e:
            print(f"  Skipping {ef.name}: {e}")
            continue
        # Extract threshold from filename (heuristic)
        try:
            thr = float(ef.stem.split("_")[-2])
        except (ValueError, IndexError):
            thr = 0.0
        corrs_and_scores.append((deepcopy(exp.corr), {"score": thr}))
        for e in exp.corr.all_edges().values():
            e.present = True  # reset

    print(f"Loaded {len(corrs_and_scores)} edge files.")

    # Ground truth correspondence
    gt_corr = deepcopy(exp.corr)
    for e in gt_corr.all_edges().values():
        e.present = False
    true_edges = gt_fn(exp=exp)
    exp.load_subgraph(true_edges)
    gt_corr = deepcopy(exp.corr)

    full_corr = deepcopy(exp.corr)
    for e in full_corr.all_edges().values():
        e.present = True

    evaluator = RocEvaluator(ground_truth=gt_corr, full_corr=full_corr)
    points = evaluator.sweep(corrs_and_scores)
    auc = RocEvaluator.auc(points)

    print(f"AUC-ROC ({args.task}): {auc:.4f}")

    # Output
    out_path = Path(args.out) if args.out else edges_dir / f"roc_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({"task": args.task, "auc": auc, "points": points}, f, indent=2)
    print(f"Saved ROC data to {out_path}")

    if args.plot:
        RocEvaluator.plot(points, label=args.task, show=True)


if __name__ == "__main__":
    main()
