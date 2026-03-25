"""Run ACDC circuit discovery.

Usage:
    # Using a config file:
    python scripts/run_acdc.py --config configs/tasks/ioi.yaml --threshold 0.1

    # Fully manual:
    python scripts/run_acdc.py --task ioi --threshold 0.1 --model gpt2-small

    # With wandb logging:
    python scripts/run_acdc.py --task ioi --threshold 0.1 --wandb --wandb-project my-project
"""
from __future__ import annotations

import argparse
import datetime
import gc
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Make src/ importable when running as a script
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPTS_DIR.parent
for _p in [str(_ROOT / "src"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wandb
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import ct, reset_network
from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import get_all_induction_things
from acdc.ioi.utils import get_all_ioi_things
from acdc.logic_gates.utils import get_all_logic_gate_things

try:
    from acdc.tracr_task.utils import get_all_tracr_things
    _TRACR_AVAILABLE = True
except Exception:
    _TRACR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_LOADERS = {
    "ioi": lambda args: get_all_ioi_things(
        num_examples=args.num_examples,
        device=args.device,
        metric_name=args.metric,
        model_name=args.model,
    ),
    "docstring": lambda args: get_all_docstring_things(
        num_examples=args.num_examples,
        seq_len=args.seq_len,
        device=args.device,
        metric_name=args.metric,
        correct_incorrect_wandb=True,
        model_name=args.model,
    ),
    "induction": lambda args: get_all_induction_things(
        num_examples=args.num_examples,
        seq_len=args.seq_len,
        device=args.device,
        metric=args.metric,
        model_name=args.model,
    ),
    "greaterthan": lambda args: get_all_greaterthan_things(
        num_examples=args.num_examples,
        metric_name=args.metric,
        device=args.device,
        model_name=args.model,
    ),
    "or_gate": lambda args: get_all_logic_gate_things(
        mode="OR",
        num_examples=1,
        seq_len=1,
        device=args.device,
        model_name=args.model,
    ),
}

if _TRACR_AVAILABLE:
    TASK_LOADERS["tracr-reverse"] = lambda args: get_all_tracr_things(
        task="reverse", metric_name=args.metric, num_examples=6, device=args.device, model_name=args.model
    )
    TASK_LOADERS["tracr-proportion"] = lambda args: get_all_tracr_things(
        task="proportion", metric_name=args.metric, num_examples=50, device=args.device, model_name=args.model
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ACDC / PAHQ circuit discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config file (optional -- overridden by individual flags)
    p.add_argument("--config", type=str, default=None, help="Path to a YAML task config file.")

    # Task settings
    p.add_argument("--task", type=str, choices=list(TASK_LOADERS), help="Task to run.")
    p.add_argument("--threshold", type=float, required=True, help="ACDC edge-inclusion threshold.")
    p.add_argument("--num-examples", type=int, default=None, help="Number of examples (overrides config).")
    p.add_argument("--seq-len", type=int, default=None, help="Sequence length (overrides config).")
    p.add_argument("--metric", type=str, default="kl_div", choices=["kl_div", "nll", "match_nll"],
                   help="Evaluation metric.")
    p.add_argument("--model", type=str, default=None, help="Model name (e.g. gpt2-small, attn-only-4l).")

    # ACDC settings
    p.add_argument("--zero-ablation", action="store_true", help="Use zero ablation instead of corrupted.")
    p.add_argument("--abs-value-threshold", action="store_true", help="Threshold on |score| not score.")
    p.add_argument("--indices-mode", type=str, default="reverse", choices=["normal", "reverse", "shuffle"])
    p.add_argument("--names-mode", type=str, default="normal", choices=["normal", "reverse", "shuffle"])
    p.add_argument("--max-epochs", type=int, default=100_000, help="Maximum ACDC steps.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reset-network", action="store_true", help="Randomise weights before running.")
    p.add_argument("--first-cache-cpu", action="store_true", help="Keep online cache on CPU.")
    p.add_argument("--second-cache-cpu", action="store_true", help="Keep corrupted cache on CPU.")

    # Output settings
    p.add_argument("--out-dir", type=str, default=str(_ROOT / "results"),
                   help="Directory for saved edges and logs.")
    p.add_argument("--run-name", type=str, default=None, help="Human-readable run name.")

    # Wandb settings
    p.add_argument("--wandb", dest="use_wandb", action="store_true", help="Enable wandb logging.")
    p.add_argument("--wandb-project", type=str, default="acdc")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="default")
    p.add_argument("--wandb-mode", type=str, default="online")

    return p


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load YAML config and fill in missing args (flags take precedence)."""
    if args.config is None:
        return args

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    defaults = {
        "task": cfg.get("task"),
        "num_examples": cfg.get("num_examples", 50),
        "seq_len": cfg.get("seq_len", 300),
        "metric": cfg.get("metric", "kl_div"),
        "model": cfg.get("model", "attn-only-4l"),
    }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = merge_config(args)

    if args.task is None:
        parser.error("--task is required (or specify via --config)")
    if args.num_examples is None:
        args.num_examples = 50
    if args.seq_len is None:
        args.seq_len = 300
    if args.model is None:
        args.model = "attn-only-4l"

    torch.manual_seed(args.seed)
    torch.autograd.set_grad_enabled(False)

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{ct()}_{args.task}_{args.threshold}"
    if args.zero_ablation:
        run_name += "_zero"

    # Load task
    things = TASK_LOADERS[args.task](args)
    tl_model = things.tl_model.to(args.device)

    if args.reset_network and args.task != "or_gate":
        reset_network(args.task, args.device, tl_model)

    tl_model.reset_hooks()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_pos_embed = args.task.startswith("tracr")
    test_metric = getattr(things, "test_metrics", {}).get("frac_correct") or \
                  getattr(things, "test_metrics", {}).get("accuracy") or \
                  getattr(things, "test_metrics", {}).get("docstring_stefan") or \
                  (list(things.test_metrics.values())[0] if hasattr(things, "test_metrics") and things.test_metrics else None)

    # Build experiment
    exp = TLACDCExperiment(
        model=tl_model,
        threshold=args.threshold,
        using_wandb=args.use_wandb,
        wandb_entity_name=args.wandb_entity or "",
        wandb_project_name=args.wandb_project,
        wandb_run_name=run_name,
        wandb_group_name=args.wandb_group,
        wandb_notes="",
        wandb_dir=str(out_dir / "wandb"),
        wandb_mode=args.wandb_mode,
        wandb_config=args,
        zero_ablation=args.zero_ablation,
        abs_value_threshold=args.abs_value_threshold,
        ds=things.validation_data,
        ref_ds=things.validation_patch_data,
        metric=things.validation_metric,
        second_metric=None,
        verbose=True,
        indices_mode=args.indices_mode,
        names_mode=args.names_mode,
        corrupted_cache_cpu=args.second_cache_cpu,
        hook_verbose=False,
        online_cache_cpu=args.first_cache_cpu,
        add_sender_hooks=True,
        use_pos_embed=use_pos_embed,
        add_receiver_hooks=False,
        remove_redundant=False,
        show_full_index=use_pos_embed,
        test_metric=test_metric,
    )

    # Run ACDC
    for _ in range(args.max_epochs):
        exp.step(testing=False)
        if exp.current_node is None:
            break

    # Save results
    edges_path = out_dir / f"{run_name}_edges.pkl"
    exp.save_edges(str(edges_path))
    print(f"Saved edges to {edges_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
