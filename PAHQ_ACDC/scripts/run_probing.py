"""Run Subnetwork Probing (SP) circuit discovery.

Subnetwork probing learns which edges to keep by training binary edge masks
with L1 regularisation on the number of active edges.

Usage:
    python scripts/run_probing.py --task ioi --device cuda --lambda-reg 10
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

from acdc.ioi.utils import get_all_ioi_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import get_all_induction_things
from acdc.docstring.utils import get_all_docstring_things

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Subnetwork Probing circuit discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", type=str, required=True, choices=list(TASK_LOADERS))
    p.add_argument("--lambda-reg", type=float, default=10.0, help="L1 regularisation coefficient.")
    p.add_argument("--n-epochs", type=int, default=2000, help="Training epochs.")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    p.add_argument("--num-examples", type=int, default=50)
    p.add_argument("--metric", type=str, default="kl_div")
    p.add_argument("--model", type=str, default="attn-only-4l")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="subnetwork-probing")
    p.add_argument("--out-dir", type=str, default=str(_ROOT / "results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.task}...")
    things = TASK_LOADERS[args.task](args)
    model = things.tl_model.to(args.device)

    from probing.train import train_sp
    train_sp(
        model=model,
        things=things,
        lambda_reg=args.lambda_reg,
        n_epochs=args.n_epochs,
        lr=args.lr,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        out_dir=Path(args.out_dir),
        task=args.task,
    )


if __name__ == "__main__":
    main()
