from experiments.launcher import KubernetesJob, launch
import subprocess
import argparse
import numpy as np
import torch

default_device = "cuda" if torch.cuda.is_available() else "cpu"
CPU = 0

def main(
    testing: bool,
    is_adria: bool,
):
    thresholds = 10 ** np.linspace(-2, 0.5, 21)
    seed = 424671755

    commands: list[list[str]] = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for loss_type in ["kl_div"]:
                for threshold in [1.0] if testing else thresholds:
                    command = [
                        "python",
                        "acdc/main.py" if (not is_adria) else "/Automatic-Circuit-Discovery/acdc/main.py",
                        "--task=induction",
                        f"--threshold={threshold:.5f}",
                        "--using-wandb",
                        "--wandb-group-name=adria-induction-3",
                        f"--device={default_device}",
                        f"--reset-network={reset_network}",
                        f"--seed={seed}",
                        f"--metric={loss_type}",
                        f"--torch-num-threads={CPU}",
                        "--wandb-dir=/autodl-tmp/acdc",
                        f"--wandb-mode={'offline' if testing else 'online'}",
                    ]
                    if zero_ablation:
                        command.append("--zero-ablation")

                    commands.append(command)

    if is_adria:
        launch(
            commands,
            name="acdc-induction",
            job=None
            if testing
            else KubernetesJob(container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.2.8", cpu=CPU, gpu=0),
        )

    else:
        for command in commands:
            print("Running", command)
            subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--is-adria", action="store_true")
    main(
        testing=parser.parse_args().testing,
        is_adria=parser.parse_args().is_adria,
    )
