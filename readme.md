# PAHQ: Accelerating Automated Circuit Discovery through Mixed-Precision Inference Optimization

This repository contains the implementation code for the paper "PAHQ: Accelerating Automated Circuit Discovery through Mixed-Precision Inference Optimization". Our implementation is based on modifications to the original ACDC codebase and the transformer_lens library, accelerating automated circuit discovery through mixed-precision inference optimization.

## System Requirements

- Python 3.12
- GPU: H20-NVLink (96GB)
- CPU: 16 vCPU AMD EPYC 9K84 96-Core Processor

## Installation Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Replace library files:
   Replace the original transformer_lens library and cmapy.py in your Python environment with the modified versions provided in this directory.

3. Navigate to the PAHQ_ACDC directory:
   ```bash
   cd PAHQ_ACDC
   ```



## Running Experiments

We provide a script to run all experimental combinations. Execute it with:

```bash
bash run_experiments.sh
```

This script automatically runs combinations of different tasks, models, thresholds, and evaluation metrics, logging results to output files.

### Configuration Parameters

The script includes the following configurable parameters:
- `THRESHOLDS`: Different threshold settings (0.001, 0.005, 0.01)
- `TASKS`: Task types ('docstring', 'greaterthan', 'ioi')
- `MODELS`: Model types ('attn-only-4l', 'redwood_attn_2l', 'gpt2')
- `METRICS`: Evaluation metrics ('kl_div', 'metric')

### Running Individual Experiments

To run a specific experiment configuration, use:

```bash
python acdc/main.py \
    --task <TASK> \
    --threshold <THRESHOLD> \
    --using-wandb \
    --wandb-project-name "PAHQ" \
    --wandb-dir "./wandb" \
    --wandb-mode "online" \
    --device "cuda" \
    --reset-network 0 \
    --metric <METRIC> \
    --model_name <MODEL>
```

## Experimental Results

Experimental results are stored in:
- Log files: `./logs/<TASK>_<MODEL>_<THRESHOLD>_<METRIC>.log`
- Weights & Biases: Online project "PAHQ"

## License

This project is licensed under the MIT License. See the LICENSE file for details.
