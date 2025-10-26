#!/bin/bash

THRESHOLDS=(0.001 0.005 0.01) #for test
TASKS=('docstring' 'greaterthan' 'ioi')
MODELS=('attn-only-4l' 'redwood_attn_2l' 'gpt2')

WANDB_PROJECT_NAME="PAHQ"
WANDB_DIR="./wandb"
WANDB_MODE="online"

DEVICE="cuda"

RESET_NETWORK=0

METRICS=('kl_div' 'metric')

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

RUN_COUNT=0
SUCCESS_COUNT=0
FAILED_COUNT=0

for METRIC in "${METRICS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for THRESHOLD in "${THRESHOLDS[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                echo "Running task: $TASK with model: $MODEL, threshold: $THRESHOLD, metric: $METRIC"
                
                LOG_FILE="${LOG_DIR}/${TASK}_${MODEL}_${THRESHOLD}_${METRIC}.log"

                python acdc/main.py \
                    --task "$TASK" \
                    --threshold "$THRESHOLD" \
                    --using-wandb \
                    --wandb-project-name "$WANDB_PROJECT_NAME" \
                    --wandb-dir "$WANDB_DIR" \
                    --wandb-mode "$WANDB_MODE" \
                    --device "$DEVICE" \
                    --reset-network "$RESET_NETWORK" \
                    --metric "$METRIC" \
                    --model_name "$MODEL" > "$LOG_FILE" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "✅ Success: $TASK with $MODEL, threshold: $THRESHOLD, metric: $METRIC"
                    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                else
                    echo "❌ Failed: $TASK with $MODEL, threshold: $THRESHOLD, metric: $METRIC"
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi
                
                RUN_COUNT=$((RUN_COUNT + 1))
            done
        done
    done
done

echo "All tasks completed. Total runs: $RUN_COUNT"

echo "System will shut down in 10 seconds..."
sleep 10
shutdown -h now