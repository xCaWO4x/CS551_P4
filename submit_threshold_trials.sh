#!/bin/bash
# Submit 5 independent trials for both CMA-PPO and ES-PPO
# Policy will freeze immediately when threshold (250) is reached

set -e

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="independent_trials_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Submitting Threshold Comparison Trials"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Threshold: 250 (policy freezes immediately when hit)"
echo "Trials per algorithm: 5"
echo ""

# ES-PPO Configuration (same as before)
ES_EPOCH=3000
ES_MAX_STEPS=256
ES_BATCH_SIZE=32
ES_N_UPDATES=1
ES_POP_SIZE=5
ES_SIGMA=0.1
ES_ES_LR=0.001
ES_PPO_LR=0.0001

# CMA-PPO Configuration (same as before)
CMA_EPOCH=937
CMA_MAX_STEPS=4096
CMA_BATCH_SIZE=1024
CMA_N_UPDATES=78

echo "ES-PPO Configuration:"
echo "  Epochs: $ES_EPOCH"
echo "  Max steps: $ES_MAX_STEPS"
echo "  Batch size: $ES_BATCH_SIZE"
echo "  Population size: $ES_POP_SIZE"
echo ""
echo "CMA-PPO Configuration:"
echo "  Epochs: $CMA_EPOCH"
echo "  Max steps: $CMA_MAX_STEPS"
echo "  Batch size: $CMA_BATCH_SIZE"
echo ""

# Submit ES-PPO trials
echo "Submitting ES-PPO trials..."
for trial in 1 2 3 4 5; do
    TRIAL_DIR="${OUTPUT_DIR}/es_ppo_trial_${trial}"
    mkdir -p "$TRIAL_DIR"
    
    JOB_ID=$(sbatch --parsable \
        --job-name=es_ppo_t${trial} \
        --output="${TRIAL_DIR}/slurm-%j.out" \
        --error="${TRIAL_DIR}/slurm-%j.err" \
        runner.sh \
        python main.py \
            --alg ESPPO \
            --epoch $ES_EPOCH \
            --n_trials 1 \
            --population_size $ES_POP_SIZE \
            --max_steps $ES_MAX_STEPS \
            --batch_size $ES_BATCH_SIZE \
            --n_updates $ES_N_UPDATES \
            --sigma $ES_SIGMA \
            --es_lr $ES_ES_LR \
            --ppo_lr $ES_PPO_LR \
            --seed $((1234 + trial)) \
            --directory "$TRIAL_DIR")
    
    echo "  ES-PPO Trial $trial: Job ID $JOB_ID → $TRIAL_DIR"
done

echo ""

# Submit CMA-PPO trials
echo "Submitting CMA-PPO trials..."
for trial in 1 2 3 4 5; do
    TRIAL_DIR="${OUTPUT_DIR}/cma_ppo_trial_${trial}"
    mkdir -p "$TRIAL_DIR"
    
    JOB_ID=$(sbatch --parsable \
        --job-name=cma_ppo_t${trial} \
        --output="${TRIAL_DIR}/slurm-%j.out" \
        --error="${TRIAL_DIR}/slurm-%j.err" \
        runner.sh \
        python main.py \
            --alg CMA_PPO \
            --epoch $CMA_EPOCH \
            --n_trials 1 \
            --max_steps $CMA_MAX_STEPS \
            --batch_size $CMA_BATCH_SIZE \
            --n_updates $CMA_N_UPDATES \
            --seed $((1234 + trial)) \
            --directory "$TRIAL_DIR")
    
    echo "  CMA-PPO Trial $trial: Job ID $JOB_ID → $TRIAL_DIR"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Total jobs: 10 (5 ES-PPO + 5 CMA-PPO)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Check status with: squeue -u \$USER"
echo "Check progress with: bash check_progress.sh"
echo "Plot results with: python3 plot_trials.py --directory $OUTPUT_DIR"
echo ""

