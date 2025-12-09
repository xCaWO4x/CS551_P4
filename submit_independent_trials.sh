#!/usr/bin/env bash
# Submit 6 independent trial jobs (3 ES-PPO + 3 CMA-PPO)

BASE_DIR="/home/jchao1/CS551-F25-jchao1/Project4/CS551_P4"
cd "$BASE_DIR"

# Create output directory
OUTPUT_DIR="${BASE_DIR}/independent_trials_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Submitting 6 Independent Trial Jobs"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# ES-PPO Config: 256 steps × 5 pop × 3000 iter = 3,840,000 env steps
# CMA-PPO Config: 4096 steps × 937 iter = 3,837,952 env steps (0.05% difference)

# Submit ES-PPO trials (3 jobs)
echo "Submitting ES-PPO trials..."
for trial in 1 2 3; do
    TRIAL_DIR="${OUTPUT_DIR}/es_ppo_trial_${trial}"
    mkdir -p "$TRIAL_DIR"
    
    JOB_ID=$(sbatch --parsable \
        --job-name=es_ppo_t${trial} \
        --output="${TRIAL_DIR}/slurm-%j.out" \
        --error="${TRIAL_DIR}/slurm-%j.err" \
        runner.sh \
        python main.py \
            --alg ESPPO \
            --epoch 3000 \
            --n_trials 1 \
            --population_size 5 \
            --max_steps 256 \
            --batch_size 32 \
            --n_updates 1 \
            --sigma 0.1 \
            --es_lr 0.001 \
            --ppo_lr 0.0001 \
            --seed $((1234 + trial)) \
            --directory "$TRIAL_DIR")
    
    echo "  ES-PPO Trial $trial: Job ID $JOB_ID → $TRIAL_DIR"
done

echo ""

# Submit CMA-PPO trials (3 jobs)
echo "Submitting CMA-PPO trials..."
for trial in 1 2 3; do
    TRIAL_DIR="${OUTPUT_DIR}/cma_ppo_trial_${trial}"
    mkdir -p "$TRIAL_DIR"
    
    JOB_ID=$(sbatch --parsable \
        --job-name=cma_ppo_t${trial} \
        --output="${TRIAL_DIR}/slurm-%j.out" \
        --error="${TRIAL_DIR}/slurm-%j.err" \
        runner.sh \
        python main.py \
            --alg CMA_PPO \
            --epoch 937 \
            --n_trials 1 \
            --max_steps 4096 \
            --batch_size 1024 \
            --n_updates 78 \
            --seed $((1234 + trial)) \
            --directory "$TRIAL_DIR")
    
    echo "  CMA-PPO Trial $trial: Job ID $JOB_ID → $TRIAL_DIR"
done

echo ""
echo "=========================================="
echo "All 6 jobs submitted!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Checking job status..."
sleep 2
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
echo ""
echo "Jobs will run concurrently if resources are available."
echo "If some show 'PENDING', they'll start as resources free."
echo ""
echo "To monitor all jobs:"
echo "  watch -n 30 'squeue -u $USER'"
echo ""

