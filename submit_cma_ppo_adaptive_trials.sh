#!/usr/bin/env bash
# Submit 5 CMA-PPO independent trial jobs with adaptive history scheduling

BASE_DIR="/home/jchao1/CS551-F25-jchao1/Project4/CS551_P4"
cd "$BASE_DIR"

# Create output directory
OUTPUT_DIR="${BASE_DIR}/cma_ppo_adaptive_trials_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Submitting 5 CMA-PPO Adaptive History Trial Jobs"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# CMA-PPO Config with adaptive history:
# - 4096 steps × 937 iter = 3,837,952 env steps
# - Adaptive history: H_max=5, H_min=1
# - Reward goal: 250, Reward high: 300 (goal + 50)

# Submit CMA-PPO trials (5 jobs)
echo "Submitting CMA-PPO trials with adaptive history..."
for trial in 1 2 3 4 5; do
    TRIAL_DIR="${OUTPUT_DIR}/cma_ppo_trial_${trial}"
    mkdir -p "$TRIAL_DIR"
    
    JOB_ID=$(sbatch --parsable \
        --job-name=cma_ppo_adaptive_t${trial} \
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
            --history_size 5 \
            --history_len_min 1 \
            --reward_high 300.0 \
            --goal_delta 50.0 \
            --goal_window 3 \
            --goal_min_consecutive 2 \
            --seed $((1234 + trial)) \
            --directory "$TRIAL_DIR")
    
    echo "  CMA-PPO Trial $trial: Job ID $JOB_ID → $TRIAL_DIR"
done

echo ""
echo "=========================================="
echo "All 5 jobs submitted!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - History size (H_max): 5"
echo "  - History min (H_min): 1"
echo "  - Reward goal: 250 (from get_goal())"
echo "  - Reward high: 300 (goal + 50)"
echo "  - Adaptive history: Enabled"
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
echo "To check for adaptive history adjustments:"
echo "  grep -r '\[CMA-PPO\] Adjusted history length' ${OUTPUT_DIR}/"
echo ""

