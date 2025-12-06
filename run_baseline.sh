#!/usr/bin/env bash
# Baseline configuration from paper
# Runs PPO and ES-PPO with exact hyperparameters from the paper

echo "=== Running Baseline Experiments (Paper Configuration) ==="
echo ""

# ============================================
# PPO Baseline (from paper section 4.1.1)
# ============================================
echo "Submitting PPO baseline..."
echo "Config: lr=0.0001, max_steps=256, batch_size=32, n_updates=3, clip=0.2, ent_coeff=0.0, gamma=0.99"
sbatch runner.sh python main.py -a PPO \
    --ppo_lr 0.0001 \
    --max_steps 256 \
    --batch_size 32 \
    --n_updates 3 \
    --gamma 0.99 \
    --clip 0.2 \
    --ent_coeff 0.0 \
    --n_trials 5 \
    --epoch 10000

echo ""

# ============================================
# ES-PPO Baseline (from paper section 4.1.3)
# ============================================
echo "Submitting ES-PPO baseline..."
echo "Config: population_size=5, sigma=0.1, es_lr=0.001, ppo_lr=0.0001, max_steps=256, batch_size=32, n_updates=1, clip=0.2, ent_coeff=0.0, gamma=0.99, n_seq=1"
sbatch runner.sh python main.py -a ESPPO \
    --population_size 5 \
    --sigma 0.1 \
    --es_lr 0.001 \
    --ppo_lr 0.0001 \
    --max_steps 256 \
    --batch_size 32 \
    --n_updates 1 \
    --gamma 0.99 \
    --clip 0.2 \
    --ent_coeff 0.0 \
    --n_seq 1 \
    --n_trials 5 \
    --epoch 10000

echo ""
echo "=== Baseline jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Results will be saved in ./checkpoints/"

