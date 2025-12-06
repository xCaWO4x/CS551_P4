#!/usr/bin/env bash
# Test script to run PPO and ESPPO with default parameters
# This verifies everything works before doing larger searches

echo "=== Testing PPO with defaults ==="
echo "Defaults: --ppo_lr 0.0001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 5 --epoch 10000"
echo "Submitting PPO test job..."
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 5 --epoch 10000

echo ""
echo "=== Testing ESPPO with defaults ==="
echo "Defaults: --population_size 5 --sigma 0.1 --n_seq 1 --ppo_lr 0.0001 --es_lr 0.001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 5 --epoch 10000"
echo "Submitting ESPPO test job..."
sbatch runner.sh python main.py -a ESPPO --population_size 5 --sigma 0.1 --n_seq 1 --ppo_lr 0.0001 --es_lr 0.001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 5 --epoch 10000

echo ""
echo "Jobs submitted! Check status with: squeue -u \$USER"
echo "View output with: tail -f slurm-<jobid>.out"

