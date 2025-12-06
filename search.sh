#!/usr/bin/env bash
# Unified hyperparameter search for PPO and ESPPO
# Usage: bash search.sh
# This will submit jobs to find optimal hyperparameters for both algorithms

echo "Starting hyperparameter search for PPO and ESPPO..."
echo "This will submit many jobs - monitor with: squeue -u \$USER"
echo ""

# ============================================
# PPO Hyperparameter Search
# ============================================
echo "=== Submitting PPO hyperparameter search jobs ==="

# PPO parameters to search
for ppo_lr in {0.0001,0.00025,0.001}; do
    for max_steps in {128,256,512}; do
        for n_updates in {3,5}; do
            for batch_size in {32,64}; do
                for gamma in {0.99}; do  # Keep gamma fixed at 0.99
                    for clip in {0.01,0.02}; do
                        for ent_coeff in {0.0,0.001}; do
                            sbatch runner.sh python main.py -a PPO \
                                --ppo_lr=$ppo_lr \
                                --max_steps=$max_steps \
                                --n_updates=$n_updates \
                                --batch_size=$batch_size \
                                --gamma=$gamma \
                                --clip=$clip \
                                --ent_coeff=$ent_coeff \
                                --n_trials=3 \
                                --epoch=10000
                        done
                    done
                done
            done
        done
    done
done

echo "PPO search: Submitted $(echo "3*3*2*2*1*2*2" | bc) jobs"
echo ""

# ============================================
# ESPPO Hyperparameter Search
# ============================================
echo "=== Submitting ESPPO hyperparameter search jobs ==="

# ESPPO parameters to search
# Shared PPO params + ES-specific params
for ppo_lr in {0.0001,0.00025,0.001}; do
    for max_steps in {128,256}; do  # Reduced for ESPPO (more expensive)
        for n_updates in {3,5}; do
            for batch_size in {32,64}; do
                for gamma in {0.99}; do  # Keep gamma fixed
                    for clip in {0.01,0.02}; do
                        for ent_coeff in {0.0,0.001}; do
                            # ES-specific parameters
                            for population_size in {5,10}; do
                                for sigma in {0.1,0.2}; do
                                    for es_lr in {0.001,0.01}; do
                                        for n_seq in {1}; do  # Keep n_seq fixed at 1 for baseline
                                            sbatch runner.sh python main.py -a ESPPO \
                                                --population_size=$population_size \
                                                --sigma=$sigma \
                                                --n_seq=$n_seq \
                                                --ppo_lr=$ppo_lr \
                                                --es_lr=$es_lr \
                                                --max_steps=$max_steps \
                                                --n_updates=$n_updates \
                                                --batch_size=$batch_size \
                                                --gamma=$gamma \
                                                --clip=$clip \
                                                --ent_coeff=$ent_coeff \
                                                --n_trials=3 \
                                                --epoch=10000
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
done
        done
    done
done

echo "ESPPO search: Submitted $(echo "3*2*2*2*1*2*2*2*2*2*1" | bc) jobs"
echo ""
echo "=== Search complete ==="
echo "Total jobs submitted: $(echo "3*3*2*2*1*2*2 + 3*2*2*2*1*2*2*2*2*2*1" | bc)"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Results will be saved in ./checkpoints/ with algorithm-specific directories"
echo ""
echo "After search completes, analyze results to find best hyperparameters,"
echo "then run those as baselines for comparison."
