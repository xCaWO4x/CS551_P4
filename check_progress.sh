#!/usr/bin/env bash
# Deterministic progress checker for independent trial runs

BASE_DIR="/home/jchao1/CS551-F25-jchao1/Project4/CS551_P4"
cd "$BASE_DIR"

# Find the most recent independent_trials directory
LATEST_DIR=$(find . -maxdepth 1 -type d -name "independent_trials_*" | sort -r | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No independent_trials directory found."
    echo "Looking for individual trial directories..."
    
    # Check for individual trial directories
    ES_DIRS=($(find . -maxdepth 1 -type d -name "es_ppo_trial_*" 2>/dev/null | sort))
    CMA_DIRS=($(find . -maxdepth 1 -type d -name "cma_ppo_trial_*" 2>/dev/null | sort))
    
    if [ ${#ES_DIRS[@]} -eq 0 ] && [ ${#CMA_DIRS[@]} -eq 0 ]; then
        echo "No trial directories found. Jobs may not have been submitted yet."
        exit 1
    fi
    
    OUTPUT_DIR="."
else
    OUTPUT_DIR="$LATEST_DIR"
    echo "Using directory: $OUTPUT_DIR"
fi

echo ""
echo "=========================================="
echo "PROGRESS CHECK - Independent Trials"
echo "=========================================="
echo ""

python3 << 'PYTHON_SCRIPT'
import os
import re
import glob
from pathlib import Path

base_dir = os.environ.get('BASE_DIR', '.')
output_dir = os.environ.get('OUTPUT_DIR', '.')

# Find all trial directories
es_trials = sorted(glob.glob(os.path.join(output_dir, "es_ppo_trial_*")))
cma_trials = sorted(glob.glob(os.path.join(output_dir, "cma_ppo_trial_*")))

def parse_trial_progress(trial_dir, alg_name):
    """Parse progress from a single trial directory."""
    slurm_files = glob.glob(os.path.join(trial_dir, "slurm-*.out"))
    threshold_hit = None
    if not slurm_files:
        return None
    
    # Use the most recent slurm file
    slurm_file = max(slurm_files, key=os.path.getmtime)
    
    trial_num = os.path.basename(trial_dir).split('_')[-1]
    
    # Parse data
    iterations = []
    rewards = []
    best_rewards = []
    completed = False
    final_reward = None
    test_stats = None
    
    try:
        with open(slurm_file, 'r') as f:
            for line in f:
                # Threshold reached
                match = re.search(r'Threshold reached! Freezing policy at iteration (\d+)', line)
                if match:
                    threshold_hit = int(match.group(1))
                    completed = True
                
                # Test statistics - parse from lines (only when in stats section)
                if 'FROZEN POLICY TEST STATISTICS' in line:
                    test_stats = {}
                elif test_stats is not None and 'Mean:' in line:
                    match = re.search(r'Mean:\s+([-\d.]+)', line)
                    if match:
                        test_stats['mean'] = float(match.group(1))
                elif test_stats is not None and 'Std:' in line:
                    match = re.search(r'Std:\s+([-\d.]+)', line)
                    if match:
                        test_stats['std'] = float(match.group(1))
                elif test_stats is not None and 'Min:' in line:
                    match = re.search(r'Min:\s+([-\d.]+)', line)
                    if match:
                        test_stats['min'] = float(match.group(1))
                elif test_stats is not None and 'Max:' in line:
                    match = re.search(r'Max:\s+([-\d.]+)', line)
                    if match:
                        test_stats['max'] = float(match.group(1))
                elif test_stats is not None and 'Median:' in line:
                    match = re.search(r'Median:\s+([-\d.]+)', line)
                    if match:
                        test_stats['median'] = float(match.group(1))
                elif test_stats is not None and '=' * 80 in line:
                    # End of stats section
                    pass
                
                # Trial completion
                match = re.search(r'Trial (\d+) completed: final reward = ([\d\.-]+)', line)
                if match:
                    completed = True
                    final_reward = float(match.group(2))
                
                # Iteration progress
                match = re.search(r'Trial (\d+), Iter (\d+): reward = ([\d\.-]+), best = ([\d\.-]+)', line)
                if match:
                    iter_num = int(match.group(2))
                    reward = float(match.group(3))
                    best = float(match.group(4))
                    iterations.append(iter_num)
                    rewards.append(reward)
                    best_rewards.append(best)
                    
                    # Check if this iteration hit threshold
                    if threshold_hit is None and reward >= 250:
                        threshold_hit = iter_num
    except Exception as e:
        return {
            'trial': trial_num,
            'alg': alg_name,
            'status': 'error',
            'error': str(e)
        }
    
    if not iterations:
        return {
            'trial': trial_num,
            'alg': alg_name,
            'status': 'no_data',
            'file': slurm_file
        }
    
    current_iter = max(iterations) if iterations else 0
    max_iter = None
    
    # Try to find max iterations from progress bar
    try:
        with open(slurm_file, 'r') as f:
            for line in f:
                match = re.search(r'Trial \d+/\d+:\s+\d+%\|.*\|\s+(\d+)/(\d+)', line)
                if match:
                    max_iter = int(match.group(2))
    except:
        pass
    
    best_reward = max(best_rewards) if best_rewards else None
    latest_reward = rewards[-1] if rewards else None
    
    status = 'completed' if completed else 'running'
    if threshold_hit:
        status = 'threshold_hit'
    
    return {
        'trial': trial_num,
        'alg': alg_name,
        'status': status,
        'current_iter': current_iter,
        'max_iter': max_iter,
        'progress_pct': (current_iter / max_iter * 100) if max_iter else None,
        'latest_reward': latest_reward,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'threshold_hit': threshold_hit,
        'test_stats': test_stats,
        'file': slurm_file
    }

# Parse all trials
all_trials = []

for trial_dir in es_trials:
    result = parse_trial_progress(trial_dir, 'ES-PPO')
    if result:
        all_trials.append(result)

for trial_dir in cma_trials:
    result = parse_trial_progress(trial_dir, 'CMA-PPO')
    if result:
        all_trials.append(result)

# Display results
print("=" * 80)
print("TRIAL PROGRESS SUMMARY")
print("=" * 80)
print()

# Group by algorithm
es_trials_data = [t for t in all_trials if t['alg'] == 'ES-PPO']
cma_trials_data = [t for t in all_trials if t['alg'] == 'CMA-PPO']

if es_trials_data:
    print("ES-PPO Trials:")
    print("-" * 80)
    for trial in sorted(es_trials_data, key=lambda x: int(x['trial'])):
        if trial['status'] == 'threshold_hit':
            stats_info = ""
            if trial.get('test_stats'):
                s = trial['test_stats']
                stats_info = f" | Test stats: mean={s.get('mean', 'N/A'):.2f}, std={s.get('std', 'N/A'):.2f}"
            print(f"  Trial {trial['trial']}: ✓ THRESHOLD HIT at iter {trial['threshold_hit']} - Reward: {trial['latest_reward']:.2f}, Best: {trial['best_reward']:.2f}{stats_info}")
        elif trial['status'] == 'completed':
            print(f"  Trial {trial['trial']}: COMPLETED - Final: {trial['final_reward']:.2f}, Best: {trial['best_reward']:.2f}")
        elif trial['status'] == 'running':
            progress = f"{trial['progress_pct']:.1f}%" if trial['progress_pct'] else "?"
            threshold_info = f" (threshold at iter {trial['threshold_hit']})" if trial.get('threshold_hit') else ""
            print(f"  Trial {trial['trial']}: RUNNING - Iter {trial['current_iter']}/{trial['max_iter']} ({progress}), Best: {trial['best_reward']:.2f}{threshold_info}")
        elif trial['status'] == 'no_data':
            print(f"  Trial {trial['trial']}: NO DATA (file: {os.path.basename(trial['file'])})")
        else:
            print(f"  Trial {trial['trial']}: ERROR - {trial.get('error', 'Unknown')}")
    print()

if cma_trials_data:
    print("CMA-PPO Trials:")
    print("-" * 80)
    for trial in sorted(cma_trials_data, key=lambda x: int(x['trial'])):
        if trial['status'] == 'threshold_hit':
            stats_info = ""
            if trial.get('test_stats'):
                s = trial['test_stats']
                stats_info = f" | Test stats: mean={s.get('mean', 'N/A'):.2f}, std={s.get('std', 'N/A'):.2f}"
            print(f"  Trial {trial['trial']}: ✓ THRESHOLD HIT at iter {trial['threshold_hit']} - Reward: {trial['latest_reward']:.2f}, Best: {trial['best_reward']:.2f}{stats_info}")
        elif trial['status'] == 'completed':
            print(f"  Trial {trial['trial']}: COMPLETED - Final: {trial['final_reward']:.2f}, Best: {trial['best_reward']:.2f}")
        elif trial['status'] == 'running':
            progress = f"{trial['progress_pct']:.1f}%" if trial['progress_pct'] else "?"
            threshold_info = f" (threshold at iter {trial['threshold_hit']})" if trial.get('threshold_hit') else ""
            print(f"  Trial {trial['trial']}: RUNNING - Iter {trial['current_iter']}/{trial['max_iter']} ({progress}), Best: {trial['best_reward']:.2f}{threshold_info}")
        elif trial['status'] == 'no_data':
            print(f"  Trial {trial['trial']}: NO DATA (file: {os.path.basename(trial['file'])})")
        else:
            print(f"  Trial {trial['trial']}: ERROR - {trial.get('error', 'Unknown')}")
    print()

# Summary statistics
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

if es_trials_data:
    es_completed = [t for t in es_trials_data if t['status'] == 'completed']
    es_running = [t for t in es_trials_data if t['status'] == 'running']
    
    if es_completed:
        es_finals = [t['final_reward'] for t in es_completed]
        es_bests = [t['best_reward'] for t in es_completed if t['best_reward']]
        print(f"ES-PPO Completed: {len(es_completed)}/3")
        print(f"  Average final reward: {sum(es_finals)/len(es_finals):.2f}")
        print(f"  Best final reward: {max(es_finals):.2f}")
        if es_bests:
            print(f"  Best reward across all: {max(es_bests):.2f}")
    if es_running:
        print(f"ES-PPO Running: {len(es_running)}/3")
    print()

if cma_trials_data:
    cma_completed = [t for t in cma_trials_data if t['status'] == 'completed']
    cma_running = [t for t in cma_trials_data if t['status'] == 'running']
    
    if cma_completed:
        cma_finals = [t['final_reward'] for t in cma_completed]
        cma_bests = [t['best_reward'] for t in cma_completed if t['best_reward']]
        print(f"CMA-PPO Completed: {len(cma_completed)}/3")
        print(f"  Average final reward: {sum(cma_finals)/len(cma_finals):.2f}")
        print(f"  Best final reward: {max(cma_finals):.2f}")
        if cma_bests:
            print(f"  Best reward across all: {max(cma_bests):.2f}")
    if cma_running:
        print(f"CMA-PPO Running: {len(cma_running)}/3")
    print()

# Overall comparison
if es_trials_data and cma_trials_data:
    es_bests = [t['best_reward'] for t in es_trials_data if t['best_reward']]
    cma_bests = [t['best_reward'] for t in cma_trials_data if t['best_reward']]
    
    if es_bests and cma_bests:
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"ES-PPO best: {max(es_bests):.2f}")
        print(f"CMA-PPO best: {max(cma_bests):.2f}")
        if max(es_bests) > max(cma_bests):
            print(f"ES-PPO leads by {max(es_bests) - max(cma_bests):.2f} points")
        else:
            print(f"CMA-PPO leads by {max(cma_bests) - max(es_bests):.2f} points")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Job Status (from Slurm):"
echo "=========================================="
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs found or squeue unavailable"

