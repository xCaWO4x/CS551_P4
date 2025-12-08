#!/usr/bin/env python3
"""
Plot all trials for a given algorithm on the same graph.
Works with the experimental structure: independent_trials_*/{algorithm}_trial_{n}/
"""
import re
import glob
import os
import sys
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not available. Creating data file instead.")
    HAS_MATPLOTLIB = False
    import numpy as np

def extract_rewards_from_trial(trial_file):
    """Extract reward data from a trial output file."""
    rewards = []
    iterations = []
    threshold_hit = None
    
    with open(trial_file, 'r') as f:
        for line in f:
            # Match: "Trial 1, Iter 450: reward = 191.42, best = 191.42"
            match = re.search(r'Trial \d+, Iter (\d+): reward = ([-\d.]+), best = ([-\d.]+)', line)
            if match:
                iter_num = int(match.group(1))
                reward = float(match.group(2))
                rewards.append(reward)
                iterations.append(iter_num)
            
            # Check if threshold was hit
            if 'Threshold reached!' in line or 'Freezing policy' in line:
                match = re.search(r'iteration (\d+)', line)
                if match:
                    threshold_hit = int(match.group(1))
    
    return np.array(iterations), np.array(rewards), threshold_hit

def find_trial_directories(base_dir, algorithm):
    """Find all trial directories for an algorithm."""
    pattern = os.path.join(base_dir, f"{algorithm}_trial_*")
    return sorted(glob.glob(pattern))

def plot_algorithm_trials(base_dir, algorithm, threshold=250):
    """Plot all trials for a given algorithm."""
    trial_dirs = find_trial_directories(base_dir, algorithm)
    
    if not trial_dirs:
        print(f"Warning: No trial directories found for {algorithm} in {base_dir}")
        return None
    
    all_trials_data = []
    
    for trial_dir in trial_dirs:
        # Extract trial number from directory name
        trial_match = re.search(r'trial_(\d+)', os.path.basename(trial_dir))
        if not trial_match:
            continue
        
        trial_num = int(trial_match.group(1))
        slurm_files = glob.glob(os.path.join(trial_dir, "slurm-*.out"))
        
        if not slurm_files:
            print(f"Warning: No output file found for {algorithm} trial {trial_num}")
            continue
        
        trial_file = max(slurm_files, key=os.path.getmtime)
        iterations, rewards, threshold_hit = extract_rewards_from_trial(trial_file)
        
        if len(rewards) > 0:
            all_trials_data.append({
                'trial': trial_num,
                'iterations': iterations,
                'rewards': rewards,
                'best': np.max(rewards),
                'final': rewards[-1],
                'avg': np.mean(rewards),
                'threshold_hit': threshold_hit,
                'threshold_iter': threshold_hit if threshold_hit else None
            })
            status = f"hit threshold at iter {threshold_hit}" if threshold_hit else "did not hit threshold"
            print(f"{algorithm.upper()} Trial {trial_num}: {len(rewards)} evals, best={np.max(rewards):.2f}, final={rewards[-1]:.2f}, {status}")
    
    if not all_trials_data:
        print(f"Error: No trial data found for {algorithm}!")
        return None
    
    if not HAS_MATPLOTLIB:
        # Save data to CSV instead
        csv_path = os.path.join(base_dir, f'{algorithm}_all_trials_data.csv')
        with open(csv_path, 'w') as f:
            f.write("trial,iteration,reward,threshold_hit\n")
            for trial_data in all_trials_data:
                for iter_val, reward_val in zip(trial_data['iterations'], trial_data['rewards']):
                    hit = 1 if (trial_data['threshold_hit'] and iter_val >= trial_data['threshold_hit']) else 0
                    f.write(f"{trial_data['trial']},{iter_val},{reward_val},{hit}\n")
        print(f"\nData saved to CSV: {csv_path}")
        return None
    
    # Create the main plot
    plt.figure(figsize=(14, 8))
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trials_data)))
    
    for i, trial_data in enumerate(all_trials_data):
        plt.plot(
            trial_data['iterations'],
            trial_data['rewards'],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
            label=f"Trial {trial_data['trial']} (best={trial_data['best']:.2f}, final={trial_data['final']:.2f})"
        )
        
        # Mark the best point
        best_idx = np.argmax(trial_data['rewards'])
        plt.scatter(
            trial_data['iterations'][best_idx],
            trial_data['rewards'][best_idx],
            color=colors[i],
            s=100,
            marker='*',
            edgecolors='black',
            linewidths=1,
            zorder=5
        )
        
        # Mark threshold hit point if applicable
        if trial_data['threshold_hit']:
            # Find the iteration index where threshold was hit
            threshold_idx = np.where(trial_data['iterations'] >= trial_data['threshold_hit'])[0]
            if len(threshold_idx) > 0:
                idx = threshold_idx[0]
                plt.scatter(
                    trial_data['iterations'][idx],
                    trial_data['rewards'][idx],
                    color=colors[i],
                    s=150,
                    marker='D',
                    edgecolors='red',
                    linewidths=2,
                    zorder=6,
                    label=f"Trial {trial_data['trial']} threshold hit"
                )
    
    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Threshold ({threshold})')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Calculate and display overall statistics
    all_rewards = np.concatenate([t['rewards'] for t in all_trials_data])
    threshold_hits = [t['threshold_hit'] for t in all_trials_data if t['threshold_hit']]
    
    stats_text = f"""Statistics:
Best overall: {np.max(all_rewards):.2f}
Final avg: {np.mean([t['final'] for t in all_trials_data]):.2f}
Best avg: {np.mean([t['best'] for t in all_trials_data]):.2f}
Overall avg: {np.mean(all_rewards):.2f}
Threshold hits: {len(threshold_hits)}/{len(all_trials_data)}"""
    
    if threshold_hits:
        stats_text += f"\nAvg iterations to threshold: {np.mean(threshold_hits):.1f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Reward', fontsize=12, fontweight='bold')
    plt.title(f'{algorithm.upper()}: All Trials - Reward vs Iteration (Threshold={threshold})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(base_dir, f'{algorithm}_all_trials_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()
    
    return all_trials_data

def main():
    parser = argparse.ArgumentParser(description='Plot trial results for algorithms')
    parser.add_argument('--algorithm', type=str, choices=['cma_ppo', 'es_ppo', 'both'],
                        default='both', help='Algorithm to plot')
    parser.add_argument('--directory', type=str, default=None,
                        help='Base directory (default: latest independent_trials_*)')
    parser.add_argument('--threshold', type=float, default=250.0,
                        help='Reward threshold (default: 250)')
    
    args = parser.parse_args()
    
    # Find base directory
    if args.directory:
        base_dir = args.directory
    else:
        trial_dirs = sorted(glob.glob("independent_trials_*"))
        if not trial_dirs:
            print("Error: No independent_trials_* directory found!")
            return
        base_dir = trial_dirs[-1]
    
    print(f"Using directory: {base_dir}")
    print(f"Threshold: {args.threshold}")
    print()
    
    if args.algorithm in ['cma_ppo', 'both']:
        plot_algorithm_trials(base_dir, 'cma_ppo', args.threshold)
    
    if args.algorithm in ['es_ppo', 'both']:
        plot_algorithm_trials(base_dir, 'es_ppo', args.threshold)
    
    print("\nPlotting complete!")

if __name__ == '__main__':
    main()

