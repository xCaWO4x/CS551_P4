import argparse

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='./checkpoints', help='experiment directory')
    parser.add_argument('-a', '--alg', type=str, help='ES or PPO or ESPPO or MAXPPO')
    parser.add_argument('--render', type=int, default=0)
    
    parser.add_argument('--population_size', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=0.1)

    parser.add_argument('--max_steps', type=int, default=256)
    parser.add_argument('--n_updates', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--ent_coeff', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=0.001) # TODO: delete
    parser.add_argument('--es_lr', type=float, default=0.001)
    parser.add_argument('--ppo_lr', type=float, default=0.0001)
    
    # CMA-PPO specific arguments
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--cma_lr_mean', type=float, default=3e-4, help='CMA-PPO mean network learning rate')
    parser.add_argument('--cma_lr_var', type=float, default=3e-4, help='CMA-PPO variance network learning rate')
    parser.add_argument('--cma_lr_value', type=float, default=1e-3, help='CMA-PPO value network learning rate')
    parser.add_argument('--history_size', type=int, default=5, help='CMA-PPO history buffer size (H)')
    parser.add_argument('--kernel_std', type=float, default=0.1, help='CMA-PPO Gaussian kernel std for mirroring')
    
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_seq', type=int, default=1)

    parser.add_argument('--n_alt', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10000)

    args = parser.parse_args()
    return args
