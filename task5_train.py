"""
Compare the performance between Source->Source, Source->Target and Target->Target. 
Using the hyperparameters found through task5_tuning.py

You can run this example as follows:
    $ python3 task5_train.py
"""

import argparse
import torch
import gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
import os 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=1000000, type=int, help='Number of training episodes')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--eval-freq', default=100000, type=int, help='Evaluation frequence')
    parser.add_argument('--env-source', default='CustomHopper-source-v0', type=str, help='Source environment')
    parser.add_argument('--env-target', default='CustomHopper-target-v0', type=str, help='Target environment')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning Rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='Number of steps')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount Factor (Gamma)')
    parser.add_argument('--seed', default=0, type = int, help = "Seed")

    return parser.parse_args()

args = parse_args()

N_TIMESTEPS = args.n_episodes
EVAL_FREQ = args.eval_freq
N_EVAL_EPISODES = args.n_eval_episodes

ENV_ID_SOURCE = args.env_source
ENV_ID_TARGET = args.env_target

SEED = args.seed
os.environ['PYTHONASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
np.random.seed(SEED)
random.seed(SEED)

DEFAULT_HYPERPARAMS = {
    "policy": args.policy,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "gamma": args.gamma,
    "n_steps": args.n_steps,
    "seed": SEED,
}

def train(env_id_train, env_id_test, log_path):
    kwargs = DEFAULT_HYPERPARAMS.copy()

    train_env = gym.make(env_id_train)
    test_env = Monitor(gym.make(env_id_test)) 

    agent = PPO(**kwargs, env=train_env)
    
    eval_callback= EvalCallback(
       test_env, best_model_save_path=log_path, log_path=log_path, eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
    
    agent.learn(N_TIMESTEPS, callback=eval_callback)

    return agent, test_env

def test(agent, test_env):
    mean, std = evaluate_policy(agent, test_env, N_EVAL_EPISODES)
    return mean, std

def main():
    #Source->Source
    agent, test_env = train(ENV_ID_SOURCE, ENV_ID_SOURCE, "./logs/source_source/")
    mean, std = test(agent, test_env)

    print(f'Source→Source: mean reward={mean}, std={std}')

    #Source->Target
    agent, test_env = train(ENV_ID_SOURCE, ENV_ID_TARGET, './logs/source_target/')
    mean, std = test(agent, test_env)
    
    print(f'Source→Target (lower bound): mean reward={mean}, std={std}')

    #Target->Target
    agent, test_env = train(ENV_ID_TARGET, ENV_ID_TARGET, './logs/target_target/')
    mean, std = test(agent, test_env)
    print(f'Target→Target (upper bound): mean reward={mean}, std={std}')

if __name__ == '__main__':
    main()
