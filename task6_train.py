""" 
Training the model with the range and scaling found through Optuna (see task6_tuning.py)

You can run this example as follows:
    $ python3 task6_train.py

"""

from typing import Any
from typing import Dict

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy

import torch
import torch.nn as nn
import argparse
import random
import os 
from env_udr.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=1000000, type=int, help='Number of training episodes')
    parser.add_argument('--eval-freq', default=20000, type=int, help='Evaluation frequence')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--env-train', default='CustomHopper-source-udr-v0', type=str, help='Source environment')
    parser.add_argument('--env-eval-source', default='CustomHopper-source-v0', type=str, help='Target environment')
    parser.add_argument('--env-eval-target', default='CustomHopper-target-v0', type=str, help='Target environment')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning Rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='Number of steps')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount Factor (Gamma)')
    parser.add_argument('--debug', default='False', type=str, help='Print model masses at each episode')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--range', default=0.5, type=float, help='Range UDR')
    parser.add_argument('--scaling', default=1, type=float, help='Scaling UDR')
	
    return parser.parse_args()

args = parse_args()
args.debug = (args.debug == 'True')
DEBUG = args.debug

N_TIMESTEPS = args.n_episodes
N_EVAL_EPISODES = args.n_eval_episodes
EVAL_FREQ = args.eval_freq

ENV_ID_TRAIN = args.env_train
ENV_ID_EVAL_SOURCE = args.env_eval_source
ENV_ID_EVAL_TARGET = args.env_eval_target

SEED = args.seed
os.environ['PYTHONASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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

SCALING = args.scaling
RANGE = args.range

def compute_bounds(params, min_mass):
	bounds = list((max(min_mass, mean - range), mean+range) for mean, range in 
               [(params['thigh_mean'], params['range_thigh']), (params['leg_mean'], params['range_leg']), 
                (params['foot_mean'], params['range_foot'])])
	return bounds


if __name__ == "__main__":
    kwargs = DEFAULT_HYPERPARAMS.copy()

    params = {
		'thigh_mean': SCALING*3.92699082,
		'leg_mean': SCALING*2.71433605,
		'foot_mean': SCALING*5.0893801,
        'range_thigh': RANGE, 
        'range_leg': RANGE,
        'range_foot': RANGE, 
	}

    bounds = compute_bounds(params, min_mass=0.1)

    train_env = gym.make(ENV_ID_TRAIN, seed=SEED)
    train_env.set_bounds(bounds)
    
    if DEBUG:
        train_env.set_debug()

    eval_env = Monitor(gym.make(ENV_ID_EVAL_TARGET))
    eval_env_source = Monitor(gym.make(ENV_ID_EVAL_SOURCE))

    model = PPO(**kwargs, env = train_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/task_6',
                             log_path='./logs/task_6', eval_freq=EVAL_FREQ,
                             n_eval_episodes=N_EVAL_EPISODES, deterministic=True)

    model.learn(N_TIMESTEPS, callback=eval_callback)

    mean, std = evaluate_policy(model, eval_env, N_EVAL_EPISODES)
    print(f'Source→Target. Mean reward={mean}, std={std}')

    mean, std = evaluate_policy(model, eval_env_source, N_EVAL_EPISODES)
    print(f'Source→Source. Mean reward={mean}, std={std}')