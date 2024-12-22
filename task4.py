""" 
Training the agent on the source environment and then test it on the source environment. 
Using default hyperparameters
"""

import argparse

import torch
import gym
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import *

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--env-source', default='CustomHopper-source-v0', type=str, help='Source environment')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

ENV_ID = args.env_source

def train():
    eval_env = Monitor(gym.make(ENV_ID)) 

    model = PPO(args.policy, env = ENV_ID)

    model.learn(args.n_episodes)

    return eval_env, model

def test(eval_env, model):
    mean, std = evaluate_policy(model, eval_env, args.n_eval_episodes)
    print(f'mean reward={mean}, std={std}')

     

def main():

    eval_env, model = train()

    test(eval_env, model)

if __name__ == '__main__':
	main()
    