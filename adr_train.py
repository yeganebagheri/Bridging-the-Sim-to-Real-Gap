""" 
Training Automatic Domain Randomization.

You can run this example as follows:
    $ python3 adr_train.py

"""

from env_autodr.custom_hopper import *
from adr import Randomizer, RandomizerCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import torch
import torch.nn as nn
import argparse
import random
import os 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=1000000, type=int, help='Number of training episodes')
    parser.add_argument('--eval-freq', default=100000, type=int, help='Evaluation frequence')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--env-source', default='CustomHopper-source-v0', type=str, help='Source environment')
    parser.add_argument('--env-target', default='CustomHopper-target-v0', type=str, help='Target environment')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--probability', default=0.5, type=float, help='Probability of evalutating the training performance with AutoDR')
    parser.add_argument('--buffer', default=25, type=int, help='Data buffer size for AutoDR')
    parser.add_argument('--lower-threshold', default=900, type=int, help='Lower threshold for performance evaluation for AutoDR')
    parser.add_argument('--upper-threshold', default=1300, type=int, help='Upper threshold for performance evaluation for AutoDR')
    parser.add_argument('--delta', default=0.075, type=float, help='Delta - Step for bounds increasing for AutoDR')
    parser.add_argument('--type', default = 'constant', type= str, help = 'Defines the behavior of delta in AutoDR. Can be constant, proportional, proportional_log')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning Rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='Number of steps')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount Factor (Gamma)')
    parser.add_argument('--log-path-tb', default='./logs_TensorBoard/', type=str, help='Tensorboard log path for AutoDR_callback')
    parser.add_argument('--log-path-eval', default='./logs/autodr/eval/', type=str, help='Log path for target evaluation during training for EvalCallback')
    parser.add_argument('--best-model-path', default='./models/autodr/best_eval/', type=str, help='Log path for the best model found so far during training for EvalCallback')
    parser.add_argument('--debug', default='False', type=str, help='Print model masses at each episode')
    return parser.parse_args()

args = parse_args()

args.debug = (args.debug == 'True')
DEBUG = args.debug

N_TIMESTEPS = args.n_episodes
N_EVAL_EPISODES = args.n_eval_episodes
EVAL_FREQ = args.eval_freq

ENV_ID_SOURCE = args.env_source
ENV_ID_TARGET = args.env_target

SEED = args.seed
os.environ['PYTHONASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(SEED)

DEFAULT_HYPERPARAMS = {
    "policy": args.policy,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "gamma": args.gamma,
    "n_steps": args.n_steps,
    "seed": SEED,
}

def main():
    kwargs = DEFAULT_HYPERPARAMS.copy()

    AutoDR = Randomizer(probability=args.probability, buffer_size=args.buffer, init_delta=args.delta, 
                  lower_threshold=args.lower_threshold, upper_threshold=args.upper_threshold, type = args.type)
    
    train_env = gym.make(ENV_ID_SOURCE)
    
    if DEBUG:
        train_env.set_debug()
        
    test_env =  Monitor(gym.make(ENV_ID_TARGET)) 
    eval_callback = EvalCallback(eval_env=test_env, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, 
                              log_path=args.log_path_eval, deterministic=True, best_model_save_path=args.best_model_path) 
    
    autodr_callback = RandomizerCallback(AutoDR, train_env)
        
    model = PPO(**kwargs, env=train_env, tensorboard_log=args.log_path_tb)
    
    callbacks = CallbackList([autodr_callback, eval_callback]) 

    model.learn(total_timesteps=N_TIMESTEPS, callback=callbacks)

    train_env.close()

if __name__ == '__main__':
    main()