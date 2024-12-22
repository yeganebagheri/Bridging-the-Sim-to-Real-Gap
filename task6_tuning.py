""" 
Using Optuna to find the best range for the training of UDR. Finding range and scaling for the three masses (thigh, leg, foot). 
Torso mass is fixed (see env_udr.custom_hopper.py)

You can run this example as follows:
    $ python3 task6_tuning.py

"""

import argparse

from typing import Any
from typing import Dict

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import random
import os 

from env_udr.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--env-train', default='CustomHopper-source-udr-v0', type=str, help='Source environment')
    parser.add_argument('--env-eval', default='CustomHopper-source-v0', type=str, help='Target environment')
    parser.add_argument('--policy', default='MlpPolicy', type=str, help='Policy')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning Rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='Number of steps')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount Factor (Gamma)')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--debug', default='False', type=str, help='Print model masses at each episode')
	
    return parser.parse_args()

args = parse_args()

args.debug = (args.debug == 'True')
DEBUG = args.debug

N_TRIALS = 30
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 10
N_TIMESTEPS = args.n_episodes
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = args.n_eval_episodes

ENV_ID_TRAIN = args.env_train
ENV_ID_EVAL = args.env_eval

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


def sample_range_params(trial: optuna.Trial) -> Dict[str, Any]:
    range = trial.suggest_float("range", 0.25, 2.5, step=0.25)
    scaling = trial.suggest_categorical("scaling", [0.75, 1, 1.5, 2])
    
    return {
        "range" : range,
        "scaling": scaling,
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5, 
        eval_freq: int = 10000, 
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def compute_bounds(params, min_mass):
	bounds = list((max(min_mass, mean - range), mean+range) for mean, range in 
               [(params['thigh_mean'], params['range_thigh']), (params['leg_mean'], params['range_leg']), 
                (params['foot_mean'], params['range_foot'])])
	return bounds

def objective(trial: optuna.Trial) -> float:
    
    kwargs = DEFAULT_HYPERPARAMS.copy()

    results = sample_range_params(trial)
    scaling = results.get('scaling')
    range = results.get('range')

    params = {
		'thigh_mean': scaling*3.92699082,
		'leg_mean': scaling*2.71433605,
		'foot_mean': scaling*5.0893801,
        'range_thigh': range,
        'range_leg': range,
        'range_foot': range,
	}

    bounds = compute_bounds(params, min_mass=0.1)

    train_env = gym.make(ENV_ID_TRAIN, seed=SEED)
    train_env.set_bounds(bounds)
    
    if DEBUG:
        train_env.set_debug()
    
    eval_env = Monitor(gym.make(ENV_ID_EVAL))

    model = PPO(**kwargs, env=train_env)
    
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )


    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    except ValueError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True   
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    mean, _ = evaluate_policy(model, eval_env, n_eval_episodes = N_EVAL_EPISODES)
    
    return mean


if __name__ == "__main__":

    search_space = {
        "range": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        "scaling": [0.75, 1, 1.5, 2]
    }

    sampler = GridSampler(search_space)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
