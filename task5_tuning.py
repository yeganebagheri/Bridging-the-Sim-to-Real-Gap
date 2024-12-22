""" Optuna that optimizes the hyperparameters of
a reinforcement learning agent using PPO implementation from Stable-Baselines3
on a Gym environment.

Using RandomSampler and a personalized pruning through evalcallback.

You can run this example as follows:
    $ python3 task5_tuning.py

"""
import argparse

from typing import Any
from typing import Dict

import gym
import optuna
from optuna.samplers import RandomSampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn
import random
import os 

from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--n-eval-episodes', default=50, type=int, help='Number of evaluation episodes')
    parser.add_argument('--env-source', default='CustomHopper-target-v0', type=str, help='Source environment')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--seed', default=0, type=int, help='Seed')

    return parser.parse_args()

args = parse_args()

N_TRIALS = 100
N_STARTUP_TRIALS = 15
N_EVALUATIONS = 10
N_TIMESTEPS = args.n_episodes
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = args.n_eval_episodes

ENV_ID = args.env_source

SEED = args.seed
os.environ['PYTHONASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:

    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:   """

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 3e-3, 5e-3, 1e-3, 3e-4, 5e-4])

    if batch_size >= n_steps:
        batch_size = n_steps
  
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
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
        tolerance: int = 50,
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
        self.tolerance = tolerance
        self.check = 0
        self.prev_mean_reward = -float('inf')
        self.even_prev_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            current_mean_reward = self.last_mean_reward
            self.trial.report(current_mean_reward, self.eval_idx)

            self.tolerance = max(self.tolerance, 0.1*self.prev_mean_reward)

            # Check if the current mean reward is higher than the last mean reward within tolerance. 
            # Prune after the second time the current mean reward is lower than the last mean reward within tolerance
            if current_mean_reward < self.prev_mean_reward - self.tolerance:
                if self.check > 0:
                    self.is_pruned = True
                    return False
                else:
                    self.check += 1
            
            # Check if the current mean reward is higher than the two last mean rewards 
            # To check that overall is actually going upwards
            # Prune if the current mean is lower than the two last mean rewards
            if (current_mean_reward < self.even_prev_mean_reward) & (current_mean_reward < self.prev_mean_reward):
                self.is_pruned = True
                return False

            self.even_prev_mean_reward = self.prev_mean_reward
            self.prev_mean_reward = current_mean_reward
            
        return True

def objective(trial: optuna.Trial) -> float:

    kwargs = sample_ppo_params(trial)

    train_env = gym.make(ENV_ID)
    eval_env = Monitor(gym.make(ENV_ID))

    model = PPO('MlpPolicy', env=train_env, seed = SEED, **kwargs, verbose=0)
    
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True, verbose = 0
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

    sampler = RandomSampler()

    study = optuna.create_study(sampler=sampler, direction="maximize") 
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
