""" 
Functions to use Automatic Domain Randomization
"""
import gym
import random
import math
from typing import List, Tuple
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class Randomizer:
    def __init__(self, probability: float, buffer_size: int, init_delta: float, lower_threshold: int, upper_threshold: int, type : str) -> None:
        eps = 1e-3

        self.init_delta = init_delta
        self.type = type
        self.delta = self.init_delta

        bounds = [
            {
                "name": "thigh",
                "lower_bound": {"value": 3.92699082, "min_value": 0.0 + eps, "max_value": 3.92699082},
                "upper_bound": {"value": 3.92699082, "min_value": 3.92699082, "max_value": 2 * 3.92699082},
                "delta": self.delta
            },
            {
                "name": "leg",
                "lower_bound": {"value": 2.71433605, "min_value": 0.0 + eps, "max_value": 2.71433605},
                "upper_bound": {"value": 2.71433605, "min_value": 2.71433605, "max_value": 2 * 2.71433605},
                "delta": self.delta
            },
            {
                "name": "foot",
                "lower_bound": {"value": 5.0893801, "min_value": 0.0 + eps, "max_value": 5.0893801},
                "upper_bound": {"value": 5.0893801, "min_value": 5.0893801, "max_value": 2 * 5.0893801},
                "delta": self.delta
            }
        ]

        self.parameters = {}
        self.buffer = {}
        self.buffer_size = buffer_size

        for param in bounds:
            lower_bound = {
                'type': "lower_bound",
                'value': param["lower_bound"]["value"],
                'min_value': param["lower_bound"]["min_value"],
                'max_value': param["lower_bound"]["max_value"]
            }
            upper_bound = {
                'type': "upper_bound",
                'value': param["upper_bound"]["value"],
                'min_value': param["upper_bound"]["min_value"],
                'max_value': param["upper_bound"]["max_value"]
            }
            self.parameters[param["name"]] = {
                'name': param["name"],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'delta': param["delta"]
            }
            self.buffer[param["name"]] = {
                "lower_bound": [],
                "upper_bound": []
            }

        self.probability = probability

        self.sampled_boundary = None
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def insert_episode(self, parameter_name: str, bound_type: str, episode_return: float) -> None:
        self.buffer[parameter_name][bound_type].append(episode_return)

    def evaluate(self, param_name: str, bound_type: str) -> None:
        if len(self.buffer[param_name][bound_type]) >= self.buffer_size: #Check if the buffer is full
            performance = np.mean(np.array(self.buffer[param_name][bound_type])) #Computer the performance with mean of the rewards
            self.buffer[param_name][bound_type].clear()
            
            if performance >= self.upper_threshold:
                if self.type == 'proportional':
                    self.delta = self.init_delta*((performance - self.upper_threshold)/100) #Set delta if proportional
                elif self.type == 'proportional_log':
                    self.delta = self.init_delta*(math.log10((performance - self.upper_threshold)/10)) #Set delta if proportional_log
                #If neither is true (i.e. it's constant, self.delta is already initialized as self.init_delta)
                if bound_type == "upper_bound":
                    self.increase(param_name, bound_type, self.delta)
                elif bound_type == "lower_bound":
                    self.decrease(param_name, bound_type, self.delta)

            if performance < self.lower_threshold:
                if self.type == 'proportional':
                    self.delta = self.init_delta*((-performance + self.lower_threshold)/100)
                elif self.type == 'proportional_log':
                    self.delta = self.init_delta*(math.log10((- performance + self.upper_threshold)/10))
                if bound_type == "upper_bound":
                    self.decrease(param_name, bound_type, self.delta)
                elif bound_type == "lower_bound":
                    self.increase(param_name, bound_type, self.delta)

    def _get_masses(self) -> dict:
        randomized_params = {}

        for param in self.parameters.values():
            lower_bound = param['lower_bound']
            upper_bound = param['upper_bound']
            randomized_params[param['name']] = random.uniform(lower_bound['value'], upper_bound['value']) #Compute the parameters

        return randomized_params

    def _get_boundary(self, parameters) -> Tuple[str, str]: #"sampling boundary" function
        param = parameters.copy()
        sampled_boundary = None

        if random.uniform(0, 1) <= self.probability:
            sampled_param = random.choice(list(self.parameters.values()))
            sampled_bound = random.choice([sampled_param['lower_bound'], sampled_param['upper_bound']])
            param[sampled_param['name']] = sampled_bound['value']
            sampled_boundary = (sampled_param['name'], sampled_bound['type'])

        return sampled_boundary, parameters

    def increase(self, param_name: str, bound_type: str, delta: float) -> None:
        param = self.parameters[param_name][bound_type]
        param['value'] = min(param['value'] + delta, param['max_value']) #Increase by delta, check that it's lower than max_value


    def decrease(self, param_name: str, bound_type: str, delta: float) -> None:
        param = self.parameters[param_name][bound_type]
        param['value'] = max(param['value'] - delta, param['min_value']) #Decrease by delta, check that it's higher than min_value


class RandomizerCallback(BaseCallback):
    def __init__(self, AutoDR, env, verbose=0):
        super().__init__(verbose=verbose)
        self.env=env
        self.autodr = AutoDR
    
    def _on_step(self) -> None:
        done, info = self.locals['dones'], self.locals['infos']

        if done:       
            if self.autodr.sampled_boundary is not None:
                param_name, bound_type = self.autodr.sampled_boundary
                self.autodr.insert_episode(param_name, bound_type, info[0]['episode']['r'])
                self.autodr.evaluate(param_name, bound_type)
                self.autodr.sampled_boundary = None  # Reset after evaluation
            
            if all(param['lower_bound']['value'] == param['upper_bound']['value'] for param in self.autodr.parameters.values()):
                #While all the lower_bound == upper_bound, it's redundant to set the parameters
                boundary, _ = self.autodr._get_boundary(self.autodr.parameters)
                self.autodr.sampled_boundary = boundary

            else:
                randomized_params = self.autodr._get_masses()
                boundary, randomized_params = self.autodr._get_boundary(randomized_params)
                self.autodr.sampled_boundary = boundary
                self.env.set_parameters(list(randomized_params.values()))
