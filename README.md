# Bridging the Sim to Real Gap with Domain Randomization
This repository contains the code for the Reinforcement Learning project in the MLDL course at PoliTO. \
Students: Yegane Bagheri and Chiara Roberta Casale.

## REINFORCE
- `agent_reinforce.py` Agent for REINFORCE without a baseline and REINFORCE with baseline = 20
- `task2_3_train.py` to train the agent
- `task2_3_test.py` to test the agent

## Actor-Critic
- `agent_ac.py` Agent for Actor-Critic
- `task2_3_train.py` to train the agent
- `task2_3_test.py` to test the agent

## PPO
- `task4.py` to train and test PPO in the source env with default parameters
- `task5_tuning.py` to tune PPO (through Optuna, train and test in the target env)
- `task5_train.py` to train and test PPO in Source -> Source, Source -> Target, Target -> Target with the hyperparameters found through task5_tuning.py

## Uniform Domain Randomization
- `env_udr` environment with functions to make UDR work
- `task6_tuning.py` to tune the parameters for UDR
- `task6_train.py` to train and test UDR in Source -> Target (and Source -> Source)

## Automatic Domain Randomization
- `env_autodr` environment with functions to make AutoDR work
- `adr.py` functions to define AutoDR
- `adr_train.py` to train and test AutoDR in Source -> Target
