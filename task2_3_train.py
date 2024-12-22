""" 
Training the agent with REINFORCE or Actor-Critic. Testing it every 250 episodes to see the performance grow. 
Uploading the results on wandb.

"""

import torch
import gym
import argparse
import pickle
import re
import wandb  

from env.custom_hopper import *
from agent_reinforce import Agent, Policy
#from agent_ac import Agent, Policy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=20000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
	parser.add_argument('--test-every', default=250, type=int, help='Test the agent every <> episodes')
	parser.add_argument('--n-episodes-test', default=10, type=int, help='Number of test episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
	parser.add_argument('--model', default=None, type=str, help='Model path')
	return parser.parse_args()

args = parse_args()

def test_agent(env, agent):
	episodes = args.n_episodes_test
	tot_reward = 0

	#returns = []
	for episode in range(episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		tot_reward +=test_reward
	tot_reward = tot_reward/episodes
	return tot_reward

def main():
	wandb.init(project="MLDL-Project", group="REINFORCE with baseline")  # Initialize wandb
	env = gym.make('CustomHopper-source-v0')
	test_env = gym.make('CustomHopper-source-v0')
	
    # env = gym.make('CustomHopper-target-v0')
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)

	if args.model is None:
		starting_episode = 0
	else:
		starting_episode = re.search(r'\d+',args.model) 
		#All the code about saving the model is needed in case it stops before finishing the run
		print(starting_episode)
		if starting_episode is None:
			starting_episode = 0
		else:
			starting_episode = int(starting_episode.group(0))
		policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device)

	print('Starting Episode:', starting_episode)
	train_returns = []
	
	for episode in range(starting_episode, args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
			
		train_returns.append(train_reward)


		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

			torch.save(agent.policy.state_dict(), f"model-{episode}.mdl")
			with open(f"train_returns-{episode}.pickle","wb") as outf:
				pickle.dump(obj=train_returns, file=outf)
		if (episode+1)%args.test_every == 0:
			test_reward = test_agent(test_env, agent)
			wandb.log({"test_reward": test_reward}, step=episode)

if __name__ == '__main__':
	main()