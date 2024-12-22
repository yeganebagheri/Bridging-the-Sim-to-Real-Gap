import torch
import torch.nn.functional as F
from torch.distributions import Normal

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))): #This process ensures that rewards received earlier in the episode are influenced by the rewards received later by considering running_add and reversed loop
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Policy(torch.nn.Module): #The class inherits from torch.nn.Module, which is the base class for all neural network modules in PyTorch.
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64 # The number of hidden units in the hidden layers of the neural network. Here it is set to 64.
        self.tanh = torch.nn.Tanh() # The hyperbolic tangent activation function, which will be used in the hidden layers.

        """
            Actor network
        """
        #actor network (for selecting actions)
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5 #A learnable parameter that represents the standard deviation of the action distribution, initialized to init_sigma.
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


# This method initializes the weights of all linear layers to values drawn from a normal distribution and sets the biases to zero.
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)



    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma) # Actions can be sampled from this distribution during training.


        """
            Critic
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        return normal_dist, state_value

class Agent(object): #The Agent class is designed to interact with the environment using a given policy, collect experiences, and update the policy based on these experiences.
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3) #The optimizer used for updating the policy parameters, here Adam with a learning rate of 0.001.

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        _, state_values = self.policy(states)
        _, next_state_values = self.policy(next_states)
        state_values = state_values.squeeze(-1)
        next_state_values = next_state_values.squeeze(-1)

        #bootstrapped returns
        returns = rewards + self.gamma * next_state_values * (1.0 - done)

        # Compute advantages
        advantages = returns - state_values

        # Actor loss
        actor_loss = (-action_log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = F.mse_loss(state_values, returns.detach())

        self.optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        self.optimizer.step()

        return


    def get_action(self, state, evaluation=False): #This method selects an action given the current state.
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean #A flag indicating whether to return the mean action (for evaluation) or sample from the action distribution (for training).
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done): #This method stores the outcome of an interaction with the environment.

        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        if done:
            self.update_policy()