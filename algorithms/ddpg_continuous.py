import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class FeedForwardCritic(nn.Module):
    def __init__(self, in_dim, act_dim, hidden_dim=128, device=device, lr=1e-4):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

        self.actor_fc = nn.Linear(act_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.actor_fc(action)

        state_action_vale = F.relu(state_value + action_value)
        state_action_value = self.q(state_action_vale)

        return state_action_value


class FeedForwardActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, device=device, lr=1e-4):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

        self.mu = nn.Linear(hidden_dim, out_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        state_value = F.relu(state_value)
        state_value = self.mu(state_value)
        state_value = F.tanh(state_value)

        return state_value


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * (self.dt ** 0.5) * \
            torch.randn(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(
            self.mu)


class DDPG:
    def __init__(self, env,
                 actor, target_actor,
                 critic, target_critic,
                 gamma=0.99, tau=0.05):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.gamma = gamma
        self.tau = tau

        self.noise = OUActionNoise(torch.zeros(size=env.action_space.shape))
        self.update_network_parameters(tau=1)

    def get_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(state.to(self.actor.device))
            mu_noisy = mu + self.noise().to(dtype=mu.dtype, device=mu.device)
        self.actor.train()
        return mu_noisy.cpu()

    def learn(self, n_epochs=10, timesteps_per_epoch=1000, batch_size=64):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}")
            states, actions, new_states, rewards, dones, batches = \
                self.collect_rollouts(timesteps_per_epoch, batch_size)
            
            mean_reward = rewards.sum() / dones.to(torch.int).sum()
            print(f"Avg reward/ep {mean_reward.item():.4f}")
            print(f"Avg reward/timestep {rewards.mean().item():.4f}")
                
            for batch in batches:
                states_batch = states[batch].clone().to(self.actor.device)
                actions_batch = actions[batch].clone().to(self.actor.device)
                new_states_batch = new_states[batch].clone().to(self.actor.device)
                rewards_batch = rewards[batch].clone().to(self.actor.device)
                dones_batch = dones[batch].clone().to(self.actor.device)
                
                self.target_actor.eval()
                self.target_critic.eval()
                self.critic.eval()
                
                with torch.no_grad():
                    target_actions = self.target_actor(new_states_batch)
                    target_values = self.target_critic(new_states_batch, target_actions)
                values = self.critic(states_batch, actions_batch)
                
                target_batch = torch.empty(
                    size=(len(rewards_batch), 1), dtype=torch.float32, device=self.actor.device)
                for i in range(len(rewards_batch)):
                    target_batch[i] = rewards_batch[i] + \
                        self.gamma * target_values[i] * (1 - int(dones_batch[i]))
                
                self.critic.train()
                critic_loss = F.mse_loss(values, target_batch)
                critic_loss.backward()
                self.critic.optimizer.step()
                self.critic.optimizer.zero_grad()
                
                self.actor.train()
                new_actions = self.actor(states_batch)
                actor_loss = torch.mean(-1 * self.critic(states_batch, new_actions))
                actor_loss.backward()
                self.actor.optimizer.step()
                self.actor.optimizer.zero_grad()
                
                self.update_network_parameters()
                
    def collect_rollouts(self, timesteps, batch_size):
        states = []
        actions = []
        new_states = []
        rewards = []
        dones = []
        
        cur_timestep = 0
        while cur_timestep < timesteps:
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.get_action(torch.tensor(state).unsqueeze(0))
                new_state, reward, done, _, _ = self.env.step(action.squeeze(0).numpy())
                states.append(state)
                actions.append(action.squeeze(0))
                new_states.append(new_state)
                rewards.append(reward)
                dones.append(done)
                cur_timestep += 1
                state = new_state

        indices = torch.randperm(len(states))[:timesteps]
        batches = [indices[start_idx:start_idx+batch_size] 
                   for start_idx in range(0, timesteps, batch_size)]
        
        return (torch.tensor(np.array(states)),
                torch.tensor(np.array(actions)),
                torch.tensor(np.array(new_states)),
                torch.tensor(rewards),
                torch.tensor(dones),
                batches)
                
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1 - tau) * target_critic_state_dict[name].clone() 
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_state_dict[name].clone() 
                
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
