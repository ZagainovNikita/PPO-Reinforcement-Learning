import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, device=device, lr=1e-4):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        
        self.device = device
        self.log_std = torch.full(size=(1, out_dim), fill_value=0.5, device=self.device)
        self.to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        outputs = self.model(state)
        
        return outputs
    
    def get_distribution(self, state):
        action_logits = self.forward(state)
        dist = Normal(action_logits, self.log_std.exp())
        
        return dist

class PPO:
    def __init__(self, env, actor, critic,
                 clip=0.1, gamma=0.95):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.clip = clip
    
    def learn(self, n_epochs=10, 
              timesteps_per_epoch=1000, updates_per_epoch=5, 
              batch_size=64, force_stop=6000):
        for epoch in range(n_epochs):
            states, actions, probs, rewards, dones, batches = \
                self.collect_rollouts(timesteps=timesteps_per_epoch, 
                                      batch_size=batch_size, force_stop=force_stop)
                
            mean_reward = torch.sum(rewards) / torch.sum(dones.to(torch.int))
                
            rwtg = torch.zeros_like(rewards, dtype=torch.float32)
            rew_to_go = 0
            for t in reversed(range(len(rewards))):
                rew_to_go = rewards[t] + 0.95 * rew_to_go * (1 - int(dones[t]))
                rwtg[t] = rew_to_go
                
            for update in range(updates_per_epoch):
                for batch in batches:                    
                    states_batch = states[batch].clone().to(self.actor.device)
                    actions_batch = actions[batch].clone().to(self.actor.device)
                    old_probs_batch = probs[batch].clone().to(self.actor.device)
                    rwtg_batch = rwtg[batch].clone().to(self.actor.device)
                    
                    new_dist = self.actor.get_distribution(states_batch)
                    new_probs_batch = new_dist.log_prob(actions_batch)
                    values_batch = self.critic(states_batch)
                    
                    advandages_batch = rwtg_batch.unsqueeze(-1) - values_batch
                    ratios = torch.exp(new_probs_batch - old_probs_batch)
                    
                    surrogate_loss1 = ratios * advandages_batch
                    surrogate_loss2 = torch.clip(ratios * advandages_batch, 
                                                 1 - self.clip, 1 + self.clip)
                    
                    actor_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
                    critic_loss = F.mse_loss(values_batch, rwtg_batch.view(values_batch.shape))
                    total_loss = actor_loss + critic_loss
                    
                    total_loss.backward()
                    
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
            

            print(f"Epoch {epoch + 1}/{n_epochs}".center(50, "-"))
            print(f"Avg episode reward {mean_reward.item():.4f}")
            print(f"Avg timestep reward {rewards.mean().item():.4f}")
            print(f"Max reward {rewards.max().item():.4f}")
            print("-" * 50)

    def collect_rollouts(self, timesteps, batch_size, force_stop):
        states = []
        actions = []
        probs = []
        rewards = []
        dones = []
        
        cur_t = 0
        while cur_t < timesteps:
            done = False
            n_steps = 0
            state, _ = self.env.reset()
            while not done:
                states.append(state)
                action, prob = self.get_action(state)
                action = action.squeeze(0)
                prob = prob.squeeze(0)
                actions.append(action)
                probs.append(prob)
    
                state, reward, done, _, _ = self.env.step(action.numpy())
                dones.append(done)
                rewards.append(reward)
                if n_steps >= force_stop:
                    rewards[-1] = -200
                    break
                cur_t += 1
                n_steps += 1
                
            
        indices = torch.randperm(len(states))[:timesteps]
        batches = [indices[start_idx:start_idx+batch_size] 
                   for start_idx in range(0, timesteps, batch_size)]
        
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(probs), dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones),
                batches)
            
    
    def get_action(self, state):
        with torch.no_grad():
            dist = self.actor.get_distribution(
                torch.tensor(state, device=self.actor.device))

            action = dist.sample()
            prob = dist.log_prob(action)
            
        return action.cpu(), prob.cpu()