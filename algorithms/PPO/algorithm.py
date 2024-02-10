import torch
from torch.nn import functinoal as F
import numpy as np
from ppo_memory import PPOMemory


class PPO:
    def __init__(
        self, env, actor, critic, 
        gamma=0.99, lambda_disc=0.95, clip=0.2,
        batch_size=64, n_epochs=10
        ):
        self.env = env
        self.actor = actor
        self.critic = critic
        
        self.gamma = gamma
        self.lamda_disc = lambda_disc
        self.clip = clip
        self.n_epochs = n_epochs
        self.memory = PPOMemory(batch_size)

    def learn(self):
        for epoch in range(self.n_epochs):
            (states, old_probs, rewards, vals,
             actions, dones, batches) = self.memory.generate_batches()

            advantages = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount_factor = 1
                cur_adv = 0
                for i in range(t, len(rewards) - 1):
                    cur_adv += discount_factor * \
                        (rewards[i] + self.gamma * vals[i + 1]
                         * (1 - int(dones[i])) - vals[i])
                    discount_factor *= self.gamma * self.lambda_disc
                advantages[t] = cur_adv

            for batch in batches:
                states_batch = torch.tensor(
                    states[batch], dtype=torch.float32, device=self.actor.device)
                old_probs_batch = torch.tensor(
                    old_probs[batch], dtype=torch.float32, device=self.actor.device)
                actions_batch = torch.tensor(
                    actions[batch], dtype=torch.int64, device=self.actor.device)
                advantages_batch = torch.tensor(
                    advantages[batch], dtype=torch.float32, device=self.actor.device)
                vals_batch = torch.tensor(
                    vals[batch], dtype=torch.float32, device=self.actor.device)

                dist = self.actor.get_distribution(states_batch)
                critic_vals = self.critic(states_batch).squeeze(0)

                new_probs_batch = dist.log_prob(actions_batch)

                probs_ratio = torch.exp(new_probs_batch - old_probs_batch)
                surrogate_loss1 = probs_ratio * advantages_batch
                surrogate_loss2 = torch.clip(probs_ratio * advantages_batch, 1 - self.clip, 1 + self.clip)
                
                actor_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()
                
                critic_target = advantages_batch + vals_batch
                critic_loss = F.mse_loss(critic_vals, critic_target)
                
                total_loss = actor_loss + critic_loss * 0.5
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
        self.memory.clear_memory()
        
    def collect_rollouts(self, n_timesteps):
        state, _ = self.env.reset()
                
