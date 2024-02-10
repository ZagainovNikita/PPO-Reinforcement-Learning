import torch
from torch.nn import functional as F
import numpy as np


class PPO:
    def __init__(self, env, actor, critic,
                 gamma=0.95, clip=0.2, batch_size=64):
        self.env = env
        self.actor = actor
        self.critic = critic

        self.gamma = gamma
        self.clip = clip
        self.batch_size = batch_size

    def learn(self, n_epochs=10, timesteps_per_epoch=1000, updates_per_epoch=5):
        for epoch in range(n_epochs):

            (states, old_probs, vals,
             actions, rewards, dones, batches) = self.collect_rollouts(1000, shuffle=True)

            avg_ep_reward = np.sum(rewards) / np.sum(dones.astype(int))
            print(avg_ep_reward)
            
            rewards_to_go = np.zeros_like(rewards)
            reward_to_go = 0
            for i, rew in enumerate(rewards):
                reward_to_go = rew + reward_to_go * self.gamma
                if dones[i]:
                    reward_to_go = 0
                rewards_to_go[i] = reward_to_go

            advantages = rewards_to_go - vals

            for update in range(updates_per_epoch):
                losses = torch.zeros(len(batches), dtype=torch.float32, device=self.actor.device)
                for i, batch in enumerate(batches):
                    states_batch = torch.tensor(
                        states[batch], dtype=torch.float32, device=self.actor.device)
                    advantages_batch = torch.tensor(
                        advantages[batch], dtype=torch.float32, device=self.actor.device)
                    rwtg_batch = torch.tensor(
                        rewards_to_go[batch], dtype=torch.float32, device=self.actor.device)
                    actions_batch = torch.tensor(
                        actions[batch], dtype=torch.float32, device=self.actor.device)
                    old_probs_batch = torch.tensor(
                        old_probs[batch], dtype=torch.float32, device=self.actor.device)
                    
                    dist = self.actor.get_distribution(states_batch)
                    new_probs_batch = dist.log_prob(actions_batch)
                    vals = self.critic(states_batch)
                    
                    ratios = torch.exp(new_probs_batch - old_probs_batch)
                    surrogate_loss1 = advantages_batch * ratios
                    surrogate_loss2 = torch.clamp(advantages_batch * ratios, 
                                                  1 - self.clip, 1 + self.clip)
                    
                    actor_loss = -1 * torch.min(surrogate_loss1, surrogate_loss2).mean()
                    critic_loss = F.mse_loss(vals, rwtg_batch.view(vals.shape))
                    losses[i] = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                
                total_loss = losses.mean()
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                    
    def get_action(self, state):
        state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=self.actor.device)
        
        with torch.no_grad():
            dist = self.actor.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            val = self.critic(state)
        
        return action.squeeze().item(), log_prob.squeeze().item(), val.squeeze().item() 
    
    def collect_rollouts(self, n_timesteps, shuffle=True):
        states = []
        probs = []
        vals = []
        actions = []
        rewards = []
        dones = []
        
        cur_timestep = 0

        while cur_timestep < n_timesteps:
            state, _ = self.env.reset()
            done = False

            while not done:
                action, log_prob, val = self.get_action(state)
                prev_state = state.copy()
                state, reward, done, _, _ = self.env.step(action)
                
                states.append(prev_state)
                probs.append(log_prob)
                vals.append(val)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                                
                cur_timestep += 1
        
        n_samples = cur_timestep
        batch_starts = np.arange(0, n_samples, self.batch_size)
        indices = np.arange(0, n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = [indices[start:start+self.batch_size] for start in batch_starts]
        
        return (np.array(states), np.array(probs), np.array(vals),
                np.array(actions), np.array(rewards), np.array(dones),
                batches)
        