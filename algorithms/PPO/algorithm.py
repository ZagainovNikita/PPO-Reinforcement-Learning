import torch
from torch.nn import functional as F
import numpy as np
from algorithms.ppo.ppo_memory import PPOMemory


class PPO:
    def __init__(
        self, env, actor, critic, 
        gamma=0.99, lambda_disc=0.95, clip=0.2,
        batch_size=64
        ):
        self.env = env
        self.actor = actor
        self.critic = critic
        
        self.gamma = gamma
        self.lambda_disc = lambda_disc
        self.clip = clip
        self.memory = PPOMemory(batch_size)

    def learn(self, n_epochs=10, timesteps_per_epoch=1000):
        for epoch in range(n_epochs):
            self.collect_rollouts(timesteps_per_epoch)
            
            (states, old_probs, vals,
             actions, rewards, dones, batches) = self.memory.generate_batches()
            
            avg_ep_reward = np.sum(rewards) / np.sum(dones.astype(int))
            print(avg_ep_reward)
            
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
                
            # running_rews = np.zeros(len(rewards), dtype=np.float32)
            # discount_factor = 0.95
            # cur_rew = 0
            # for i, rew in enumerate(rewards):
            #     cur_rew = rew + discount_factor * cur_rew
            #     if dones[i - 1]:
            #         cur_rew = rew
            #     running_rews[i] = cur_rew
            
            # advantages = running_rews - vals
            # advantages = (advantages - advantages.mean()) / advantages.std()

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
                # critic_target = torch.tensor(running_rews[batch], dtype=torch.float32, device=self.actor.device)
                critic_loss = F.mse_loss(critic_vals, critic_target.view(critic_vals.shape))
                
                total_loss = actor_loss + critic_loss * 0.5
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            self.memory.clear_memory()
        
    def get_action(self, state):
        state = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, device=self.actor.device)
        
        with torch.no_grad():
            dist = self.actor.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            val = self.critic(state)
        
        return action.squeeze().item(), log_prob.squeeze().item(), val.squeeze().item() 
        
    def collect_rollouts(self, n_timesteps):
        cur_timestep = 0

        while cur_timestep < n_timesteps:
            state, _ = self.env.reset()
            done = False

            while not done:
                action, log_prob, val = self.get_action(state)
                prev_state = state.copy()
                state, reward, done, _, _ = self.env.step(action)
                self.memory.store(prev_state, log_prob, val, action, reward, done)

                cur_timestep += 1
            