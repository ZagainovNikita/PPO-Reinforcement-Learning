import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from gymnasium import Env


class PPO:
    def __init__(
        self,
        env: Env,
        model_class: nn.Module,
        device: torch.DeviceObjType = torch.device("cuda:0"),
        lr: float = 25e-6,
        batch_size: int = 6000,
        episode_length: int = 2000,
        gamma: float = 0.95,
        n_updates: int = 10,
        clip: float = 0.2
    ):
        self.env = env
        self.model_class = model_class
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.gamma = gamma
        self.n_updates = n_updates
        self.clip = clip

        self.act_dim = env.action_space.n
        self.obs_dim = env.observation_space.shape[0]

        self.actor = self.model_class(self.obs_dim, self.act_dim).to(device)
        self.critic = self.model_class(self.obs_dim, 1).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(
            size=(self.act_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var, device=self.device)

    def learn(self, total_timesteps):
        cur_timestep = 0
        n_iterations = 0

        while cur_timestep < total_timesteps:
            (batch_obs, batch_acts, batch_log_probs,
             batch_rews, batch_lens, batch_rtgs) = self.run_env()
            
            n_iterations += 1
            cur_timestep += np.sum(batch_lens)
            average_episode_length = np.mean(batch_lens)
            
            running_actor_loss = 0
            running_critic_loss = 0
            
            V = self.predict_rew(batch_obs)
            A = batch_rews - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-9) 
            
            for n_update in range(self.n_updates):
                
                V = self.predict_rew(batch_obs)
                cur_log_probs = self.get_actions_log_probs(batch_obs, batch_acts)
                prob_ratios = torch.exp(cur_log_probs - batch_log_probs)
                
                surrogate_loss1 = A * prob_ratios
                surrogate_loss2 = A * torch.clamp(prob_ratios, 1 - self.clip, 1 + self.clip)
                
                actor_loss = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
                critic_loss = F.mse_loss(V, batch_rtgs)
                
                running_actor_loss += actor_loss.item()
                running_critic_loss += critic_loss.item()
                
                
                
                
                
    def run_env(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        cur_timestep = 0
        
        while cur_timestep < self.batch_size:
            episode_rews = []
            obs, _ = self.env.reset()
            done = False

            for i in range(self.episode_length):
                cur_timestep += 1

                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(torch.argmax(action))

                batch_log_probs.append(log_prob)
                episode_rews.append(rew)
                batch_acts.append(action)

                if done:
                    break
            batch_lens.append(i + 1)
            batch_rews.append(episode_rews)

        batch_rtgs = torch.tensor(
            self.compute_rtgs(batch_rews), device=self.device)
        batch_obs = torch.tensor(np.array(batch_obs), device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), device=self.device)
        
        return (batch_obs, batch_acts, batch_log_probs,
                batch_rews, batch_lens, batch_rtgs)

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.cpu().detach().numpy(), log_prob.detach()

    def get_actions_log_probs(self, batch_obs, batch_acts):
        mean = self.critic(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        return log_probs
    
    def predict_rew(self, batch_obs):
        V = self.critic(batch_obs)
        
        return V

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for episode_rews in reversed(batch_rews):
            running_reward = 0.0

            for rew in reversed(episode_rews):
                running_reward = rew + self.gamma * running_reward
                batch_rtgs.append(running_reward)

        batch_rtgs.reverse()
        return batch_rtgs
