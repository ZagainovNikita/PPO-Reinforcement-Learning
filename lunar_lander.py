from algorithms.ppo_continuous import PPO, FeedForwardNN
import gymnasium as gym
import torch


env = gym.make("LunarLanderContinuous-v2")
actor = FeedForwardNN(8, 2)
critic = FeedForwardNN(8, 1)
model = PPO(env, actor, critic)

model.learn(100, 4096, batch_size=1024, force_stop=6000)