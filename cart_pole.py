import gymnasium as gym
from algorithms.ppo.gae import PPO
from models.feed_forward import FeedForwardNN


env = gym.make("CartPole-v1")
actor = FeedForwardNN(env.observation_space.shape[0], env.action_space.n, lr=1e-4)
critic = FeedForwardNN(env.observation_space.shape[0], 1, lr=1e-4)

ppo = PPO(env, actor, critic, batch_size=64)

ppo.learn(30, 2500)