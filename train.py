from model import FeedForward
import gymnasium as gym
from ppo import PPO

env = gym.make("CartPole-v1")
model = PPO(
    env=env,
    policy_class=FeedForward,
    gamma=0.95,
    n_updates=5,
    batch_size=4800,
    episode_length=1600,
    device="cpu",
    lr=1e-3
)

model.learn(100000)