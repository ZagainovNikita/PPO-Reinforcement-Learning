from model import FeedForward
import gymnasium as gym
from ppo import PPO

env = gym.make("CartPole-v1")
model = PPO(
    env=env,
    policy_class=FeedForward
)

model.learn(1)