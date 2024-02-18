import gymnasium as gym
from algorithms.ddpg_continuous import DDPG, FeedForwardCritic, FeedForwardActor


env = gym.make("LunarLanderContinuous-v2")
actor = FeedForwardActor(*env.observation_space.shape, *env.action_space.shape)
critic = FeedForwardCritic(*env.observation_space.shape, *env.action_space.shape)
target_actor = FeedForwardActor(*env.observation_space.shape, *env.action_space.shape)
target_critic = FeedForwardCritic(*env.observation_space.shape, *env.action_space.shape)

model = DDPG(env, actor, target_actor, critic, target_critic)
model.learn(50, 4000, 256)