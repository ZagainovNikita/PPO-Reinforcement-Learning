from model import ConvolutionalNN
import gymnasium as gym
from ppo import PPO


def main():
    env = gym.make("ALE/Assault-v5")
    model = PPO(
        env=env,
        policy_class=ConvolutionalNN,
    )

    model.learn(400000, log_dir="./logs/")


if __name__ == "__main__":
    main()