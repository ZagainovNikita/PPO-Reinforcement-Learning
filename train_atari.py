from model import ConvolutionalNN
import gymnasium as gym
from ppo import PPO


def main():
    env = gym.make("ALE/Assault-v5")
    model = PPO(
        env=env,
        policy_class=ConvolutionalNN,
        gamma=0.95,
        n_updates=5,
        episodes_per_update=2000,
        episode_length=1000,
        batch_size=32,
        device="cuda",
        lr=1e-3
    )

    def save_callback(
            n_iter, n_timestep, mean_episode_reward,
            mean_actor_loss, mean_critic_loss, mean_episode_length):
        if n_iter % 4 == 0:
            return True
        return False

    model.learn(400000, save_callback=save_callback, log_dir="./logs/")


if __name__ == "__main__":
    main()