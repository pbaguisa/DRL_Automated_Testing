import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from stable_baselines3 import PPO
from envs.game.bubble_game_env import BubbleGameEnv

def make_env(render_mode=None, reward_mode="survival", seed=7):
    env = BubbleGameEnv()
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--reward_mode", type=str, default="survival", choices=["survival", "coverage"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--modeldir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    env = make_env(reward_mode=args.reward_mode, seed=args.seed)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir, seed=args.seed)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model.save(os.path.join(args.modeldir, "ppo_bubble_game"))
    print(f"Saved model to {args.modeldir}/ppo_bubble_game")

if __name__ == "__main__":
    main()
