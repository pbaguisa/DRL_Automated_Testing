import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from envs.game.bubble_game_env import BubbleGameEnv

def make_env(reward_mode: str, seed: int):
    # No render during training for speed; env handles its own seeding
    env = BubbleGameEnv(reward_mode=reward_mode, seed=seed)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=float, default=200_000)
    parser.add_argument("--reward_mode", type=str, default="survivor",
                        choices=["survivor", "speedrunner"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--modeldir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    env = make_env(reward_mode=args.reward_mode, seed=args.seed)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir, seed=args.seed)

    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    model_path = os.path.join(args.modeldir, f"ppo_bubble_{args.reward_mode}")
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")

if __name__ == "__main__":
    main()
