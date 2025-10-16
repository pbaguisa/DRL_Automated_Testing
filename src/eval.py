import argparse
import numpy as np
import csv
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.game.bubble_game_env import BubbleGameEnv

def run_episode(model, env, reward_mode="survival", render=False):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_reward += r

    return ep_reward

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/ppo_bubble_game")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="survival", choices=["survival", "coverage"])
    p.add_argument("--csv_out", type=str, default="logs/eval_metrics.csv")
    args = p.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    model = PPO.load(args.model_path)
    env = BubbleGameEnv()

    rows = []
    for ep in range(1, args.episodes + 1):
        ep_reward = run_episode(model, env, reward_mode=args.reward_mode, render=bool(args.render))
        rows.append({"episode": ep, "reward": ep_reward})

    # Save metrics to CSV
    fieldnames = ["episode", "reward"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Saved evaluation metrics to {args.csv_out}")

if __name__ == "__main__":
    main()
