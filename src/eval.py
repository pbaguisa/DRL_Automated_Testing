import argparse
import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO

from envs.game.bubble_game_env import BubbleGameEnv

def run_episode(model, env) -> float:
    obs, info = env.reset()
    done = False
    trunc = False
    ep_reward = 0.0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_reward += r
    return float(ep_reward)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to .zip model, e.g., models/ppo_bubble_survivor")
    p.add_argument("--reward_mode", type=str, default="survivor",
                   choices=["survivor", "speedrunner"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--csv_out", type=str, default="logs/eval_metrics.csv")
    args = p.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    model = PPO.load(args.model_path)
    env = BubbleGameEnv(reward_mode=args.reward_mode)

    rows = []
    scores = []
    for ep in range(1, args.episodes + 1):
        ep_reward = run_episode(model, env)
        rows.append({"episode": ep, "reward": ep_reward})
        scores.append(ep_reward)

    # Save metrics to CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reward"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved evaluation metrics to {args.csv_out}")
    print(f"[{args.reward_mode}] mean reward over {args.episodes} eps: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

if __name__ == "__main__":
    main()
