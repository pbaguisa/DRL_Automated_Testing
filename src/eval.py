import argparse
import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO, A2C  # allow either algo

from envs.game.bubble_game_env import BubbleGameEnv

def run_episode(model, env) -> dict:
    obs, info = env.reset()
    done = False
    trunc = False
    ep_reward = 0.0
    last_info = {}
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_reward += float(r)
        last_info = info  
    metrics = {
        "reward": ep_reward,
        "shots": last_info.get("shots", 0),
        "pops": last_info.get("pops", 0),
        "deaths": last_info.get("deaths", int(done and not trunc)),
        "frames_alive": last_info.get("frames_alive", 0),
        "wall_ratio": last_info.get("wall_ratio", 0.0),
        "accuracy": last_info.get("accuracy", 0.0),
        "avg_dist": last_info.get("avg_dist", 0.0),
        "reward_mode": last_info.get("reward_mode", getattr(env, "reward_mode", "unknown")),
    }
    return metrics

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

    # Load PPO by default; fall back to A2C if needed
    Loader = PPO
    try:
        model = Loader.load(args.model_path)
    except Exception:
        Loader = A2C
        model = Loader.load(args.model_path)

    env = BubbleGameEnv(reward_mode=args.reward_mode)

    rows = []
    rewards = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, env)
        rows.append({"episode": ep, **metrics})
        rewards.append(metrics["reward"])

    # to CSV
    fieldnames = ["episode", "reward", "shots", "pops", "deaths", "frames_alive",
                  "wall_ratio", "accuracy", "avg_dist", "reward_mode"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved evaluation metrics to {args.csv_out}")
    print(f"[{args.reward_mode}] mean reward over {args.episodes} eps: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

if __name__ == "__main__":
    main()
