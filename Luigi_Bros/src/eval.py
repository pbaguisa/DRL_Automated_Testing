"""
Evaluation script for trained Super Luigi Bros agents.
Collects detailed performance metrics and saves to CSV.
"""
import argparse
import csv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO, A2C

from envs.game.luigi_env import LuigiEnv


def run_episode(model, env, render=False) -> dict:
    """Run one episode and collect metrics"""
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
        
    # Extract metrics from final info
    metrics = {
        "reward": ep_reward,
        "deaths": last_info.get("deaths", 0),
        "completions": last_info.get("completions", 0),
        "frames_alive": last_info.get("frames_alive", 0),
        "coins_collected": last_info.get("coins_collected", 0),
        "blocks_hit": last_info.get("blocks_hit", 0),
        "mushroom_blocks_hit": last_info.get("mushroom_blocks_hit", 0),
        "blocks_hit_ratio": last_info.get("blocks_hit_ratio", 0.0),
        "max_x_reached": last_info.get("max_x_reached", 0.0),
        "distance_traveled": last_info.get("distance_traveled", 0.0),
        "jumps": last_info.get("jumps", 0),
        "reward_mode": last_info.get("reward_mode", "unknown"),
        "completed": int(last_info.get("completions", 0) > 0),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Super Luigi Bros agent"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to .zip model (e.g., models/ppo_mario_explorer_seed7)"
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="explorer",
        choices=["explorer", "speedrunner"],
        help="Reward mode to evaluate with"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        type=int,
        default=0,
        help="Render episodes (1) or not (0)"
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="logs/eval_metrics.csv",
        help="Path to save evaluation metrics CSV"
    )
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    # Load model (try PPO first, fall back to A2C)
    print(f"\n📦 Loading model from {args.model_path}.zip")
    try:
        model = PPO.load(args.model_path)
        algo = "PPO"
    except Exception:
        model = A2C.load(args.model_path)
        algo = "A2C"
    
    print(f"   Algorithm: {algo}")
    print(f"   Reward Mode: {args.reward_mode}")
    print(f"   Episodes: {args.episodes}\n")

    # Create environment
    render_mode = "human" if args.render else None
    env = LuigiEnv(reward_mode=args.reward_mode, render_mode=render_mode)

    # Run episodes
    rows = []
    print("🎮 Running evaluation episodes...")
    for ep in range(1, args.episodes + 1):
        print(f"   Episode {ep}/{args.episodes}...", end=" ")
        metrics = run_episode(model, env, render=bool(args.render))
        metrics["episode"] = ep
        rows.append(metrics)
        
        # Print episode result
        if metrics["completed"]:
            print(f"✅ Completed! Reward: {metrics['reward']:.1f}")
        else:
            print(f"💀 Died. Reward: {metrics['reward']:.1f}")

    # Calculate summary statistics
    rewards = [r["reward"] for r in rows]
    completions = [r["completed"] for r in rows]
    distances = [r["distance_traveled"] for r in rows]
    coins = [r["coins_collected"] for r in rows]
    blocks = [r["blocks_hit"] for r in rows]
    
    print(f"\n📊 Summary Statistics ({args.episodes} episodes):")
    print(f"   Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Completion Rate: {np.mean(completions)*100:.1f}%")
    print(f"   Mean Distance: {np.mean(distances):.1f} pixels")
    print(f"   Mean Coins: {np.mean(coins):.1f}")
    print(f"   Mean Blocks Hit: {np.mean(blocks):.1f}")
    
    if args.reward_mode == "explorer":
        blocks_ratio = [r["blocks_hit_ratio"] for r in rows]
        print(f"   Mean Block Hit Ratio: {np.mean(blocks_ratio)*100:.1f}%")

    # Save to CSV
    fieldnames = [
        "episode", "reward", "deaths", "completions", "completed",
        "frames_alive", "coins_collected", "blocks_hit", "mushroom_blocks_hit",
        "blocks_hit_ratio", "max_x_reached", "distance_traveled", "jumps",
        "reward_mode"
    ]
    
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n💾 Saved detailed metrics to: {args.csv_out}\n")

    env.close()


if __name__ == "__main__":
    main()
