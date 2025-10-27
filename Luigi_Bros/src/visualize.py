"""
Visualization script for trained Super Luigi Bros agents.
Watch the agent play in real-time with rendering.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, A2C

from envs.game.luigi_env import LuigiEnv


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained Super Luigi Bros agent"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_luigi_explorer_seed7",
        help="Path to .zip model"
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="explorer",
        choices=["explorer", "speedrunner"],
        help="Reward mode for environment"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for rendering"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (deterministic=False)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (will show all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (useful to replay specific episode)"
    )
    args = parser.parse_args()

    # Load model
    print(f"\n📦 Loading model from {args.model_path}.zip")
    try:
        model = PPO.load(args.model_path)
        algo = "PPO"
    except Exception:
        model = A2C.load(args.model_path)
        algo = "A2C"
    
    print(f"   Algorithm: {algo}")
    print(f"   Reward Mode: {args.reward_mode}")
    print(f"   FPS: {args.fps}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Seed: {args.seed if args.seed is not None else 'Random'}")
    print(f"   Deterministic: {not args.stochastic}\n")

    # Create environment with rendering
    env = LuigiEnv(
        render_mode="human",
        reward_mode=args.reward_mode
    )
    env.metadata["render_fps"] = args.fps

    print("🎮 Starting visualization (close window to exit)...\n")

    # Run episodes
    for ep in range(1, args.episodes + 1):
        # Use seed if provided (for reproducing specific episodes)
        if args.seed is not None:
            obs, info = env.reset(seed=args.seed + ep - 1)
        else:
            obs, info = env.reset()
            
        done, trunc = False, False
        total_reward = 0.0
        
        print(f"Episode {ep}/{args.episodes}:")
        
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

        # Print episode results
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Completed: {'✅ Yes' if info.get('completions', 0) > 0 else '❌ No'}")
        print(f"   Coins: {info.get('coins_collected', 0)}")
        print(f"   Blocks Hit: {info.get('blocks_hit', 0)}")
        print(f"   Distance: {info.get('distance_traveled', 0):.1f} pixels")
        print(f"   Frames: {info.get('frames_alive', 0)}\n")

    env.close()


if __name__ == "__main__":
    main()
