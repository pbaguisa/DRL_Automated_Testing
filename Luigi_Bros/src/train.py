"""
Training script for Super Luigi Bros DRL agents.
Supports PPO and A2C algorithms with explorer/speedrunner personas.
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from envs.game.luigi_env import LuigiEnv

ALGOS = {"ppo": PPO, "a2c": A2C}


def make_env(reward_mode="explorer", seed=7):
    """Create and wrap the Luigi environment"""
    env = LuigiEnv(reward_mode=reward_mode, seed=seed)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO or A2C agent on Super Luigi Bros."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "a2c"],
        default="ppo",
        help="RL algorithm to use (ppo or a2c)"
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["explorer", "speedrunner"],
        default="explorer",
        help="Reward shaping mode (explorer or speedrunner)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    # Construct clean folder names
    run_name = f"{args.algo}_luigi_{args.reward_mode}_seed{args.seed}"
    log_path = os.path.join(args.logdir, run_name)
    model_path = os.path.join(args.modeldir, run_name)

    print(f"\n🎮 Training {args.algo.upper()} on Super Luigi Bros")
    print(f"   Reward Mode: {args.reward_mode}")
    print(f"   Seed: {args.seed}")
    print(f"   Timesteps: {args.timesteps:,}\n")

    # Create environment
    env = make_env(reward_mode=args.reward_mode, seed=args.seed)

    # Initialize algorithm
    if args.algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=2048,
            batch_size=256,
            gamma=0.999,
            gae_lambda=0.98,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=log_path,
            seed=args.seed,
        )
    elif args.algo == "a2c":
        model = A2C(
            policy="MlpPolicy",
            env=env,
            n_steps=1024,
            gamma=0.999,
            gae_lambda=0.98,
            learning_rate=7e-4,
            verbose=1,
            tensorboard_log=log_path,
            seed=args.seed,
        )

    # Configure logger
    new_logger = configure(log_path, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Train
    print("🚀 Starting training...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save model
    model.save(model_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {model_path}.zip")
    print(f"   TensorBoard logs: {log_path}")
    print(f"\n📊 View training progress with:")
    print(f"   tensorboard --logdir {args.logdir}\n")

    env.close()


if __name__ == "__main__":
    main()
