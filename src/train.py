import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, A2C
from envs.game.bubble_game_env import BubbleGameEnv

ALGOS = {"ppo": PPO, "a2c": A2C}

def main():
    p = argparse.ArgumentParser(description="Train PPO or A2C agent on BubbleGameEnv.")
    p.add_argument("--algo", type=str, choices=["ppo", "a2c"], default="ppo",
                   help="RL algorithm to use (ppo or a2c)")
    p.add_argument("--reward_mode", type=str, choices=["survivor", "speedrunner"],
                   default="survivor", help="Reward shaping mode")
    p.add_argument("--timesteps", type=int, default=200_000,
                   help="Number of training timesteps")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--logdir", type=str, default="logs", help="TensorBoard log directory")
    p.add_argument("--modeldir", type=str, default="models", help="Directory to save models")
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    # Construct clean folder names
    run_name = f"{args.algo}_bubble_{args.reward_mode}_seed{args.seed}"
    log_path = os.path.join(args.logdir, run_name)
    model_path = os.path.join(args.modeldir, run_name)

    print(f"\n Training {args.algo.upper()} on BubbleGameEnv ({args.reward_mode} mode, seed={args.seed})\n")

    env = BubbleGameEnv(reward_mode=args.reward_mode, seed=args.seed)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=256,
            gamma=0.999,
            gae_lambda=0.98,
            verbose=1,
            tensorboard_log=log_path,
            seed=args.seed,
            learning_rate=1e-4
        )
    elif args.algo == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
            seed=args.seed,
            learning_rate=6e-4 
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(model_path)

    print(f"\n Training complete! Model saved to: {model_path}.zip")
    print(f"TensorBoard logs saved to: {log_path}\n")
    print("View training progress with:")
    print(f"  tensorboard --logdir {args.logdir}\n")

if __name__ == "__main__":
    main()
