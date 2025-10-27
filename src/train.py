import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO, A2C
from envs.game.bubble_game_env import BubbleGameEnv

# YAML Loader
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        raise ImportError("Config found but PyYAML not installed. Run: pip install pyyaml")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}
# -------------------------------------------------------------------

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

    # CLI stablization
    p.add_argument("--algo_cfg", type=str, default=None)
    p.add_argument("--seed_cfg", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    # --- pull hyperparams/seed from configs (with fallback to current defaults) ---
    algo_cfg_path = args.algo_cfg or os.path.join("configs", "algos", f"{args.algo}.yaml")
    seed_cfg_path = args.seed_cfg or os.path.join("configs", "seeds", f"seed{args.seed}.yaml")
    algo_cfg = _load_yaml(algo_cfg_path)
    seed_cfg = _load_yaml(seed_cfg_path)
    seed_val = int(seed_cfg.get("seed", args.seed))
    # -----------------------------------------------------------------------------------

    # Construct clean folder names
    run_name = f"{args.algo}_bubble_{args.reward_mode}_seed{seed_val}"
    log_path = os.path.join(args.logdir, run_name)
    model_path = os.path.join(args.modeldir, run_name)

    print(f"\n Training {args.algo.upper()} on BubbleGameEnv ({args.reward_mode} mode, seed={seed_val})\n")

    env = BubbleGameEnv(reward_mode=args.reward_mode, seed=seed_val)

    if args.algo == "ppo":
        # defaults preserved if keys not present in YAML
        model = PPO(
            algo_cfg.get("policy", "MlpPolicy"),
            env,
            n_steps=algo_cfg.get("n_steps", 2048),
            batch_size=algo_cfg.get("batch_size", 256),
            gamma=algo_cfg.get("gamma", 0.999),
            gae_lambda=algo_cfg.get("gae_lambda", 0.98),
            verbose=algo_cfg.get("verbose", 1),
            tensorboard_log=log_path,
            seed=seed_val,
            learning_rate=algo_cfg.get("learning_rate", 1e-4),
        )
    elif args.algo == "a2c":
        model = A2C(
            algo_cfg.get("policy", "MlpPolicy"),
            env,
            verbose=algo_cfg.get("verbose", 1),
            tensorboard_log=log_path,
            seed=seed_val,
            learning_rate=algo_cfg.get("learning_rate", 6e-4),
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(model_path)

    print(f"\n Training complete! Model saved to: {model_path}.zip")
    print(f"TensorBoard logs saved to: {log_path}\n")
    print("View training progress with:")
    print(f"  tensorboard --logdir {args.logdir}\n")

if __name__ == "__main__":
    main()
