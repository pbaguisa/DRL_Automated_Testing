import argparse
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.game.bubble_game_env import BubbleGameEnv

# YAML loader
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_bubble_survivor")
    parser.add_argument("--reward_mode", type=str, default="survivor",
                        choices=["survivor", "speedrunner"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use deterministic=False to see more varied actions")
    # optional seed cfg
    parser.add_argument("--seed_cfg", type=str, default=None)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    seed_cfg_path = args.seed_cfg or os.path.join("configs", "seeds", "seed0.yaml")
    seed_cfg = _load_yaml(seed_cfg_path)
    seed_val = int(seed_cfg.get("seed", 0))

    env = BubbleGameEnv(render_mode="human", reward_mode=args.reward_mode, seed=seed_val)
    env.metadata["render_fps"] = args.fps

    obs, info = env.reset()
    done, trunc = False, False

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, r, done, trunc, info = env.step(action)
        env.render(mode='human')

    env.close()

if __name__ == "__main__":
    main()
