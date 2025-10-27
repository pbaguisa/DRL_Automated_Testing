import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from envs.game.bubble_game_env import BubbleGameEnv

# read YAML configs (optional)
try:
    import yaml 
except Exception:
    yaml = None


def load_yaml_if_exists(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        raise ImportError(
            f"Config file found at '{path}' but PyYAML isn't installed. Run: pip install pyyaml"
        )
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_bubble_survivor_seed7")
    parser.add_argument("--reward_mode", type=str, default="survivor", choices=["survivor", "speedrunner"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions (deterministic=False)")
    # Optional explicit config paths (for seeds/persona metadata)
    parser.add_argument("--persona_cfg", type=str, default=None)
    parser.add_argument("--seed_cfg", type=str, default=None)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    persona_cfg_path = args.persona_cfg or os.path.join("configs", "personas", f"{args.reward_mode}.yaml")
    seed_cfg_path    = args.seed_cfg    or os.path.join("configs", "seeds", "seed0.yaml")
    persona_cfg = load_yaml_if_exists(persona_cfg_path)
    seed_cfg    = load_yaml_if_exists(seed_cfg_path)
    seed_val    = int(seed_cfg.get("seed", 0))

    if persona_cfg:
        print(f"[CONFIG] Loaded persona config: {persona_cfg_path} (persona={persona_cfg.get('persona','unknown')})")
    if seed_cfg:
        print(f"[CONFIG] Loaded seed config: {seed_cfg_path} (seed={seed_val})")

    
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
