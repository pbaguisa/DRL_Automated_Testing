import argparse
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.game.bubble_game_env import BubbleGameEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_bubble_survivor")
    parser.add_argument("--reward_mode", type=str, default="survivor",
                        choices=["survivor", "speedrunner"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use deterministic=False to see more varied actions")
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    env = BubbleGameEnv(render_mode="human", reward_mode=args.reward_mode)
    env.metadata["render_fps"] = args.fps

    obs, info = env.reset()
    done, trunc = False, False

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, r, done, trunc, info = env.step(action)
        env.render(mode='human')
        # env._clock.tick(args.fps)

    env.close()

if __name__ == "__main__":
    main()
