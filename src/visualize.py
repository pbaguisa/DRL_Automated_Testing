import argparse
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.game.bubble_game_env import BubbleGameEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_bubble_game")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    env = BubbleGameEnv()

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        env.render(mode='human')  # Render the game window for visualization

    env.close()

if __name__ == "__main__":
    main()
