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

    model = PPO.load(args.model_path)  # Load the trained model

    # Initialize environment
    env = BubbleGameEnv()

    # Run one trial
    obs, info = env.reset()
    done = False

    # Run a single trial until the agent loses or the trial reaches max steps
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Predict action based on current observation
        obs, r, done, trunc, info = env.step(action)  # Step in the environment
        
        # Render the environment (show the game window)
        env.render(mode='human')

        # If the trial ends (either collision or max steps reached), stop the loop
        if done or trunc:  
            print("Trial ended. Resetting environment.")
            break  # End the current trial and stop the loop

        # Add a delay to control the frame rate (FPS)
        env._clock.tick(args.fps)  # Control the FPS for rendering

    env.close()  # Close the environment once the trial ends

if __name__ == "__main__":
    main()
