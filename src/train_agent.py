import gym
from stable_baselines3 import PPO
from envs.game.bubble_game_env import BubbleGameEnv

# Create the environment
env = BubbleGameEnv()

# Instantiate the PPO agent with CNN policy for image-based inputs
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("bubble_game_model")

# Test the trained agent
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

# Close the environment after testing
env.close()
