# DRL Automated Testing

## Overview
This project explores the automation of **application testing** using **Deep Reinforcement Learning (DRL).** AI agents were trained using two algorithms—**PPO (Proximal Policy Optimization)** and **A2C (Advantage Actor-Critic)**—within a custom environment called the **Bubble Game.**

The goal is to evaluate how well different DRL agents can **learn to test an application autonomously**, simulating two different behaviour modes:

🛡️ Survivor Mode – Prioritizes longevity, avoiding risky behaviors.
⚡ Speedrunner Mode – Prioritizes aggressive, high-throughput interaction and speed.

## Project Objectives
- Develop a custom **Game Environment** for reinforcement learning experiments
- Train and compare **PPO** and **A2C** agents using **Stable Baselines3**
- Explore how different **reward structures** (Survivor v. Speedrunner) affect agent behaviour and performance
- Log and visualize results using **Tensorboard** and **Matplotlib**

## Algorithms Used

**PPO (Proximal Policy Optimization)**
- Stable and sample-efficient on-policy algorithm
- Performs clipped updates to prevent large, unstable policy changes

**A2C (Advantage Actor-Critic)**
- Simpler actor-critic method that updates after each rollout
- Faster but typically less stable than PPO
Both algorithms were implemented using the **Stable Baselines3 library**.

## Game Modes

🛡️ Survivor Mode
- Rewards for longevity, survival, and avoiding collisions
- Penalties for idling or being too passive
- Encourages conservative movement and survival strategy

⚡ Speedrunner Mode
- Rewards for rapid shooting and high throughput
- Penalties for inactivity or taking too long to complete tasks
- Encourages fast, aggressive gameplay and efficiency

## Setup

### Create/Start Virtual Environment
    python -m venv venv
    .\venv\Scripts\activate

### Install libraries
    pip install pygame
    pip install gym[all]
    pip install stable_baselines3
    pip install typing
    pip install tensorboard
    pip install numpy
    pip install pandas
    pip install matplotlib

### To run training:
    python src/train.py --algo ppo --reward_mode survivor --timesteps 200000 --seed 7
    python src/train.py --algo a2c --reward_mode speedrunner --timesteps 200000 --seed 7

### To evaluate trained models:
    python src/eval.py --model_path models/ppo_bubble_survivor_seed7 --reward_mode survivor --episodes 20 --csv_out logs/ppo_survivor.csv

### To visualize: 
    python src/visualize.py --model_path models/ppo_bubble_survivor_seed7 --reward_mode survivor --fps 60

### To view Tensorboard:
    tensorboard --logdir logs


## Environment 
### Observations & Actions
| Component        | Type / Shape             | Description                                                                                                          |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Observations** | `np.array`, shape `(N,)` | Player x-pos, bubble positions, nearest-bubble distance, velocity, time alive, shots, pops, accuracy, wall proximity |
| **Actions**      | `Discrete(4)`            | `0=Idle`, `1=MoveLeft`, `2=MoveRight`, `3=Shoot`                                                                     |

### Rewards (by Mode)
| Event                 | Survivor Mode | Speedrunner Mode |
| --------------------- | ------------: | ---------------: |
| Shoot                 |         +0.10 |            +0.20 |
| Pop bubble            |         +5.00 |           +10.00 |
| Stay alive (per step) |         +0.05 |            +0.01 |
| Collision / Death     |        −10.00 |            −2.00 |
| Idle (no action)      |         −0.50 |             0.00 |
| Wall camping          |         −2.00 |            −5.00 |


## Algorithm Configuration
| Hyperparameter    |        PPO (used) |        A2C (used) | Notes                        |
| ----------------- | ----------------: | ----------------: | ---------------------------- |
| `policy`          |       `MlpPolicy` |       `MlpPolicy` | Feed-forward MLP             |
| `n_steps`         |    2048 (or 4096) |    1024 (or 2048) | Rollout length before update |
| `batch_size`      |      256 (or 512) |      256 (or 512) | Larger → stabler gradients   |
| `gamma`           |     0.995 → 0.999 |     0.995 → 0.999 | Long-term reward weighting   |
| `gae_lambda`      |       0.98 → 0.99 |       0.98 → 0.99 | Advantage estimation         |
| `n_epochs`        |       20 (±10–30) |                10 | SGD passes per update        |
| `learning_rate`   | 1e-4 (±5e-5…5e-4) | 1e-4 (±5e-5…5e-4) | Lower = more stable          |
| `clip_range`      |           0.1–0.2 |                 — | PPO policy clip only         |
| `total_timesteps` |           500,000 |           500,000 | Training budget              |
| `seed`            |                 7 |                 7 | Reproducibility              |





