# 🤖 Deep Reinforcement Learning for Automated Testing

## 🧭 Overview
This project explores the automation of **application testing** using **Deep Reinforcement Learning (DRL).** AI agents were trained using two algorithms—**PPO (Proximal Policy Optimization)** and **A2C (Advantage Actor-Critic)**—within a custom environment called the **Bubble Game.**

The goal is to evaluate how well different DRL agents can **learn to test an application autonomously**, simulating two different behaviour modes:

🛡️ Survivor Mode – Prioritizes longevity, avoiding risky behaviors.
⚡ Speedrunner Mode – Prioritizes aggressive, high-throughput interaction and speed.

## 🎯 Project Objectives
- Develop a custom **Game Environment** for reinforcement learning experiments
- Train and compare **PPO** and **A2C** agents using **Stable Baselines3**
- Explore how different **reward structures** (Survivor v. Speedrunner) affect agent behaviour and performance
- Log and visualize results using **Tensorboard** and **Matplotlib**

## 🧮 Algorithms Used

**PPO (Proximal Policy Optimization)**
- Stable and sample-efficient on-policy algorithm
- Performs clipped updates to prevent large, unstable policy changes

**A2C (Advantage Actor-Critic)**
- Simpler actor-critic method that updates after each rollout
- Faster but typically less stable than PPO
Both algorithms were implemented using the **Stable Baselines3 library**.

## 🎮 Game Modes

🛡️ Survivor Mode
- Rewards for longevity, survival, and avoiding collisions
- Penalties for idling or being too passive
- Encourages conservative movement and survival strategy

⚡ Speedrunner Mode
- Rewards for rapid shooting and high throughput
- Penalties for inactivity or taking too long to complete tasks
- Encourages fast, aggressive gameplay and efficiency

## ⚙️ Setup

### 🧩 Create/Start Virtual Environment
    python -m venv venv
    .\venv\Scripts\activate

### 📦 Install libraries
    pip install gym[all] stable_baselines3 typing numpy tensorboard numpy pandas matplotlib

### 🧠 To run training:
    python src/train.py --algo ppo --reward_mode survivor --timesteps 200000 --seed 7

### 🧪 To evaluate trained models:
    python src/eval.py --model_path models/ppo_bubble_survivor_seed7 --reward_mode survivor --episodes 20 --csv_out logs/ppo_survivor.csv

### 🎥 To visualize: 
    python src/visualize.py --model_path models/ppo_bubble_survivor_seed7 --reward_mode survivor --fps 60

### 📊 To view Tensorboard:
    tensorboard --logdir logs

## 🌍 Environment 
### 🎮 Actions
| ID | Meaning                   |
| -: | ------------------------- |
|  0 | Move Left                 |
|  1 | Move Right                |
|  2 | Shoot (cooldown 2 frames) |
|  3 | No-op                     |

### 👁️ Observations (shape = 8, float32)
| Indexes | Description                          |
| ------- | ------------------------------------ |
| 0–1     | Player `(x, y)`                      |
| 2–3     | Bullet `(x, y)` (zeros if no bullet) |
| 4–5     | Bubble 1 `(x, y)`                    |
| 6–7     | Bubble 2 `(x, y)`                    |

### 💰 Rewards (by Mode)
| Event / Term                                 |                                   Survivor Mode |                                Speedrunner Mode |
| -------------------------------------------- | ----------------------------------------------: | ----------------------------------------------: |
| **Shoot**                                    |                                         `+0.50` |                                         `+0.75` |
| **Per-step time shaping**                    |                           `+0.05` (alive bonus) |                          `-0.01` (step penalty) |
| **Align gain**                               |             `+0.002 × (WIDTH − min(dx, WIDTH))` |             `+0.004 × (WIDTH − min(dx, WIDTH))` |
| **Proximity bonus** *(under bubble & close)* |                          `+0.10` *(SAFE_BONUS)* |                         `+0.15` *(CLOSE_BONUS)* |
| **Wall penalty**                             |            `-0.05` per step *(left/right wall)* |                                               — |
| **Bullet drag**                              |       `-0.005` per step *(while bullet exists)* |                                               — |
| **Pop bubble**                               |                                         `+10.0` |                                         `+20.0` |
| **Death (player collision)**                 |                                        `-100.0` |                                        `-100.0` |
| **Truncation**                               | `max_steps=200000` → no special terminal reward | `max_steps=200000` → no special terminal reward |

Here's the aligned version of your table:

## ⚖️ Algorithm Configuration
| Hyperparameter    |        PPO (used) |        A2C (used) | Notes                        |
|-------------------|------------------:|------------------:|------------------------------|
| `policy`          |       `MlpPolicy` |       `MlpPolicy` | Feed-forward MLP             |
| `n_steps`         |              2048 |                —  | Rollout length before update |
| `batch_size`      |               256 |                —  | Larger → stabler gradients   |
| `gamma`           |     0.995 → 0.999 |              0.99 | Long-term reward weighting   |
| `gae_lambda`      |       0.98 → 0.99 |              1.00 | Advantage estimation         |
| `learning_rate`   |       3e-4 → 1e-4 |       7e-4 → 6e-4 | Lower = more stable          |
| `total_timesteps` |           200,000 |           200,000 | Training budget              |
| `seed`            |                 7 |                 7 | Reproducibility              |

## 🎬 PPO Agent in Speedrunner Mode shooting bubbles
<img src="https://github.com/user-attachments/assets/7c45bec4-f6a4-454a-a4f6-fe6495cc0e19" width="650" alt="game_example">

