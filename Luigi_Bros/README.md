# 🍄 Super Luigi Bros - DRL for Automated Testing

## 🎯 Overview
This project explores **automated game testing** using **Deep Reinforcement Learning (DRL)**. AI agents are trained using **PPO** and **A2C** algorithms to test a Super Mario Bros 1-1 clone game with two distinct testing personas:

🔍 **Explorer Mode** – Maximizes question block interaction, coin collection, and thorough exploration before level completion.

⚡ **Speedrunner Mode** – Minimizes completion time by rushing to the flagpole as quickly as possible.

## 🎮 The Game
A faithful recreation of Super Mario Bros World 1-1 featuring:
- Classic platforming physics (gravity, jumping, sprinting)
- Enemies (Goombas with edge detection)
- Question blocks (coins and mushrooms)
- Pipes, bricks, power-ups
- Flagpole victory condition
- Small→Super Luigi transformation

## 🎯 Project Objectives
- Develop a **Gymnasium-compatible environment** for the Mario game
- Train and compare **PPO vs A2C** agents
- Explore how different **reward structures** (Explorer vs Speedrunner) affect agent behavior
- **Automatically collect metrics** for testing analysis
- Log and visualize results using **TensorBoard** and **CSV exports**

## 🧮 Algorithms Used

### PPO (Proximal Policy Optimization)
- Stable, sample-efficient on-policy algorithm
- Clipped updates prevent large policy changes
- Hyperparameters: n_steps=2048, batch_size=256, lr=3e-4

### A2C (Advantage Actor-Critic)
- Simpler actor-critic method with faster updates
- Less stable but quicker to train
- Hyperparameters: n_steps=1024, lr=7e-4

Both implemented using **Stable Baselines3**.

## 🎭 Personas (Reward Modes)

### 🔍 Explorer Mode
**Goal**: Thoroughly test the game by interacting with all question blocks and collecting coins.

**Rewards**:
- `+0.1` per step (stay alive bonus)
- `+15.0` per question block hit
- `+25.0` per mushroom block hit
- `+5.0` per coin collected
- `+0.08 × dx` for forward progress
- `+0.2 × dx` for reaching new furthest point (×2.0 multiplier)
- `+0.5 to 2.0` for jumping near question blocks (proximity-based)

**Penalties**:
- `-0.15` for moving backward
- `-0.1` for standing still

**Testing Value**: Discovers power-up mechanics, question block bugs, collision edge cases, exploratory behavior.

### ⚡ Speedrunner Mode
**Goal**: Complete the level as fast as possible.

**Rewards**:
- `+0.25 × dx` for forward progress
- `+0.3 × dx` for reaching new furthest point (×3.0 multiplier)

**Penalties**:
- `-0.005` per step (time penalty)
- `-0.15` for not moving forward or standing still

**Testing Value**: Stress-tests physics engine, finds optimal paths, exposes softlocks, tests speed-running strategies.

## 🌍 Environment Details

### 🎮 Actions (8 discrete)
| ID | Action                  |
|----|-------------------------|
| 0  | No-op                   |
| 1  | Left                    |
| 2  | Right                   |
| 3  | Jump                    |
| 4  | Right + Jump            |
| 5  | Sprint + Right          |
| 6  | Sprint + Right + Jump   |
| 7  | Left + Jump             |

### 👁️ Observations (97 floats)
| Component | Size | Description |
|-----------|------|-------------|
| Player State | 8 | x, y, vx, vy, super, on_ground, coins, furthest_x |
| Tile Grid | 63 | 9×7 grid around player (wider view ahead) |
| Enemies | 15 | 5 nearest enemies (x_offset, y_offset, alive) |
| Question Blocks | 9 | 3 nearest blocks (x_offset, y_offset, used) |
| Progress | 2 | distance_to_flag, completion_progress |

**Tile Types**: 0=empty, 1=ground, 2=brick, 3=question, 4=question_used, 5=pipe

**Note**: Enemy positions are randomized ±1 tile (±32 pixels) each episode for variance in testing.

### 📊 Metrics Collected
Per-episode metrics automatically logged:
- `reward` - Total episode reward
- `deaths` - Number of deaths (0 or 1)
- `completions` - Level completed (0 or 1)
- `frames_alive` - Episode length in frames
- `coins_collected` - Total coins collected
- `blocks_hit` - Question blocks activated
- `mushroom_blocks_hit` - Power-up blocks hit
- `blocks_hit_ratio` - % of all blocks hit
- `max_x_reached` - Furthest position reached
- `distance_traveled` - Total forward progress
- `jumps` - Number of jump actions taken

## ⚙️ Setup

### 📋 Prerequisites
- Python 3.8+
- pip or conda

### 🔧 Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd luigi_bros

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install gymnasium stable-baselines3[extra] pygame numpy tensorboard matplotlib pandas pyyaml scipy seaborn
```

### 📦 Dependencies
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
pygame>=2.5.0
numpy>=1.24.0
tensorboard>=2.14.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0.0
scipy>=1.10.0
seaborn>=0.12.0
```

## 🚀 Usage

### 🎓 Training

Train an agent with PPO algorithm in explorer mode:
```bash
python src/train.py --algo ppo --reward_mode explorer --timesteps 500000 --seed 7
```

Train an agent with A2C algorithm in speedrunner mode:
```bash
python src/train.py --algo a2c --reward_mode speedrunner --timesteps 500000 --seed 7
```

**Arguments**:
- `--algo`: Algorithm choice (`ppo` or `a2c`)
- `--reward_mode`: Persona (`explorer` or `speedrunner`)
- `--timesteps`: Training duration (default: 500,000)
- `--seed`: Random seed for reproducibility (default: 7)
- `--logdir`: TensorBoard log directory (default: `logs/`)
- `--modeldir`: Model save directory (default: `models/`)

**Output**:
- Trained model: `models/{algo}_luigi_{mode}_seed{seed}.zip`
- Training logs: `logs/{algo}_luigi_{mode}_seed{seed}/`

### 🧪 Evaluation

Evaluate a trained model over multiple episodes:
```bash
python src/eval.py --model_path models/ppo_luigi_explorer_seed7 --reward_mode explorer --episodes 20 --csv_out logs/ppo_explorer_seed7.csv
```

**Arguments**:
- `--model_path`: Path to trained model (without `.zip`)
- `--reward_mode`: Environment mode (should match training)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Show pygame window (1) or headless (0)
- `--csv_out`: Output CSV path (default: `logs/eval_metrics.csv`)

**Output**: CSV file with detailed per-episode metrics + summary statistics.

### 🎥 Visualization

Watch a trained agent play with live rendering:
```bash
python src/visualize.py --model_path models/ppo_luigi_explorer_seed7 --reward_mode explorer --fps 60
```

**Arguments**:
- `--model_path`: Path to trained model
- `--reward_mode`: Environment mode
- `--fps`: Rendering frame rate (default: 60)
- `--stochastic`: Use stochastic actions (default: deterministic)

### 📊 TensorBoard

View training progress:
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## 📁 Project Structure

```
luigi_bros/
├── envs/
│   └── game/
│       ├── __init__.py
│       ├── luigi_env.py           # Gymnasium wrapper
├── src/
│   ├── train.py                   # Training script
│   ├── eval.py                    # Evaluation script
│   └── visualize.py               # Visualization script
├── configs/
│   ├── rewards.yaml               # Reward shaping configuration
│   ├── ppo.yaml                   # PPO hyperparameters (documentation)
│   └── a2c.yaml                   # A2C hyperparameters (documentation)
├── models/                        # Saved trained models (.zip)
├── logs/                          # TensorBoard logs + CSV metrics
│   ├── ppo_explorer_seed7.csv
│   ├── ppo_speedrunner_seed7.csv
│   ├── a2c_explorer_seed7.csv
│   ├── a2c_speedrunner_seed7.csv
    └── Comparisons/
│       ├── summary_all_models.csv     # Comparison summary
│       ├── explorer_comparison_seed7.csv
│       └── speedrunner_comparison_seed7.csv
├── notebooks/
│   └── analysis.ipynb             # Analysis and visualizations
├── playable/
│   └── luigi.py                   # Original playable game
├── commands.txt                   # Usable commands
├── README.md                      
└── requirements.txt               # Python dependencies
```

## Demos

### A2C Explorer Agent
![A2C Explorer Demo](Demos/a2c_exp.gif)
### PPO Explorer Agent
![PPO Explorer Demo](Demos/ppo_exp.gif)
### A2C Speedrunner Agent
![A2C Speedrunner Demo](Demos/a2c_speed.gif)
### PPO Speedrunner Agent
![PPO Speedrunner Demo](Demos/ppo_speed.gif)