# MOAG - Mother Of All Games

A modern Python/Pygame recreation of **Scorched Earth**, the classic turn-based artillery game. Includes a complete reinforcement learning training system for training AI opponents.

## Features

- **Destructible Terrain** - Explosions carve realistic craters into the landscape
- **Wind System** - Dynamic wind affects projectile trajectories each turn
- **Multiple Weapons** - 5 weapon types with unique behaviors
- **Hot-Seat Multiplayer** - 2-4 players on the same computer
- **AI Opponents** - Heuristic AI with 3 difficulty levels, plus trained RL models
- **Reinforcement Learning** - Train your own AI using PPO or DQN algorithms
- **Retro Pixel Aesthetic** - DOS-era inspired visuals
- **Procedural Audio** - Generated sound effects and background music (no audio files needed)

## Weapons

| Weapon | Damage | Radius | Special |
|--------|--------|--------|---------|
| Standard Shell | 25 | 30 | - |
| Big Bertha | 40 | 50 | Larger explosion |
| Baby Nuke | 75 | 80 | Massive damage |
| Dirt Ball | 0 | 40 | Adds terrain |
| MIRV | 20 | 25 | Splits into 5 bombs |

## Controls

| Action | Keys |
|--------|------|
| Aim Left/Right | LEFT/RIGHT or A/D |
| Adjust Power | UP/DOWN or W/S |
| Fire | SPACE or ENTER |
| Next Weapon | TAB or Q |
| Previous Weapon | SHIFT+TAB or E |

## Requirements

**Core Game:**
- Python 3.8+
- Pygame 2.0+
- NumPy

**RL Training (optional):**
- PyTorch 2.0+
- TensorBoard (for logging)
- anthropic or openai (for LLM data generation)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## AI Difficulty Levels

The game supports multiple AI types selectable in the setup screen:

| Difficulty | Description |
|------------|-------------|
| EASY | High error, slow reactions |
| MEDIUM | Moderate accuracy |
| HARD | Near-precise shots |
| RL_PPO | Trained PPO neural network |
| RL_DQN | Trained DQN neural network |

Cycle through AI types by pressing the player number key (1-4) in the setup screen.

## Training RL Agents

Train a PPO agent (recommended):
```bash
python -m training.train --algo ppo --total-steps 1000000
```

Train a DQN agent:
```bash
python -m training.train --algo dqn --total-steps 500000
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

Trained models are saved to `models/` and automatically loaded when selecting RL_PPO or RL_DQN difficulty.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation of the training system.

## Project Structure

```
MOAG/
├── main.py              # Entry point, game loop
├── settings.py          # Constants and configuration
├── game.py              # Game state management
├── terrain.py           # Procedural terrain generation & destruction
├── tank.py              # Tank class with controls & damage
├── projectile.py        # Projectile physics and explosions
├── weapons.py           # Weapon definitions
├── physics.py           # Gravity, wind, collisions
├── ai.py                # AI logic (heuristic + RLAgent)
├── ui.py                # HUD, menus, overlays
├── sound.py             # Procedural audio generation
│
├── training/            # RL training system
│   ├── env.py           # Gym-like environment wrapper
│   ├── networks.py      # Neural network architectures
│   ├── ppo.py           # PPO agent implementation
│   ├── dqn.py           # DQN agent implementation
│   ├── replay_buffer.py # Experience replay buffers
│   ├── config.py        # Training hyperparameters
│   ├── train.py         # CLI training script
│   └── llm_data_generator.py  # Offline LLM data generation
│
├── models/              # Saved model checkpoints (gitignored)
├── logs/                # TensorBoard logs (gitignored)
│
├── ARCHITECTURE.md      # Detailed system documentation
├── CLAUDE.md            # AI development guide
└── VIBE_HISTORY.md      # Development log
```

## Gameplay

1. Select number of players (2-4) with LEFT/RIGHT arrows
2. Press number keys (1-4) to cycle each player between Human and AI difficulties
3. Press SPACE to start
4. Take turns adjusting angle and power
5. Fire to hit enemy tanks
6. Last tank standing wins

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive system architecture, design decisions, and engineering details
- **[CLAUDE.md](CLAUDE.md)** - Quick reference for AI-assisted development
- **[VIBE_HISTORY.md](VIBE_HISTORY.md)** - Development session log

## License

MIT
