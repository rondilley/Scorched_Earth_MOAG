# MOAG - Mother Of All Games

A modern Python/Pygame recreation of **Scorched Earth**, the classic turn-based artillery game.

## Features

- **Destructible Terrain** - Explosions carve realistic craters into the landscape
- **Wind System** - Dynamic wind affects projectile trajectories each turn
- **Multiple Weapons** - 5 weapon types with unique behaviors
- **Hot-Seat Multiplayer** - 2-4 players on the same computer
- **AI Opponents** - Play against computer-controlled tanks
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

- Python 3.8+
- Pygame 2.0+
- NumPy (for procedural audio generation)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## Project Structure

```
MOAG/
├── main.py          # Entry point, game loop
├── settings.py      # Constants and configuration
├── game.py          # Game state management
├── terrain.py       # Procedural terrain generation & destruction
├── tank.py          # Tank class with controls & damage
├── projectile.py    # Projectile physics and explosions
├── weapons.py       # Weapon definitions
├── physics.py       # Gravity, wind, collisions
├── ai.py            # AI opponent logic
├── ui.py            # HUD, menus, overlays
└── sound.py         # Procedural audio generation
```

## Gameplay

1. Select number of players (2-4)
2. Toggle each player between Human and AI
3. Take turns adjusting angle and power
4. Fire to hit enemy tanks
5. Last tank standing wins

## License

MIT
