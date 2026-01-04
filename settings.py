# Scorched Earth Modern - Settings and Constants

# Screen settings
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60
TITLE = "Scorched Earth Modern"

# Audio settings
SOUND_ENABLED = True
MUSIC_ENABLED = True
SFX_VOLUME = 0.7      # 0.0 to 1.0
MUSIC_VOLUME = 0.3    # 0.0 to 1.0
SAMPLE_RATE = 22050   # Audio sample rate in Hz

# Physics
GRAVITY = 500  # pixels per second squared
WIND_MIN = -8
WIND_MAX = 8

# Terrain
TERRAIN_MIN_HEIGHT = 100  # minimum terrain height from bottom
TERRAIN_MAX_HEIGHT = 500  # maximum terrain height from bottom
TERRAIN_ROUGHNESS = 0.5   # for midpoint displacement (0-1)

# Tank settings
TANK_WIDTH = 40
TANK_HEIGHT = 20
TANK_TURRET_LENGTH = 25
TANK_MAX_HEALTH = 100
ANGLE_MIN = 0    # degrees (0 = right)
ANGLE_MAX = 180  # degrees (180 = left)
ANGLE_SPEED = 60  # degrees per second
POWER_MIN = 50
POWER_MAX = 500
POWER_SPEED = 200  # units per second

# Game settings
MAX_PLAYERS = 4
MIN_PLAYERS = 2

# Retro color palette
COLORS = {
    # Sky gradient
    'sky_top': (26, 26, 46),      # #1a1a2e
    'sky_bottom': (22, 33, 62),   # #16213e

    # Terrain
    'terrain_top': (92, 64, 51),    # #5c4033
    'terrain_bottom': (74, 55, 40), # #4a3728
    'terrain_grass': (34, 85, 34),  # green top layer

    # UI
    'text': (255, 255, 255),
    'text_shadow': (0, 0, 0),
    'hud_bg': (20, 20, 40, 180),
    'health_bar_bg': (60, 60, 60),
    'health_bar_fg': (50, 205, 50),
    'health_bar_low': (255, 69, 0),
    'power_bar': (255, 215, 0),

    # Tanks (player colors)
    'tank_1': (220, 50, 50),    # Red
    'tank_2': (50, 100, 220),   # Blue
    'tank_3': (50, 180, 50),    # Green
    'tank_4': (220, 180, 50),   # Yellow

    # Explosions
    'explosion_outer': (255, 100, 0),
    'explosion_inner': (255, 200, 50),
    'explosion_flash': (255, 255, 255),

    # Projectile
    'projectile': (255, 255, 255),
    'trail': (200, 200, 200),

    # Wind indicator
    'wind_arrow': (150, 200, 255),

    # Menu
    'menu_bg': (15, 15, 30),
    'menu_selected': (80, 80, 120),
    'menu_text': (200, 200, 220),
}

# Tank colors list for easy indexing
TANK_COLORS = [
    COLORS['tank_1'],
    COLORS['tank_2'],
    COLORS['tank_3'],
    COLORS['tank_4'],
]

# Game states
class GameState:
    MENU = 'menu'
    SETUP = 'setup'
    PLAYING = 'playing'
    AIMING = 'aiming'
    FIRING = 'firing'
    EXPLOSION = 'explosion'
    TURN_TRANSITION = 'turn_transition'
    GAME_OVER = 'game_over'

# AI difficulty levels
class AIDifficulty:
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'
    RL_PPO = 'rl_ppo'    # Trained PPO agent
    RL_DQN = 'rl_dqn'    # Trained DQN agent

# AI settings
AI_SETTINGS = {
    AIDifficulty.EASY: {
        'angle_error': 15,   # degrees of random error
        'power_error': 50,   # power units of random error
        'reaction_time': 1.5, # seconds before AI shoots
    },
    AIDifficulty.MEDIUM: {
        'angle_error': 8,
        'power_error': 25,
        'reaction_time': 1.0,
    },
    AIDifficulty.HARD: {
        'angle_error': 3,
        'power_error': 10,
        'reaction_time': 0.5,
    },
}
