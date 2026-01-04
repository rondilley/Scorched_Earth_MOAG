# Training configuration and hyperparameters

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Environment settings
ENV_CONFIG = {
    'num_players': 2,
    'max_turns': 100,
    'fixed_dt': 1.0 / 10.0,  # 10 FPS - faster simulation for training
    'fast_mode': True,  # Skip animations and delays
}

# Observation normalization constants
OBS_CONFIG = {
    'terrain_size': 1024,
    'max_opponents': 3,
    'tank_features': 6,  # x, y, health, angle, power, weapon
    'opponent_features': 4,  # x, y, health, alive
}

# Action space configuration
ACTION_CONFIG = {
    'angle_min': 0,
    'angle_max': 180,
    'power_min': 50,
    'power_max': 500,
    'num_weapons': 5,
    # DQN discretization
    'angle_bins': 19,  # 0, 10, 20, ..., 180
    'power_bins': 10,  # 50, 100, ..., 500
}

# Reward shaping
REWARD_CONFIG = {
    'damage_multiplier': 0.1,  # Per point of damage dealt
    'kill_bonus': 5.0,
    'win_bonus': 20.0,
    'lose_penalty': -10.0,
    'self_damage_multiplier': -0.05,
    'step_penalty': -0.01,
}

# PPO hyperparameters
PPO_CONFIG = {
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'n_steps': 2048,
    'n_epochs': 10,
    'batch_size': 64,
}

# DQN hyperparameters
DQN_CONFIG = {
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_steps': 50000,
    'target_update_freq': 1000,
    'batch_size': 32,
    'buffer_size': 100000,
    'learning_starts': 1000,
}

# Network architecture
NETWORK_CONFIG = {
    'hidden_size': 256,
    'terrain_channels': [32, 64, 64],
    'terrain_kernels': [8, 4, 3],
    'terrain_strides': [4, 2, 2],
}

# LLM agent settings
LLM_CONFIG = {
    'default_provider': 'anthropic',
    'temperature': 0.7,
    'timeout': 30.0,
    'max_retries': 3,
    'models': {
        'anthropic': 'claude-sonnet-4-20250514',
        'openai': 'gpt-4o',
    }
}
