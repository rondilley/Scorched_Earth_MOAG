# Training module for Scorched Earth RL agents
"""
This module provides reinforcement learning training capabilities:
- ScorchedEarthEnv: Gym-like environment wrapper
- PPOAgent: Proximal Policy Optimization
- DQNAgent: Deep Q-Network
- LLMAgent: LLM-based decision making via API
"""

from training.env import ScorchedEarthEnv

__all__ = ['ScorchedEarthEnv']
