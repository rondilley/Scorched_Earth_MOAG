# Gym-like environment wrapper for Scorched Earth

import sys
import os
import random
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from game import Game
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GameState,
    TANK_MAX_HEALTH, ANGLE_MIN, ANGLE_MAX,
    POWER_MIN, POWER_MAX, WIND_MIN, WIND_MAX,
    TERRAIN_MAX_HEIGHT
)
from ai import AI
from training.config import (
    ENV_CONFIG, OBS_CONFIG, ACTION_CONFIG, REWARD_CONFIG
)


class ScorchedEarthEnv:
    """Gym-like environment wrapper for Scorched Earth game.

    Supports training RL agents via step/reset interface.
    """

    def __init__(
        self,
        num_players=2,
        opponent_type='heuristic',
        opponent_difficulty='medium',
        max_turns=100,
        render_mode=None,
        seed=None
    ):
        """Initialize the environment.

        Args:
            num_players: Number of tanks (2-4)
            opponent_type: 'heuristic', 'self', or 'random'
            opponent_difficulty: For heuristic opponents ('easy', 'medium', 'hard')
            max_turns: Maximum turns before truncation
            render_mode: None for headless, 'human' for display
            seed: Random seed for reproducibility
        """
        self.num_players = num_players
        self.opponent_type = opponent_type
        self.opponent_difficulty = opponent_difficulty
        self.max_turns = max_turns
        self.render_mode = render_mode

        # Initialize pygame minimally for headless mode
        if not pygame.get_init():
            pygame.init()

        # Create screen only if rendering
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Scorched Earth - Training")
            self.headless = False
        else:
            self.screen = None
            self.headless = True

        # Game instance (created on reset)
        self.game = None
        self.agent_player_id = 0  # RL agent controls player 0

        # Episode tracking
        self.turn_count = 0
        self.episode_reward = 0

        # Previous state for reward calculation
        self._prev_health = {}
        self._prev_alive = {}

        # Set seed if provided
        if seed is not None:
            self.seed(seed)

        # Fixed timestep for deterministic simulation
        self.dt = ENV_CONFIG['fixed_dt']

    def seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, seed=None):
        """Reset environment for new episode.

        Args:
            seed: Optional random seed

        Returns:
            observation: Initial observation dict
            info: Additional info dict
        """
        if seed is not None:
            self.seed(seed)

        # Create new game in headless mode
        self.game = Game(screen=self.screen, headless=self.headless)

        # Configure players - agent is player 0, rest are opponents
        self.game.num_players = self.num_players
        self.game.player_types = ['human']  # Agent acts as "human" (manual control)
        for i in range(1, self.num_players):
            if self.opponent_type == 'heuristic':
                self.game.player_types.append('ai')
            else:
                self.game.player_types.append('human')  # We'll control these manually

        # Start the game (skip menu/setup states)
        self.game._start_game()

        # Initialize opponent AI controllers for heuristic opponents
        if self.opponent_type == 'heuristic':
            for i in range(1, self.num_players):
                if i in self.game.ai_controllers:
                    # AI controllers already created in _start_game
                    pass

        # Reset episode tracking
        self.turn_count = 0
        self.episode_reward = 0

        # Store initial health states
        self._store_health_state()

        # Make sure it's the agent's turn
        self._advance_to_agent_turn()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """Execute one action and advance game state.

        Args:
            action: Dict with 'angle', 'power', 'weapon' keys

        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: True if episode ended (win/lose)
            truncated: True if max turns reached
            info: Additional info dict
        """
        # Validate action
        angle = np.clip(action.get('angle', 90), ANGLE_MIN, ANGLE_MAX)
        power = np.clip(action.get('power', 275), POWER_MIN, POWER_MAX)
        weapon = int(np.clip(action.get('weapon', 0), 0, 4))

        # Get agent's tank
        agent_tank = self.game.tanks[self.agent_player_id]

        # Apply action to tank
        agent_tank.angle = angle
        agent_tank.power = power
        agent_tank.current_weapon = weapon

        # Fire projectile
        self.game._fire_projectile()

        # Simulate until projectile resolves
        self._simulate_until_state(GameState.AIMING, GameState.GAME_OVER)

        # If game not over, let opponents take their turns
        if self.game.state != GameState.GAME_OVER:
            self._advance_to_agent_turn()

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Store new health state for next reward calculation
        self._store_health_state()

        # Check termination conditions
        terminated = self.game.state == GameState.GAME_OVER
        self.turn_count += 1
        truncated = self.turn_count >= self.max_turns and not terminated

        # Render if requested
        if self.render_mode == 'human':
            self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _simulate_until_state(self, *target_states):
        """Run game simulation until reaching one of target states.

        Uses large timesteps and limits iterations for speed.
        """
        max_iterations = 2000  # Safety limit
        iterations = 0

        while self.game.state not in target_states and iterations < max_iterations:
            self.game.update(self.dt)
            iterations += 1

            # Also check for game over during simulation
            if self.game.state == GameState.GAME_OVER:
                break

    def _advance_to_agent_turn(self):
        """Simulate opponent turns until it's the agent's turn again.

        Optimized for speed: opponents fire immediately without "thinking" delay.
        """
        max_turns = 10  # Max opponent turns before giving up
        turns_taken = 0

        while turns_taken < max_turns:
            current_tank = self.game._get_current_tank()

            # Check if game is over
            if self.game.state == GameState.GAME_OVER:
                break

            # Check if it's the agent's turn
            if current_tank and current_tank.player_id == self.agent_player_id:
                if self.game.state == GameState.AIMING:
                    break

            # Handle opponent turns - fire immediately, no delay
            if current_tank and current_tank.player_id != self.agent_player_id:
                if self.game.state == GameState.AIMING:
                    turns_taken += 1

                    if self.opponent_type == 'heuristic':
                        # Get AI to calculate shot instantly (no animation)
                        ai = self.game.ai_controllers.get(current_tank.player_id)
                        if ai:
                            ai.start_turn(self.game.tanks, self.game.terrain, self.game.physics)
                            # Apply calculated values directly
                            current_tank.angle = ai.target_angle
                            current_tank.power = ai.target_power
                        else:
                            # Fallback to random
                            current_tank.angle = random.uniform(ANGLE_MIN, ANGLE_MAX)
                            current_tank.power = random.uniform(POWER_MIN, POWER_MAX)
                            current_tank.current_weapon = random.randint(0, 4)
                    else:
                        # Random action for opponent
                        current_tank.angle = random.uniform(ANGLE_MIN, ANGLE_MAX)
                        current_tank.power = random.uniform(POWER_MIN, POWER_MAX)
                        current_tank.current_weapon = random.randint(0, 4)

                    # Fire immediately
                    self.game._fire_projectile()

                    # Simulate projectile to completion
                    self._simulate_until_state(GameState.AIMING, GameState.GAME_OVER)
                    continue

            # Advance game state (turn transitions, etc)
            self.game.update(self.dt)

    def _store_health_state(self):
        """Store current health states for reward calculation."""
        self._prev_health = {}
        self._prev_alive = {}
        for tank in self.game.tanks:
            self._prev_health[tank.player_id] = tank.health
            self._prev_alive[tank.player_id] = tank.alive

    def _calculate_reward(self):
        """Calculate reward based on state changes."""
        reward = 0.0
        agent_tank = self.game.tanks[self.agent_player_id]

        # Damage dealt to opponents
        for tank in self.game.tanks:
            if tank.player_id == self.agent_player_id:
                continue

            prev_health = self._prev_health.get(tank.player_id, TANK_MAX_HEALTH)
            damage_dealt = prev_health - tank.health
            if damage_dealt > 0:
                reward += damage_dealt * REWARD_CONFIG['damage_multiplier']

            # Kill bonus
            prev_alive = self._prev_alive.get(tank.player_id, True)
            if prev_alive and not tank.alive:
                reward += REWARD_CONFIG['kill_bonus']

        # Self damage penalty
        prev_self_health = self._prev_health.get(self.agent_player_id, TANK_MAX_HEALTH)
        self_damage = prev_self_health - agent_tank.health
        if self_damage > 0:
            reward += self_damage * REWARD_CONFIG['self_damage_multiplier']

        # Win/lose bonus
        if self.game.state == GameState.GAME_OVER:
            alive_tanks = [t for t in self.game.tanks if t.alive]
            if len(alive_tanks) == 1 and alive_tanks[0].player_id == self.agent_player_id:
                reward += REWARD_CONFIG['win_bonus']
            elif not agent_tank.alive:
                reward += REWARD_CONFIG['lose_penalty']

        # Step penalty
        reward += REWARD_CONFIG['step_penalty']

        return reward

    def _get_observation(self):
        """Build observation dictionary from game state.

        Returns:
            Dict with terrain, player_tank, opponent_tanks, wind
        """
        # Normalize terrain heights (0-1 range)
        terrain_heights = np.array(self.game.terrain.heights, dtype=np.float32)
        terrain_obs = terrain_heights / TERRAIN_MAX_HEIGHT

        # Agent tank features
        agent_tank = self.game.tanks[self.agent_player_id]
        player_features = np.array([
            agent_tank.x / SCREEN_WIDTH,
            agent_tank.y / SCREEN_HEIGHT,
            agent_tank.health / TANK_MAX_HEALTH,
            agent_tank.angle / ANGLE_MAX,
            (agent_tank.power - POWER_MIN) / (POWER_MAX - POWER_MIN),
            agent_tank.current_weapon / 4.0,
        ], dtype=np.float32)

        # Opponent features (pad to max_opponents)
        opponent_features = []
        for tank in self.game.tanks:
            if tank.player_id == self.agent_player_id:
                continue
            opponent_features.extend([
                tank.x / SCREEN_WIDTH,
                tank.y / SCREEN_HEIGHT,
                tank.health / TANK_MAX_HEALTH,
                float(tank.alive),
            ])

        # Pad if fewer opponents
        max_opponent_features = OBS_CONFIG['max_opponents'] * OBS_CONFIG['opponent_features']
        while len(opponent_features) < max_opponent_features:
            opponent_features.extend([0.0, 0.0, 0.0, 0.0])

        opponent_obs = np.array(opponent_features[:max_opponent_features], dtype=np.float32)

        # Wind normalized to -1 to 1
        wind_obs = np.array([self.game.physics.wind / max(abs(WIND_MIN), abs(WIND_MAX))],
                           dtype=np.float32)

        return {
            'terrain': terrain_obs,
            'player_tank': player_features,
            'opponent_tanks': opponent_obs,
            'wind': wind_obs,
        }

    def _get_info(self):
        """Get additional info about current state."""
        agent_tank = self.game.tanks[self.agent_player_id]
        alive_opponents = sum(1 for t in self.game.tanks
                             if t.player_id != self.agent_player_id and t.alive)

        return {
            'turn': self.turn_count,
            'agent_health': agent_tank.health,
            'agent_alive': agent_tank.alive,
            'alive_opponents': alive_opponents,
            'game_state': self.game.state,
            'episode_reward': self.episode_reward,
        }

    def render(self):
        """Render the game (if render_mode is 'human')."""
        if self.render_mode == 'human' and self.screen is not None:
            self.game.render(self.screen)
            pygame.display.flip()

    def close(self):
        """Clean up resources."""
        if self.render_mode == 'human':
            pygame.quit()


def action_to_dict(angle, power, weapon):
    """Convert individual action values to action dict."""
    return {
        'angle': float(angle),
        'power': float(power),
        'weapon': int(weapon),
    }


def index_to_action(action_index):
    """Convert flat action index to action dict (for DQN).

    Total actions = angle_bins * power_bins * num_weapons
    """
    num_weapons = ACTION_CONFIG['num_weapons']
    power_bins = ACTION_CONFIG['power_bins']
    angle_bins = ACTION_CONFIG['angle_bins']

    weapon = action_index % num_weapons
    action_index //= num_weapons
    power_bin = action_index % power_bins
    angle_bin = action_index // power_bins

    # Convert bins to actual values
    angle = angle_bin * (ACTION_CONFIG['angle_max'] / (angle_bins - 1))
    power_step = (ACTION_CONFIG['power_max'] - ACTION_CONFIG['power_min']) / (power_bins - 1)
    power = ACTION_CONFIG['power_min'] + power_bin * power_step

    return action_to_dict(angle, power, weapon)


def action_to_index(action):
    """Convert action dict to flat index (for DQN)."""
    num_weapons = ACTION_CONFIG['num_weapons']
    power_bins = ACTION_CONFIG['power_bins']
    angle_bins = ACTION_CONFIG['angle_bins']

    # Convert values to bins
    angle_bin = int(round(action['angle'] * (angle_bins - 1) / ACTION_CONFIG['angle_max']))
    angle_bin = max(0, min(angle_bins - 1, angle_bin))

    power_step = (ACTION_CONFIG['power_max'] - ACTION_CONFIG['power_min']) / (power_bins - 1)
    power_bin = int(round((action['power'] - ACTION_CONFIG['power_min']) / power_step))
    power_bin = max(0, min(power_bins - 1, power_bin))

    weapon = int(action['weapon'])
    weapon = max(0, min(num_weapons - 1, weapon))

    return angle_bin * power_bins * num_weapons + power_bin * num_weapons + weapon


def get_total_actions():
    """Get total number of discrete actions for DQN."""
    return (ACTION_CONFIG['angle_bins'] *
            ACTION_CONFIG['power_bins'] *
            ACTION_CONFIG['num_weapons'])
