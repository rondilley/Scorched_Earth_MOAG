"""AI opponent logic."""

import os
import math
import random
from settings import (
    AIDifficulty, AI_SETTINGS, GRAVITY,
    ANGLE_MIN, ANGLE_MAX, POWER_MIN, POWER_MAX,
    SCREEN_WIDTH, SCREEN_HEIGHT, TANK_MAX_HEALTH,
    WIND_MIN, WIND_MAX, TERRAIN_MAX_HEIGHT
)
from weapons import get_weapon_count


class AI:
    """AI controller for a tank."""

    def __init__(self, tank, difficulty=AIDifficulty.MEDIUM):
        """Initialize AI controller.

        Args:
            tank: Tank this AI controls
            difficulty: Difficulty level
        """
        self.tank = tank
        self.difficulty = difficulty
        self.settings = AI_SETTINGS[difficulty]

        # Turn state
        self.thinking_timer = 0
        self.has_aimed = False
        self.target_tank = None
        self.target_angle = 0
        self.target_power = 0

    def start_turn(self, tanks, terrain, physics):
        """Called at the start of the AI's turn.

        Args:
            tanks: List of all tanks
            terrain: Terrain object
            physics: Physics object
        """
        self.thinking_timer = 0
        self.has_aimed = False

        # Choose a target (closest alive enemy)
        self.target_tank = self._choose_target(tanks)

        if self.target_tank:
            # Calculate ideal shot
            self.target_angle, self.target_power = self._calculate_shot(
                self.target_tank, terrain, physics
            )

            # Add error based on difficulty
            angle_error = random.uniform(
                -self.settings['angle_error'],
                self.settings['angle_error']
            )
            power_error = random.uniform(
                -self.settings['power_error'],
                self.settings['power_error']
            )

            self.target_angle = max(ANGLE_MIN, min(ANGLE_MAX, self.target_angle + angle_error))
            self.target_power = max(POWER_MIN, min(POWER_MAX, self.target_power + power_error))

            # Choose weapon based on distance
            self._choose_weapon(self.target_tank)

    def update(self, dt, tanks, terrain, physics):
        """Update AI state.

        Args:
            dt: Delta time
            tanks, terrain, physics: Game objects

        Returns:
            True if AI fires this frame
        """
        self.thinking_timer += dt

        # Wait for "thinking" time
        if self.thinking_timer < self.settings['reaction_time'] * 0.5:
            return False

        # Animate aiming
        if not self.has_aimed:
            aim_speed = 100 * dt

            # Adjust angle toward target
            angle_diff = self.target_angle - self.tank.angle
            if abs(angle_diff) > aim_speed:
                self.tank.angle += aim_speed if angle_diff > 0 else -aim_speed
            else:
                self.tank.angle = self.target_angle

            # Adjust power toward target
            power_diff = self.target_power - self.tank.power
            if abs(power_diff) > aim_speed * 3:
                self.tank.power += aim_speed * 3 if power_diff > 0 else -aim_speed * 3
            else:
                self.tank.power = self.target_power

            # Check if aiming is complete
            if (abs(self.tank.angle - self.target_angle) < 1 and
                abs(self.tank.power - self.target_power) < 5):
                self.has_aimed = True

            return False

        # Fire after reaction time
        if self.thinking_timer >= self.settings['reaction_time']:
            return True

        return False

    def _choose_target(self, tanks):
        """Choose a target tank.

        Args:
            tanks: List of all tanks

        Returns:
            Target tank or None
        """
        enemies = [t for t in tanks if t != self.tank and t.alive]
        if not enemies:
            return None

        # Choose closest enemy
        closest = None
        closest_dist = float('inf')

        for enemy in enemies:
            dist = abs(enemy.x - self.tank.x)
            if dist < closest_dist:
                closest_dist = dist
                closest = enemy

        return closest

    def _calculate_shot(self, target, terrain, physics):
        """Calculate angle and power to hit target.

        Args:
            target: Target tank
            terrain: Terrain object
            physics: Physics object

        Returns:
            Tuple of (angle, power)
        """
        dx = target.x - self.tank.x
        dy = target.y - self.tank.y

        distance = abs(dx)

        # Determine base angle (shoot toward target)
        if dx > 0:
            base_angle = 45  # Shooting right
        else:
            base_angle = 135  # Shooting left

        # Estimate power based on distance
        # Using simplified projectile motion formula
        # Range = v^2 * sin(2*theta) / g
        # For 45 degrees: Range = v^2 / g
        # So: v = sqrt(Range * g)

        angle_rad = math.radians(base_angle)
        sin_2theta = math.sin(2 * angle_rad)

        if abs(sin_2theta) > 0.1:
            # Solve for velocity
            estimated_power = math.sqrt(abs(distance * GRAVITY / sin_2theta))

            # Account for wind
            wind_factor = physics.wind * distance / 500
            if (dx > 0 and physics.wind < 0) or (dx < 0 and physics.wind > 0):
                # Wind against us
                estimated_power *= 1.1
            else:
                # Wind with us
                estimated_power *= 0.9

            # Account for height difference
            if dy > 0:  # Target is below us
                estimated_power *= 0.9
            elif dy < 0:  # Target is above us
                estimated_power *= 1.1
        else:
            estimated_power = (POWER_MIN + POWER_MAX) / 2

        # Clamp power
        estimated_power = max(POWER_MIN, min(POWER_MAX, estimated_power))

        return base_angle, estimated_power

    def _choose_weapon(self, target):
        """Choose an appropriate weapon.

        Args:
            target: Target tank
        """
        distance = abs(target.x - self.tank.x)

        # Simple weapon selection
        if distance > 400:
            # Far away - use standard or big bertha
            self.tank.current_weapon = random.choice([0, 1])
        elif distance > 200:
            # Medium range - maybe use nuke for big damage
            self.tank.current_weapon = random.choice([0, 1, 2])
        else:
            # Close range - careful with nuke!
            self.tank.current_weapon = random.choice([0, 1])

        # Small chance to use MIRV
        if random.random() < 0.1:
            self.tank.current_weapon = 4  # MIRV

        # Make sure weapon index is valid
        self.tank.current_weapon = self.tank.current_weapon % get_weapon_count()


class RLAgent:
    """RL-based AI controller using trained PPO or DQN models.

    Falls back to heuristic AI if model is not found.
    """

    def __init__(self, tank, model_path=None, algorithm='ppo', difficulty=AIDifficulty.MEDIUM):
        """Initialize RL agent.

        Args:
            tank: Tank this agent controls
            model_path: Path to trained model checkpoint
            algorithm: 'ppo' or 'dqn'
            difficulty: Fallback difficulty for heuristic AI
        """
        self.tank = tank
        self.algorithm = algorithm
        self.model = None
        self.device = None

        # Turn state
        self.thinking_timer = 0
        self.has_aimed = False
        self.target_angle = 0
        self.target_power = 0
        self.action_ready = False

        # Fallback AI
        self.fallback_ai = AI(tank, difficulty)

        # Try to load model
        if model_path:
            self._load_model(model_path)
        else:
            # Try default paths
            default_paths = [
                os.path.join('models', f'{algorithm}_best.pt'),
                os.path.join('models', f'{algorithm}_latest.pt'),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self._load_model(path)
                    break

    def _load_model(self, path):
        """Load trained model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        if not os.path.exists(path):
            print(f"Model not found at {path}, using fallback AI")
            return

        try:
            import torch
            from training.networks import ActorCritic, DQNNetwork

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(path, map_location=self.device)

            if self.algorithm == 'ppo':
                self.model = ActorCritic()
                self.model.load_state_dict(checkpoint['network_state_dict'])
            elif self.algorithm == 'dqn':
                self.model = DQNNetwork()
                self.model.load_state_dict(checkpoint['q_network_state_dict'])

            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded {self.algorithm.upper()} model from {path}")

        except ImportError as e:
            print(f"Could not import training modules: {e}")
            print("Using fallback AI")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback AI")
            self.model = None

    def _build_observation(self, tanks, terrain, physics):
        """Build observation dict from game state.

        Args:
            tanks: List of all tanks
            terrain: Terrain object
            physics: Physics object

        Returns:
            Observation dict compatible with training env
        """
        import numpy as np

        # Normalize terrain heights
        terrain_obs = np.array(terrain.heights, dtype=np.float32) / TERRAIN_MAX_HEIGHT

        # Player tank features (normalized)
        player_features = np.array([
            self.tank.x / SCREEN_WIDTH,
            self.tank.y / SCREEN_HEIGHT,
            self.tank.health / TANK_MAX_HEALTH,
            self.tank.angle / ANGLE_MAX,
            (self.tank.power - POWER_MIN) / (POWER_MAX - POWER_MIN),
            self.tank.current_weapon / 4.0,
        ], dtype=np.float32)

        # Opponent features
        opponent_features = []
        for t in tanks:
            if t != self.tank:
                opponent_features.extend([
                    t.x / SCREEN_WIDTH,
                    t.y / SCREEN_HEIGHT,
                    t.health / TANK_MAX_HEALTH,
                    float(t.alive),
                ])

        # Pad to max opponents (3)
        while len(opponent_features) < 12:
            opponent_features.extend([0.0, 0.0, 0.0, 0.0])
        opponent_obs = np.array(opponent_features[:12], dtype=np.float32)

        # Wind
        wind_obs = np.array([physics.wind / max(abs(WIND_MIN), abs(WIND_MAX))],
                          dtype=np.float32)

        return {
            'terrain': terrain_obs,
            'player_tank': player_features,
            'opponent_tanks': opponent_obs,
            'wind': wind_obs,
        }

    def start_turn(self, tanks, terrain, physics):
        """Called at the start of the agent's turn.

        Args:
            tanks: List of all tanks
            terrain: Terrain object
            physics: Physics object
        """
        self.thinking_timer = 0
        self.has_aimed = False
        self.action_ready = False

        if self.model is None:
            # Use fallback AI
            self.fallback_ai.start_turn(tanks, terrain, physics)
            self.target_angle = self.fallback_ai.target_angle
            self.target_power = self.fallback_ai.target_power
            return

        try:
            import torch
            from training.networks import obs_to_tensor
            from training.env import index_to_action

            # Build observation
            obs = self._build_observation(tanks, terrain, physics)
            obs_tensor = obs_to_tensor(obs, self.device)

            # Get action from model
            with torch.no_grad():
                if self.algorithm == 'ppo':
                    action, _, _ = self.model.get_action(obs_tensor, deterministic=True)
                    self.target_angle = action['angle'].item()
                    self.target_power = action['power'].item()
                    self.tank.current_weapon = action['weapon'].item()
                elif self.algorithm == 'dqn':
                    q_values = self.model(obs_tensor)
                    action_idx = q_values.argmax(dim=-1).item()
                    action = index_to_action(action_idx)
                    self.target_angle = action['angle']
                    self.target_power = action['power']
                    self.tank.current_weapon = action['weapon']

            # Clamp values
            self.target_angle = max(ANGLE_MIN, min(ANGLE_MAX, self.target_angle))
            self.target_power = max(POWER_MIN, min(POWER_MAX, self.target_power))

            # Fix direction: RL model was trained as player 0 (left side, shoots right)
            # If target is to the left but angle aims right (or vice versa), flip it
            target = self._find_target(tanks)
            if target:
                dx = target.x - self.tank.x
                # Angle 0-90 shoots right, 90-180 shoots left
                aims_right = self.target_angle < 90
                target_is_right = dx > 0
                if aims_right != target_is_right:
                    # Flip angle: mirror around 90 degrees
                    self.target_angle = 180 - self.target_angle

            # Exclude Dirt Ball (weapon 3) - it adds terrain instead of damaging
            # Dirt Ball is a strategic weapon that untrained RL models misuse
            if self.tank.current_weapon == 3:
                self.tank.current_weapon = 0  # Use Standard Shell instead
            self.tank.current_weapon = self.tank.current_weapon % get_weapon_count()

            self.action_ready = True

        except Exception as e:
            print(f"Error getting RL action: {e}")
            # Fallback
            self.fallback_ai.start_turn(tanks, terrain, physics)
            self.target_angle = self.fallback_ai.target_angle
            self.target_power = self.fallback_ai.target_power

    def _find_target(self, tanks):
        """Find closest alive enemy tank.

        Args:
            tanks: List of all tanks

        Returns:
            Target tank or None
        """
        enemies = [t for t in tanks if t != self.tank and t.alive]
        if not enemies:
            return None

        closest = None
        closest_dist = float('inf')
        for enemy in enemies:
            dist = abs(enemy.x - self.tank.x)
            if dist < closest_dist:
                closest_dist = dist
                closest = enemy
        return closest

    def update(self, dt, tanks, terrain, physics):
        """Update agent state.

        Args:
            dt: Delta time
            tanks, terrain, physics: Game objects

        Returns:
            True if agent fires this frame
        """
        if self.model is None:
            return self.fallback_ai.update(dt, tanks, terrain, physics)

        self.thinking_timer += dt

        # Brief thinking delay
        if self.thinking_timer < 0.3:
            return False

        # Animate aiming
        if not self.has_aimed:
            aim_speed = 150 * dt  # Slightly faster than heuristic AI

            # Adjust angle
            angle_diff = self.target_angle - self.tank.angle
            if abs(angle_diff) > aim_speed:
                self.tank.angle += aim_speed if angle_diff > 0 else -aim_speed
            else:
                self.tank.angle = self.target_angle

            # Adjust power
            power_diff = self.target_power - self.tank.power
            if abs(power_diff) > aim_speed * 4:
                self.tank.power += aim_speed * 4 if power_diff > 0 else -aim_speed * 4
            else:
                self.tank.power = self.target_power

            # Check if aiming complete
            if (abs(self.tank.angle - self.target_angle) < 1 and
                abs(self.tank.power - self.target_power) < 5):
                self.has_aimed = True

            return False

        # Fire after aiming
        if self.thinking_timer >= 0.8:
            return True

        return False
