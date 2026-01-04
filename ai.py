"""AI opponent logic."""

import math
import random
from settings import (
    AIDifficulty, AI_SETTINGS, GRAVITY,
    ANGLE_MIN, ANGLE_MAX, POWER_MIN, POWER_MAX
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
