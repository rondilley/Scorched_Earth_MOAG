"""Physics system for gravity, wind, and collisions."""

import random
import math
from settings import GRAVITY, WIND_MIN, WIND_MAX, SCREEN_WIDTH, SCREEN_HEIGHT


class Physics:
    """Handles physics simulation for the game."""

    def __init__(self):
        """Initialize physics system."""
        self.gravity = GRAVITY
        self.wind = 0
        self.randomize_wind()

    def randomize_wind(self):
        """Set a new random wind value."""
        self.wind = random.uniform(WIND_MIN, WIND_MAX)

    def apply_to_projectile(self, projectile, dt):
        """Apply physics to a projectile.

        Args:
            projectile: Projectile object with x, y, vx, vy attributes
            dt: Delta time in seconds
        """
        # Apply wind to horizontal velocity
        projectile.vx += self.wind * 10 * dt

        # Apply gravity to vertical velocity
        projectile.vy += self.gravity * dt

        # Update position
        projectile.x += projectile.vx * dt
        projectile.y += projectile.vy * dt

    def check_terrain_collision(self, x, y, terrain):
        """Check if a point collides with terrain.

        Args:
            x, y: Point to check
            terrain: Terrain object

        Returns:
            True if collision detected
        """
        return terrain.check_collision(x, y)

    def check_bounds(self, x, y):
        """Check if a point is out of screen bounds.

        Args:
            x, y: Point to check

        Returns:
            True if out of bounds (left, right, or bottom)
        """
        # Allow going off the top
        if x < 0 or x > SCREEN_WIDTH:
            return True
        if y > SCREEN_HEIGHT + 100:  # Some buffer for explosions
            return True
        return False

    def check_tank_collision(self, x, y, tanks, exclude_tank=None):
        """Check if a point collides with any tank.

        Args:
            x, y: Point to check
            tanks: List of tank objects
            exclude_tank: Tank to exclude (usually the one that fired)

        Returns:
            Tank that was hit, or None
        """
        for tank in tanks:
            if tank == exclude_tank or not tank.alive:
                continue
            if tank.check_hit(x, y, 5):  # Small collision radius for direct hit
                return tank
        return None

    def calculate_trajectory(self, start_x, start_y, vx, vy, terrain, max_steps=1000):
        """Calculate the trajectory of a projectile (for AI prediction).

        Args:
            start_x, start_y: Starting position
            vx, vy: Initial velocity
            terrain: Terrain object
            max_steps: Maximum simulation steps

        Returns:
            List of (x, y) points along trajectory
        """
        points = []
        x, y = start_x, start_y
        dt = 0.016  # Simulate at ~60fps

        for _ in range(max_steps):
            points.append((x, y))

            # Apply physics
            vx += self.wind * 10 * dt
            vy += self.gravity * dt
            x += vx * dt
            y += vy * dt

            # Check for termination
            if self.check_bounds(x, y):
                break
            if terrain.check_collision(x, y):
                points.append((x, y))
                break

        return points

    def calculate_damage(self, distance, max_damage, explosion_radius):
        """Calculate damage based on distance from explosion center.

        Args:
            distance: Distance from explosion center
            max_damage: Maximum damage at center
            explosion_radius: Radius of explosion

        Returns:
            Damage value (0 if outside radius)
        """
        if distance >= explosion_radius:
            return 0

        # Linear falloff
        damage_ratio = 1 - (distance / explosion_radius)
        return int(max_damage * damage_ratio)

    def get_distance(self, x1, y1, x2, y2):
        """Calculate distance between two points."""
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)
