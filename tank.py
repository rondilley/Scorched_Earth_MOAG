"""Tank class for player and AI tanks."""

import pygame
import math
from settings import (
    TANK_WIDTH, TANK_HEIGHT, TANK_TURRET_LENGTH, TANK_MAX_HEALTH,
    ANGLE_MIN, ANGLE_MAX, ANGLE_SPEED,
    POWER_MIN, POWER_MAX, POWER_SPEED,
    TANK_COLORS, COLORS, SCREEN_HEIGHT
)


class Tank:
    """Represents a tank in the game."""

    def __init__(self, x, y, player_id, is_ai=False, color=None):
        """Initialize the tank.

        Args:
            x: X position
            y: Y position (bottom of tank)
            player_id: Player number (0-3)
            is_ai: Whether this tank is AI controlled
            color: Override color (or use default for player_id)
        """
        self.x = x
        self.y = y
        self.player_id = player_id
        self.is_ai = is_ai
        self.color = color or TANK_COLORS[player_id % len(TANK_COLORS)]

        # Combat stats
        self.health = TANK_MAX_HEALTH
        self.alive = True

        # Aiming
        self.angle = 45 if player_id % 2 == 0 else 135  # Face toward center
        self.power = (POWER_MIN + POWER_MAX) // 2

        # Current weapon index
        self.current_weapon = 0

        # Falling state
        self.is_falling = False
        self.fall_velocity = 0

    def update(self, dt, terrain):
        """Update tank state.

        Args:
            dt: Delta time in seconds
            terrain: Terrain object for collision
        """
        if not self.alive:
            return

        # Handle falling
        if self.is_falling:
            self.fall_velocity += 500 * dt  # Gravity
            self.y += self.fall_velocity * dt

            # Check if we've landed
            surface_y = terrain.get_surface_y(self.x)
            if self.y >= surface_y:
                self.y = surface_y
                self.is_falling = False
                self.fall_velocity = 0

                # Take fall damage if fell far
                if self.fall_velocity > 200:
                    fall_damage = int(self.fall_velocity / 10)
                    self.take_damage(fall_damage)
        else:
            # Check if ground was removed
            surface_y = terrain.get_surface_y(self.x)
            if self.y < surface_y - 2:
                self.is_falling = True
                self.fall_velocity = 0

    def adjust_angle(self, direction, dt):
        """Adjust turret angle.

        Args:
            direction: 1 for increase (counter-clockwise), -1 for decrease
            dt: Delta time
        """
        self.angle += direction * ANGLE_SPEED * dt
        self.angle = max(ANGLE_MIN, min(ANGLE_MAX, self.angle))

    def adjust_power(self, direction, dt):
        """Adjust firing power.

        Args:
            direction: 1 for increase, -1 for decrease
            dt: Delta time
        """
        self.power += direction * POWER_SPEED * dt
        self.power = max(POWER_MIN, min(POWER_MAX, self.power))

    def get_turret_end(self):
        """Get the position of the turret end (where projectile spawns)."""
        angle_rad = math.radians(self.angle)
        turret_x = self.x + math.cos(angle_rad) * TANK_TURRET_LENGTH
        turret_y = self.y - TANK_HEIGHT - math.sin(angle_rad) * TANK_TURRET_LENGTH
        return turret_x, turret_y

    def get_firing_velocity(self):
        """Get the initial velocity components for a fired projectile."""
        angle_rad = math.radians(self.angle)
        vx = math.cos(angle_rad) * self.power
        vy = -math.sin(angle_rad) * self.power  # Negative because y increases downward
        return vx, vy

    def take_damage(self, damage):
        """Apply damage to the tank.

        Args:
            damage: Amount of damage to apply
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def check_hit(self, x, y, radius):
        """Check if an explosion hits this tank.

        Args:
            x, y: Center of explosion
            radius: Explosion radius

        Returns:
            True if tank is hit
        """
        if not self.alive:
            return False

        # Simple box collision with explosion circle
        tank_left = self.x - TANK_WIDTH // 2
        tank_right = self.x + TANK_WIDTH // 2
        tank_top = self.y - TANK_HEIGHT
        tank_bottom = self.y

        # Find closest point on tank to explosion center
        closest_x = max(tank_left, min(x, tank_right))
        closest_y = max(tank_top, min(y, tank_bottom))

        # Check distance
        dx = x - closest_x
        dy = y - closest_y
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= radius

    def render(self, screen, is_current=False):
        """Render the tank.

        Args:
            screen: Pygame surface to draw on
            is_current: Whether this is the current player's turn
        """
        if not self.alive:
            return

        # Tank body (rectangle)
        body_rect = pygame.Rect(
            self.x - TANK_WIDTH // 2,
            self.y - TANK_HEIGHT,
            TANK_WIDTH,
            TANK_HEIGHT
        )
        pygame.draw.rect(screen, self.color, body_rect)

        # Tank body outline
        outline_color = (
            min(255, self.color[0] + 50),
            min(255, self.color[1] + 50),
            min(255, self.color[2] + 50)
        )
        pygame.draw.rect(screen, outline_color, body_rect, 2)

        # Tank dome (semicircle on top)
        dome_rect = pygame.Rect(
            self.x - TANK_WIDTH // 4,
            self.y - TANK_HEIGHT - TANK_WIDTH // 4,
            TANK_WIDTH // 2,
            TANK_WIDTH // 2
        )
        pygame.draw.ellipse(screen, self.color, dome_rect)
        pygame.draw.ellipse(screen, outline_color, dome_rect, 2)

        # Turret
        turret_start = (self.x, self.y - TANK_HEIGHT)
        turret_end = self.get_turret_end()
        pygame.draw.line(screen, outline_color, turret_start, turret_end, 4)
        pygame.draw.line(screen, self.color, turret_start, turret_end, 2)

        # Draw current player indicator
        if is_current:
            indicator_y = self.y - TANK_HEIGHT - 30
            # Arrow pointing down
            pygame.draw.polygon(screen, COLORS['text'], [
                (self.x, indicator_y + 10),
                (self.x - 8, indicator_y),
                (self.x + 8, indicator_y)
            ])

    def render_health_bar(self, screen):
        """Render health bar above tank."""
        if not self.alive:
            return

        bar_width = TANK_WIDTH + 10
        bar_height = 6
        bar_x = self.x - bar_width // 2
        bar_y = self.y - TANK_HEIGHT - 20

        # Background
        pygame.draw.rect(screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))

        # Health fill
        health_ratio = self.health / TANK_MAX_HEALTH
        health_width = int(bar_width * health_ratio)
        health_color = COLORS['health_bar_fg'] if health_ratio > 0.3 else COLORS['health_bar_low']
        pygame.draw.rect(screen, health_color,
                        (bar_x, bar_y, health_width, bar_height))

        # Border
        pygame.draw.rect(screen, COLORS['text'],
                        (bar_x, bar_y, bar_width, bar_height), 1)

    def set_position_on_terrain(self, terrain):
        """Position tank on terrain surface at current x."""
        self.y = terrain.get_surface_y(self.x)
