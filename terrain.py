"""Terrain generation and destruction system."""

import pygame
import random
import math
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLORS,
    TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT, TERRAIN_ROUGHNESS
)


class Terrain:
    """Manages the destructible terrain."""

    def __init__(self):
        """Initialize terrain."""
        self.heights = []  # Height of terrain at each x position
        self.surface = None  # Pre-rendered terrain surface
        self.generate()

    def generate(self):
        """Generate terrain using midpoint displacement algorithm."""
        # Initialize with random endpoints
        self.heights = [0] * SCREEN_WIDTH

        # Start with two random heights
        self.heights[0] = random.randint(TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT)
        self.heights[-1] = random.randint(TERRAIN_MIN_HEIGHT, TERRAIN_MAX_HEIGHT)

        # Midpoint displacement
        self._midpoint_displacement(0, SCREEN_WIDTH - 1, TERRAIN_ROUGHNESS)

        # Smooth the terrain slightly
        self._smooth(2)

        # Create the rendered surface
        self._render_surface()

    def _midpoint_displacement(self, left, right, roughness):
        """Recursively apply midpoint displacement."""
        if right - left < 2:
            return

        mid = (left + right) // 2
        avg = (self.heights[left] + self.heights[right]) / 2

        # Add random displacement based on distance and roughness
        displacement = (right - left) * roughness * random.uniform(-0.5, 0.5)
        self.heights[mid] = int(avg + displacement)

        # Clamp to valid range
        self.heights[mid] = max(TERRAIN_MIN_HEIGHT,
                                min(TERRAIN_MAX_HEIGHT, self.heights[mid]))

        # Recurse on both halves
        self._midpoint_displacement(left, mid, roughness * 0.9)
        self._midpoint_displacement(mid, right, roughness * 0.9)

    def _smooth(self, iterations):
        """Smooth the terrain with averaging."""
        for _ in range(iterations):
            new_heights = self.heights.copy()
            for x in range(1, SCREEN_WIDTH - 1):
                new_heights[x] = (self.heights[x-1] + self.heights[x] + self.heights[x+1]) // 3
            self.heights = new_heights

    def _render_surface(self):
        """Pre-render the terrain to a surface for efficient drawing."""
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        # Draw terrain column by column
        for x in range(SCREEN_WIDTH):
            height = self.heights[x]
            terrain_top = SCREEN_HEIGHT - height

            # Draw grass layer (top few pixels)
            grass_height = 3
            pygame.draw.line(
                self.surface,
                COLORS['terrain_grass'],
                (x, terrain_top),
                (x, terrain_top + grass_height)
            )

            # Draw terrain gradient
            for y in range(terrain_top + grass_height, SCREEN_HEIGHT):
                # Calculate gradient color based on depth
                depth_ratio = (y - terrain_top) / height if height > 0 else 0
                color = self._lerp_color(
                    COLORS['terrain_top'],
                    COLORS['terrain_bottom'],
                    depth_ratio
                )
                self.surface.set_at((x, y), color)

    def _lerp_color(self, color1, color2, t):
        """Linear interpolate between two colors."""
        t = max(0, min(1, t))
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t)
        )

    def get_height_at(self, x):
        """Get terrain height at given x position."""
        x = int(x)
        if 0 <= x < SCREEN_WIDTH:
            return self.heights[x]
        return 0

    def get_surface_y(self, x):
        """Get the Y coordinate of the terrain surface at x."""
        return SCREEN_HEIGHT - self.get_height_at(x)

    def destroy_circle(self, center_x, center_y, radius):
        """Remove terrain in a circular area (explosion crater)."""
        center_x = int(center_x)
        center_y = int(center_y)
        radius = int(radius)

        modified = False

        for x in range(max(0, center_x - radius), min(SCREEN_WIDTH, center_x + radius + 1)):
            dx = x - center_x
            # Calculate the vertical half-height of the circle at this x
            if abs(dx) > radius:
                continue

            half_height = math.sqrt(radius * radius - dx * dx)
            circle_bottom_y = center_y + half_height  # Bottom of circle at this x

            # Current terrain surface (y coordinate, smaller = higher on screen)
            current_surface_y = SCREEN_HEIGHT - self.heights[x]

            # If terrain surface is above the bottom of the circle, carve it out
            if current_surface_y < circle_bottom_y:
                # New surface is at the bottom of the circle
                new_surface_y = circle_bottom_y
                new_height = SCREEN_HEIGHT - new_surface_y

                if new_height < self.heights[x]:
                    self.heights[x] = max(0, int(new_height))
                    modified = True

        if modified:
            self._render_surface()

        return modified

    def add_circle(self, center_x, center_y, radius):
        """Add terrain in a circular area (dirt ball)."""
        center_x = int(center_x)
        center_y = int(center_y)
        radius = int(radius)

        for x in range(max(0, center_x - radius), min(SCREEN_WIDTH, center_x + radius + 1)):
            dx = x - center_x
            # Height of circle at this x
            circle_height = int(math.sqrt(max(0, radius * radius - dx * dx)))

            # Add to terrain height
            terrain_surface_y = SCREEN_HEIGHT - self.heights[x]
            new_surface_y = min(terrain_surface_y, center_y - circle_height)
            self.heights[x] = SCREEN_HEIGHT - new_surface_y

            # Clamp height
            self.heights[x] = min(self.heights[x], SCREEN_HEIGHT - 50)

        self._render_surface()

    def check_collision(self, x, y):
        """Check if a point is inside terrain."""
        x = int(x)
        if x < 0 or x >= SCREEN_WIDTH:
            return False
        terrain_y = SCREEN_HEIGHT - self.heights[x]
        return y >= terrain_y

    def render(self, screen):
        """Render the terrain to the screen."""
        if self.surface:
            screen.blit(self.surface, (0, 0))

    def settle_objects(self, objects):
        """Make objects fall if terrain below them is destroyed.

        Returns list of objects that need to fall.
        """
        falling = []
        for obj in objects:
            if hasattr(obj, 'x') and hasattr(obj, 'y'):
                surface_y = self.get_surface_y(obj.x)
                if obj.y < surface_y - 5:  # Object is floating
                    falling.append(obj)
        return falling
