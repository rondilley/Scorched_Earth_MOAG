"""Projectile class and explosion effects."""

import pygame
import math
import random
from settings import COLORS, SCREEN_WIDTH, SCREEN_HEIGHT


class Projectile:
    """A projectile fired from a tank."""

    def __init__(self, x, y, vx, vy, weapon, owner):
        """Initialize projectile.

        Args:
            x, y: Starting position
            vx, vy: Initial velocity
            weapon: Weapon object
            owner: Tank that fired this projectile
        """
        self.x = x
        self.y = y
        self.prev_x = x  # Previous position for collision detection
        self.prev_y = y
        self.vx = vx
        self.vy = vy
        self.weapon = weapon
        self.owner = owner

        # Trail history
        self.trail = []
        self.trail_max_length = 50

        # State
        self.active = True
        self.has_split = False  # For MIRV
        self.apex_reached = False  # For MIRV timing

    def update(self, dt, physics, terrain, tanks):
        """Update projectile position and check collisions.

        Args:
            dt: Delta time
            physics: Physics object
            terrain: Terrain object
            tanks: List of tanks

        Returns:
            Tuple of (exploded, hit_position, sub_projectiles)
        """
        if not self.active:
            return False, None, []

        # Store previous position for collision detection
        self.prev_x = self.x
        self.prev_y = self.y

        # Store trail position
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_max_length:
            self.trail.pop(0)

        # Track if we're going up or down for MIRV
        old_vy = self.vy

        # Apply physics
        physics.apply_to_projectile(self, dt)

        # Check for MIRV split at apex
        if self.weapon.special == 'mirv' and not self.has_split:
            if old_vy < 0 and self.vy >= 0:  # Just passed apex
                self.apex_reached = True
            if self.apex_reached and self.vy > 50:  # Split shortly after apex
                self.has_split = True
                self.active = False
                return False, None, self._create_mirv_projectiles()

        # Check for out of bounds
        if physics.check_bounds(self.x, self.y):
            self.active = False
            return False, None, []

        # Check terrain collision along the path (sub-stepping)
        hit_pos = self._check_terrain_along_path(terrain)
        if hit_pos:
            self.active = False
            return True, hit_pos, []

        # Check tank collision along the path
        hit_result = self._check_tank_along_path(tanks, physics)
        if hit_result:
            self.active = False
            return True, hit_result, []

        return False, None, []

    def _check_terrain_along_path(self, terrain):
        """Check terrain collision along the path from prev to current position."""
        # Number of steps based on distance traveled
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        distance = math.sqrt(dx * dx + dy * dy)
        steps = max(1, int(distance / 5))  # Check every 5 pixels

        for i in range(steps + 1):
            t = i / steps
            check_x = self.prev_x + dx * t
            check_y = self.prev_y + dy * t
            if terrain.check_collision(check_x, check_y):
                return (check_x, check_y)
        return None

    def _check_tank_along_path(self, tanks, physics):
        """Check tank collision along the path from prev to current position."""
        dx = self.x - self.prev_x
        dy = self.y - self.prev_y
        distance = math.sqrt(dx * dx + dy * dy)
        steps = max(1, int(distance / 3))  # Check every 3 pixels for tanks

        for i in range(steps + 1):
            t = i / steps
            check_x = self.prev_x + dx * t
            check_y = self.prev_y + dy * t

            for tank in tanks:
                if tank == self.owner or not tank.alive:
                    continue
                if tank.check_hit(check_x, check_y, 3):
                    return (check_x, check_y)
        return None

    def _create_mirv_projectiles(self):
        """Create sub-projectiles for MIRV weapon."""
        from weapons import Weapon, WeaponType

        sub_projectiles = []
        num_subs = 5

        # Create a simpler weapon for sub-projectiles
        sub_weapon = Weapon(
            name="MIRV Bomb",
            weapon_type=WeaponType.STANDARD,
            damage=self.weapon.damage,
            explosion_radius=self.weapon.explosion_radius,
            color=self.weapon.color
        )

        for i in range(num_subs):
            # Spread horizontally
            spread = (i - num_subs // 2) * 30
            sub = Projectile(
                self.x + spread * 0.5,
                self.y,
                self.vx + spread,
                self.vy,
                sub_weapon,
                self.owner
            )
            sub_projectiles.append(sub)

        return sub_projectiles

    def render(self, screen):
        """Render projectile and trail."""
        if not self.active:
            return

        # Draw trail
        if len(self.trail) > 1:
            for i, (tx, ty) in enumerate(self.trail):
                alpha = int(255 * i / len(self.trail) * 0.5)
                size = max(1, int(3 * i / len(self.trail)))
                trail_color = (
                    min(255, COLORS['trail'][0]),
                    min(255, COLORS['trail'][1]),
                    min(255, COLORS['trail'][2])
                )
                pygame.draw.circle(screen, trail_color, (int(tx), int(ty)), size)

        # Draw projectile
        pygame.draw.circle(screen, self.weapon.color, (int(self.x), int(self.y)), 4)
        pygame.draw.circle(screen, COLORS['projectile'], (int(self.x), int(self.y)), 2)


class Explosion:
    """Visual explosion effect."""

    def __init__(self, x, y, radius, weapon):
        """Initialize explosion.

        Args:
            x, y: Center position
            radius: Explosion radius
            weapon: Weapon that caused the explosion
        """
        self.x = x
        self.y = y
        self.max_radius = radius
        self.weapon = weapon

        # Animation state
        self.current_radius = 0
        self.phase = 'expand'  # 'expand', 'hold', 'fade'
        self.alpha = 255
        self.expand_speed = radius * 8  # Expand quickly
        self.hold_time = 0.1
        self.hold_timer = 0
        self.fade_speed = 500

        # Particles
        self.particles = []
        self._create_particles()

        self.finished = False

    def _create_particles(self):
        """Create explosion particles."""
        num_particles = int(self.max_radius * 0.5)
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            self.particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 50,  # Bias upward
                'size': random.randint(2, 5),
                'life': random.uniform(0.3, 0.8),
                'color': random.choice([
                    COLORS['explosion_outer'],
                    COLORS['explosion_inner'],
                    COLORS['explosion_flash']
                ])
            })

    def update(self, dt):
        """Update explosion animation."""
        if self.finished:
            return

        # Update main explosion
        if self.phase == 'expand':
            self.current_radius += self.expand_speed * dt
            if self.current_radius >= self.max_radius:
                self.current_radius = self.max_radius
                self.phase = 'hold'
        elif self.phase == 'hold':
            self.hold_timer += dt
            if self.hold_timer >= self.hold_time:
                self.phase = 'fade'
        elif self.phase == 'fade':
            self.alpha -= self.fade_speed * dt
            if self.alpha <= 0:
                self.alpha = 0
                self.finished = True

        # Update particles
        for p in self.particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 200 * dt  # Gravity
            p['life'] -= dt
            p['size'] = max(1, p['size'] - 2 * dt)

        # Remove dead particles
        self.particles = [p for p in self.particles if p['life'] > 0]

    def render(self, screen):
        """Render explosion."""
        if self.finished:
            return

        # Create a surface for the explosion with alpha
        if self.current_radius > 0:
            # Draw outer explosion
            outer_color = (*COLORS['explosion_outer'], int(self.alpha * 0.6))
            inner_color = (*COLORS['explosion_inner'], int(self.alpha * 0.8))
            flash_color = (*COLORS['explosion_flash'], int(self.alpha))

            # We need a separate surface for alpha blending
            explosion_surf = pygame.Surface((self.max_radius * 2 + 10, self.max_radius * 2 + 10), pygame.SRCALPHA)

            center = (self.max_radius + 5, self.max_radius + 5)

            # Outer ring
            if self.current_radius > 5:
                pygame.draw.circle(explosion_surf, outer_color[:3], center, int(self.current_radius))

            # Inner glow
            inner_radius = int(self.current_radius * 0.6)
            if inner_radius > 3:
                pygame.draw.circle(explosion_surf, inner_color[:3], center, inner_radius)

            # Center flash
            flash_radius = int(self.current_radius * 0.3)
            if flash_radius > 2:
                pygame.draw.circle(explosion_surf, flash_color[:3], center, flash_radius)

            # Apply alpha
            explosion_surf.set_alpha(int(self.alpha))

            # Blit to screen
            screen.blit(explosion_surf, (self.x - self.max_radius - 5, self.y - self.max_radius - 5))

        # Draw particles
        for p in self.particles:
            if p['life'] > 0:
                alpha = int(255 * p['life'])
                pygame.draw.circle(screen, p['color'],
                                  (int(p['x']), int(p['y'])),
                                  int(p['size']))
