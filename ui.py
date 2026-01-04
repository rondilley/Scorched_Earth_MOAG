"""UI rendering - HUD, menus, and overlays."""

import pygame
import math
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, TANK_COLORS,
    ANGLE_MIN, ANGLE_MAX, POWER_MIN, POWER_MAX,
    MIN_PLAYERS, MAX_PLAYERS
)
from weapons import get_weapon


class UI:
    """Handles all UI rendering."""

    def __init__(self):
        """Initialize UI system."""
        pygame.font.init()

        # Fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

    def render_menu(self, screen):
        """Render main menu.

        Args:
            screen: Pygame surface
        """
        # Title
        title = self.font_large.render("SCORCHED EARTH", True, COLORS['text'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        screen.blit(title, title_rect)

        subtitle = self.font_medium.render("M O D E R N", True, COLORS['menu_text'])
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 200))
        screen.blit(subtitle, subtitle_rect)

        # Instructions
        instructions = [
            "Press SPACE or ENTER to start",
            "",
            "Controls:",
            "LEFT/RIGHT or A/D - Aim",
            "UP/DOWN or W/S - Power",
            "SPACE - Fire",
            "TAB or Q/E - Change Weapon",
            "",
            "ESC - Quit"
        ]

        y = 300
        for line in instructions:
            if line:
                text = self.font_small.render(line, True, COLORS['menu_text'])
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y))
                screen.blit(text, text_rect)
            y += 30

    def render_setup(self, screen, num_players, player_types):
        """Render player setup screen.

        Args:
            screen: Pygame surface
            num_players: Number of players
            player_types: List of 'human' or 'ai' for each player
        """
        # Title
        title = self.font_large.render("GAME SETUP", True, COLORS['text'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(title, title_rect)

        # Player count
        count_text = f"< Players: {num_players} >"
        count = self.font_medium.render(count_text, True, COLORS['text'])
        count_rect = count.get_rect(center=(SCREEN_WIDTH // 2, 180))
        screen.blit(count, count_rect)

        # Player type selection
        y = 260
        for i in range(num_players):
            color = TANK_COLORS[i]
            type_str = player_types[i].upper()

            # Player indicator
            pygame.draw.rect(screen, color, (SCREEN_WIDTH // 2 - 150, y - 10, 20, 20))

            # Player text
            text = f"Player {i + 1}: {type_str}"
            player_text = self.font_medium.render(text, True, COLORS['text'])
            screen.blit(player_text, (SCREEN_WIDTH // 2 - 120, y - 10))

            # Instructions
            key_text = f"(Press {i + 1} to toggle)"
            key = self.font_small.render(key_text, True, COLORS['menu_text'])
            screen.blit(key, (SCREEN_WIDTH // 2 + 100, y - 5))

            y += 50

        # Start instruction
        start_text = self.font_medium.render("Press SPACE to start", True, COLORS['text'])
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100))
        screen.blit(start_text, start_rect)

        back_text = self.font_small.render("ESC - Back to menu", True, COLORS['menu_text'])
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60))
        screen.blit(back_text, back_rect)

    def render_hud(self, screen, tank, wind):
        """Render the heads-up display.

        Args:
            screen: Pygame surface
            tank: Current player's tank
            wind: Current wind value
        """
        # HUD background
        hud_height = 80
        hud_surface = pygame.Surface((SCREEN_WIDTH, hud_height), pygame.SRCALPHA)
        hud_surface.fill((20, 20, 40, 200))
        screen.blit(hud_surface, (0, 0))

        # Player indicator
        player_text = f"Player {tank.player_id + 1}"
        if tank.is_ai:
            player_text += " (AI)"
        player_label = self.font_medium.render(player_text, True, tank.color)
        screen.blit(player_label, (20, 10))

        # Weapon
        weapon = get_weapon(tank.current_weapon)
        weapon_text = f"Weapon: {weapon.name}"
        weapon_label = self.font_small.render(weapon_text, True, COLORS['text'])
        screen.blit(weapon_label, (20, 45))

        # Angle gauge
        self._render_gauge(
            screen, 250, 15, 150, 20,
            tank.angle, ANGLE_MIN, ANGLE_MAX,
            f"Angle: {int(tank.angle)}°",
            COLORS['text']
        )

        # Power gauge
        self._render_gauge(
            screen, 250, 45, 150, 20,
            tank.power, POWER_MIN, POWER_MAX,
            f"Power: {int(tank.power)}",
            COLORS['power_bar']
        )

        # Wind indicator
        self._render_wind(screen, SCREEN_WIDTH - 150, 30, wind)

    def _render_gauge(self, screen, x, y, width, height, value, min_val, max_val, label, color):
        """Render a gauge bar.

        Args:
            screen: Pygame surface
            x, y: Position
            width, height: Size
            value: Current value
            min_val, max_val: Value range
            label: Text label
            color: Fill color
        """
        # Background
        pygame.draw.rect(screen, COLORS['health_bar_bg'], (x, y, width, height))

        # Fill
        ratio = (value - min_val) / (max_val - min_val)
        fill_width = int(width * ratio)
        pygame.draw.rect(screen, color, (x, y, fill_width, height))

        # Border
        pygame.draw.rect(screen, COLORS['text'], (x, y, width, height), 2)

        # Label
        text = self.font_small.render(label, True, COLORS['text'])
        screen.blit(text, (x + width + 10, y))

    def _render_wind(self, screen, x, y, wind):
        """Render wind indicator.

        Args:
            screen: Pygame surface
            x, y: Center position
            wind: Wind value (-10 to 10)
        """
        # Label
        label = self.font_small.render("Wind", True, COLORS['text'])
        screen.blit(label, (x - 20, y - 25))

        # Arrow
        arrow_length = abs(wind) * 8
        if arrow_length > 5:
            # Arrow body
            if wind > 0:
                # Wind blowing right
                start = (x - arrow_length // 2, y)
                end = (x + arrow_length // 2, y)
                arrow_points = [
                    end,
                    (end[0] - 10, y - 8),
                    (end[0] - 10, y + 8)
                ]
            else:
                # Wind blowing left
                start = (x + arrow_length // 2, y)
                end = (x - arrow_length // 2, y)
                arrow_points = [
                    end,
                    (end[0] + 10, y - 8),
                    (end[0] + 10, y + 8)
                ]

            pygame.draw.line(screen, COLORS['wind_arrow'], start, end, 3)
            pygame.draw.polygon(screen, COLORS['wind_arrow'], arrow_points)
        else:
            # No wind - draw a circle
            pygame.draw.circle(screen, COLORS['wind_arrow'], (x, y), 5)

        # Wind value
        value_text = f"{wind:+.1f}"
        value = self.font_small.render(value_text, True, COLORS['text'])
        value_rect = value.get_rect(center=(x, y + 20))
        screen.blit(value, value_rect)

    def render_turn_transition(self, screen, tank, progress):
        """Render turn transition overlay.

        Args:
            screen: Pygame surface
            tank: Tank whose turn it is
            progress: Animation progress (0-1)
        """
        if not tank:
            return

        # Fade overlay
        alpha = int(150 * (1 - abs(progress - 0.5) * 2))
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, alpha))
        screen.blit(overlay, (0, 0))

        # Player text
        if progress > 0.2 and progress < 0.8:
            player_text = f"Player {tank.player_id + 1}'s Turn"
            if tank.is_ai:
                player_text = f"Player {tank.player_id + 1} (AI)"

            text = self.font_large.render(player_text, True, tank.color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

            # Add shadow
            shadow = self.font_large.render(player_text, True, COLORS['text_shadow'])
            shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH // 2 + 3, SCREEN_HEIGHT // 2 + 3))
            screen.blit(shadow, shadow_rect)
            screen.blit(text, text_rect)

    def render_game_over(self, screen, tanks):
        """Render game over screen.

        Args:
            screen: Pygame surface
            tanks: List of all tanks
        """
        # Darken background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        # Find winner
        winner = None
        for tank in tanks:
            if tank.alive:
                winner = tank
                break

        # Title
        title = self.font_large.render("GAME OVER", True, COLORS['text'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
        screen.blit(title, title_rect)

        # Winner announcement
        if winner:
            winner_text = f"Player {winner.player_id + 1} Wins!"
            text = self.font_large.render(winner_text, True, winner.color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 300))
            screen.blit(text, text_rect)
        else:
            draw_text = "It's a Draw!"
            text = self.font_large.render(draw_text, True, COLORS['text'])
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 300))
            screen.blit(text, text_rect)

        # Instructions
        restart = self.font_medium.render("Press SPACE to return to menu", True, COLORS['menu_text'])
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, 450))
        screen.blit(restart, restart_rect)

        quit_text = self.font_small.render("ESC - Quit", True, COLORS['menu_text'])
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH // 2, 500))
        screen.blit(quit_text, quit_rect)
