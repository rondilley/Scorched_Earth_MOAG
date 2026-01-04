"""Main game state management."""

import pygame
import random
from settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, GameState,
    MIN_PLAYERS, MAX_PLAYERS
)
from terrain import Terrain
from tank import Tank
from physics import Physics
from projectile import Projectile, Explosion
from weapons import get_weapon, get_weapon_count
from ai import AI
from ui import UI
from sound import SoundManager


class Game:
    """Main game class managing all game state."""

    def __init__(self, screen):
        """Initialize the game.

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        self.state = GameState.MENU
        self.ui = UI()
        self.sound = SoundManager()

        # Game objects
        self.terrain = None
        self.tanks = []
        self.projectiles = []
        self.explosions = []
        self.physics = Physics()
        self.ai_controllers = {}

        # Turn management
        self.current_player_index = 0
        self.turn_transition_timer = 0
        self.turn_transition_duration = 1.0

        # Player setup
        self.num_players = 2
        self.player_types = ['human', 'ai']  # 'human' or 'ai'

        # Input tracking
        self.keys_held = {}

    def handle_event(self, event):
        """Handle pygame events.

        Args:
            event: Pygame event
        """
        if event.type == pygame.KEYDOWN:
            self.keys_held[event.key] = True
            self._handle_keydown(event)
        elif event.type == pygame.KEYUP:
            self.keys_held[event.key] = False

    def _handle_keydown(self, event):
        """Handle key press events."""
        if self.state == GameState.MENU:
            self._handle_menu_input(event)
        elif self.state == GameState.SETUP:
            self._handle_setup_input(event)
        elif self.state == GameState.AIMING:
            self._handle_aiming_input(event)
        elif self.state == GameState.GAME_OVER:
            self._handle_game_over_input(event)

    def _handle_menu_input(self, event):
        """Handle input in menu state."""
        if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
            self.state = GameState.SETUP
            self.sound.play('menu_select')
        elif event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _handle_setup_input(self, event):
        """Handle input in setup state."""
        if event.key == pygame.K_LEFT:
            self.num_players = max(MIN_PLAYERS, self.num_players - 1)
            self._adjust_player_types()
            self.sound.play('menu_move')
        elif event.key == pygame.K_RIGHT:
            self.num_players = min(MAX_PLAYERS, self.num_players + 1)
            self._adjust_player_types()
            self.sound.play('menu_move')
        elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
            # Toggle player type
            player_idx = event.key - pygame.K_1
            if player_idx < self.num_players:
                if self.player_types[player_idx] == 'human':
                    self.player_types[player_idx] = 'ai'
                else:
                    self.player_types[player_idx] = 'human'
                self.sound.play('menu_select')
        elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
            self._start_game()
        elif event.key == pygame.K_ESCAPE:
            self.state = GameState.MENU
            self.sound.play('menu_move')

    def _handle_aiming_input(self, event):
        """Handle input during aiming phase."""
        tank = self._get_current_tank()
        if not tank or tank.is_ai:
            return

        if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            self._fire_projectile()
        elif event.key == pygame.K_TAB:
            # Next weapon
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                tank.current_weapon = (tank.current_weapon - 1) % get_weapon_count()
            else:
                tank.current_weapon = (tank.current_weapon + 1) % get_weapon_count()
        elif event.key == pygame.K_q:
            tank.current_weapon = (tank.current_weapon + 1) % get_weapon_count()
        elif event.key == pygame.K_e:
            tank.current_weapon = (tank.current_weapon - 1) % get_weapon_count()

    def _handle_game_over_input(self, event):
        """Handle input in game over state."""
        if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
            self.state = GameState.MENU
            self.sound.play('menu_select')
        elif event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _adjust_player_types(self):
        """Adjust player types list to match player count."""
        while len(self.player_types) < self.num_players:
            self.player_types.append('ai')
        self.player_types = self.player_types[:self.num_players]

    def _start_game(self):
        """Initialize and start a new game."""
        # Play game start sounds
        self.sound.play('game_start')
        self.sound.play_battle_music()

        # Create terrain
        self.terrain = Terrain()

        # Create tanks
        self.tanks = []
        self.ai_controllers = {}

        spacing = SCREEN_WIDTH // (self.num_players + 1)
        for i in range(self.num_players):
            x = spacing * (i + 1)
            is_ai = self.player_types[i] == 'ai'
            tank = Tank(x, 0, i, is_ai=is_ai)
            tank.set_position_on_terrain(self.terrain)
            self.tanks.append(tank)

            if is_ai:
                self.ai_controllers[i] = AI(tank)

        # Reset game state
        self.projectiles = []
        self.explosions = []
        self.current_player_index = 0
        self.physics.randomize_wind()

        self.state = GameState.AIMING

    def _get_current_tank(self):
        """Get the tank for the current player."""
        alive_tanks = [t for t in self.tanks if t.alive]
        if not alive_tanks:
            return None

        # Find the next alive tank from current index
        for _ in range(len(self.tanks)):
            tank = self.tanks[self.current_player_index]
            if tank.alive:
                return tank
            self.current_player_index = (self.current_player_index + 1) % len(self.tanks)

        return None

    def _fire_projectile(self):
        """Fire a projectile from the current tank."""
        tank = self._get_current_tank()
        if not tank:
            return

        # Get firing position and velocity
        x, y = tank.get_turret_end()
        vx, vy = tank.get_firing_velocity()

        # Create projectile
        weapon = get_weapon(tank.current_weapon)
        projectile = Projectile(x, y, vx, vy, weapon, tank)
        self.projectiles.append(projectile)

        # Play fire sound
        self.sound.play_fire(weapon.name)

        self.state = GameState.FIRING

    def _next_turn(self):
        """Advance to the next player's turn."""
        # Check for game over
        alive_tanks = [t for t in self.tanks if t.alive]
        if len(alive_tanks) <= 1:
            self.state = GameState.GAME_OVER
            self.sound.fade_out_music(1500)
            self.sound.play('victory')
            return

        # Find next alive player
        for _ in range(len(self.tanks)):
            self.current_player_index = (self.current_player_index + 1) % len(self.tanks)
            if self.tanks[self.current_player_index].alive:
                break

        # New wind for new turn
        self.physics.randomize_wind()
        self.sound.play('wind_change', volume_multiplier=0.5)

        # Reset AI controller for new turn
        current_tank = self._get_current_tank()
        if current_tank and current_tank.is_ai:
            if current_tank.player_id in self.ai_controllers:
                self.ai_controllers[current_tank.player_id].start_turn(
                    self.tanks, self.terrain, self.physics
                )

        self.state = GameState.TURN_TRANSITION
        self.turn_transition_timer = 0
        self.sound.play('turn_change')

    def update(self, dt):
        """Update game state.

        Args:
            dt: Delta time in seconds
        """
        # Handle music for menu states
        if self.state == GameState.MENU or self.state == GameState.SETUP:
            self.sound.play_menu_music()

        if self.state == GameState.AIMING:
            self._update_aiming(dt)
        elif self.state == GameState.FIRING:
            self._update_firing(dt)
        elif self.state == GameState.EXPLOSION:
            self._update_explosions(dt)
        elif self.state == GameState.TURN_TRANSITION:
            self._update_turn_transition(dt)

        # Always update tanks (for falling)
        for tank in self.tanks:
            tank.update(dt, self.terrain)

    def _update_aiming(self, dt):
        """Update during aiming phase."""
        tank = self._get_current_tank()
        if not tank:
            return

        if tank.is_ai:
            # Let AI take its turn
            ai = self.ai_controllers.get(tank.player_id)
            if ai:
                if ai.update(dt, self.tanks, self.terrain, self.physics):
                    self._fire_projectile()
        else:
            # Handle held keys for smooth aiming
            if self.keys_held.get(pygame.K_LEFT) or self.keys_held.get(pygame.K_a):
                tank.adjust_angle(1, dt)
            if self.keys_held.get(pygame.K_RIGHT) or self.keys_held.get(pygame.K_d):
                tank.adjust_angle(-1, dt)
            if self.keys_held.get(pygame.K_UP) or self.keys_held.get(pygame.K_w):
                tank.adjust_power(1, dt)
            if self.keys_held.get(pygame.K_DOWN) or self.keys_held.get(pygame.K_s):
                tank.adjust_power(-1, dt)

    def _update_firing(self, dt):
        """Update during firing phase."""
        projectiles_to_remove = []
        new_projectiles = []

        for projectile in self.projectiles:
            exploded, hit_pos, sub_projectiles = projectile.update(
                dt, self.physics, self.terrain, self.tanks
            )

            # Add any MIRV sub-projectiles
            if sub_projectiles:
                new_projectiles.extend(sub_projectiles)
                self.sound.play('mirv_split')

            if exploded and hit_pos:
                self._create_explosion(hit_pos[0], hit_pos[1], projectile.weapon)
                projectiles_to_remove.append(projectile)
            elif not projectile.active:
                projectiles_to_remove.append(projectile)

        # Update projectile list
        for p in projectiles_to_remove:
            if p in self.projectiles:
                self.projectiles.remove(p)
        self.projectiles.extend(new_projectiles)

        # Check if all projectiles are done
        if not self.projectiles:
            if self.explosions:
                self.state = GameState.EXPLOSION
            else:
                self._next_turn()

    def _update_explosions(self, dt):
        """Update explosion animations."""
        for explosion in self.explosions:
            explosion.update(dt)

        # Remove finished explosions
        self.explosions = [e for e in self.explosions if not e.finished]

        # Check if all explosions are done
        if not self.explosions:
            self._next_turn()

    def _update_turn_transition(self, dt):
        """Update turn transition timer."""
        self.turn_transition_timer += dt
        if self.turn_transition_timer >= self.turn_transition_duration:
            self.state = GameState.AIMING

            # Start AI turn if needed
            current_tank = self._get_current_tank()
            if current_tank and current_tank.is_ai:
                ai = self.ai_controllers.get(current_tank.player_id)
                if ai:
                    ai.start_turn(self.tanks, self.terrain, self.physics)

    def _create_explosion(self, x, y, weapon):
        """Create an explosion at the given position.

        Args:
            x, y: Explosion center
            weapon: Weapon that caused the explosion
        """
        # Create visual explosion
        explosion = Explosion(x, y, weapon.explosion_radius, weapon)
        self.explosions.append(explosion)

        # Play explosion sound based on size
        self.sound.play_explosion(weapon.explosion_radius)

        # Handle terrain modification
        if weapon.special == 'add_terrain':
            self.terrain.add_circle(x, y, weapon.explosion_radius)
        else:
            self.terrain.destroy_circle(x, y, weapon.explosion_radius)

        # Apply damage to tanks
        for tank in self.tanks:
            if not tank.alive:
                continue

            distance = self.physics.get_distance(x, y, tank.x, tank.y)
            damage = self.physics.calculate_damage(
                distance, weapon.damage, weapon.explosion_radius
            )
            if damage > 0:
                was_alive = tank.alive
                tank.take_damage(damage)
                # Play tank hit/destroy sound
                if was_alive:
                    self.sound.play_tank_hit(destroyed=not tank.alive)

    def render(self, screen):
        """Render the game.

        Args:
            screen: Pygame display surface
        """
        # Draw sky gradient
        self._render_sky(screen)

        if self.state == GameState.MENU:
            self.ui.render_menu(screen)
        elif self.state == GameState.SETUP:
            self.ui.render_setup(screen, self.num_players, self.player_types)
        elif self.state == GameState.GAME_OVER:
            # Render game state in background
            self._render_game(screen)
            self.ui.render_game_over(screen, self.tanks)
        else:
            self._render_game(screen)

    def _render_sky(self, screen):
        """Render sky gradient background."""
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            color = (
                int(COLORS['sky_top'][0] + (COLORS['sky_bottom'][0] - COLORS['sky_top'][0]) * ratio),
                int(COLORS['sky_top'][1] + (COLORS['sky_bottom'][1] - COLORS['sky_top'][1]) * ratio),
                int(COLORS['sky_top'][2] + (COLORS['sky_bottom'][2] - COLORS['sky_top'][2]) * ratio),
            )
            pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))

    def _render_game(self, screen):
        """Render the main game elements."""
        # Render terrain
        self.terrain.render(screen)

        # Render tanks
        current_tank = self._get_current_tank()
        for tank in self.tanks:
            tank.render(screen, tank == current_tank)
            tank.render_health_bar(screen)

        # Render projectiles
        for projectile in self.projectiles:
            projectile.render(screen)

        # Render explosions
        for explosion in self.explosions:
            explosion.render(screen)

        # Render HUD
        if current_tank:
            self.ui.render_hud(screen, current_tank, self.physics.wind)

        # Render turn transition
        if self.state == GameState.TURN_TRANSITION:
            self.ui.render_turn_transition(
                screen, current_tank, self.turn_transition_timer / self.turn_transition_duration
            )
