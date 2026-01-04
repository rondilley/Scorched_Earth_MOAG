#!/usr/bin/env python3
"""Scorched Earth Modern - Main Entry Point"""

import pygame
import sys
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TITLE, COLORS
from game import Game


def main():
    """Main entry point for the game."""
    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()

    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)

    # Create clock for FPS control
    clock = pygame.time.Clock()

    # Create game instance
    game = Game(screen)

    # Main game loop
    running = True
    while running:
        # Calculate delta time
        dt = clock.tick(FPS) / 1000.0  # Convert to seconds

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(event)

        # Update game state
        game.update(dt)

        # Render
        game.render(screen)

        # Flip the display
        pygame.display.flip()

    # Clean up
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
