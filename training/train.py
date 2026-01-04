#!/usr/bin/env python3
"""Training script for Scorched Earth RL agents.

Usage:
    python -m training.train --algo ppo --total-steps 1000000
    python -m training.train --algo dqn --opponent heuristic
    python -m training.train --algo llm-gen --num-examples 100
"""

import argparse
import os
import sys
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Scorched Earth RL agents',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm selection
    parser.add_argument(
        '--algo', type=str, default='ppo',
        choices=['ppo', 'dqn', 'llm-gen'],
        help='Training algorithm (llm-gen for offline data generation)'
    )

    # Environment settings
    parser.add_argument(
        '--num-players', type=int, default=2,
        help='Number of players (2-4)'
    )
    parser.add_argument(
        '--opponent', type=str, default='heuristic',
        choices=['heuristic', 'self', 'random'],
        help='Opponent type'
    )
    parser.add_argument(
        '--opponent-difficulty', type=str, default='medium',
        choices=['easy', 'medium', 'hard'],
        help='Heuristic opponent difficulty'
    )
    parser.add_argument(
        '--max-turns', type=int, default=100,
        help='Maximum turns per episode'
    )

    # Training settings
    parser.add_argument(
        '--total-steps', type=int, default=1000000,
        help='Total training steps'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='Log every N updates'
    )
    parser.add_argument(
        '--save-interval', type=int, default=50000,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--render', action='store_true',
        help='Render training (slower)'
    )

    # Directories
    parser.add_argument(
        '--model-dir', type=str, default='models',
        help='Model save directory'
    )
    parser.add_argument(
        '--log-dir', type=str, default='logs',
        help='Tensorboard log directory'
    )

    # PPO specific
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per rollout (PPO)')
    parser.add_argument('--n-epochs', type=int, default=None, help='Epochs per update (PPO)')
    parser.add_argument('--batch-size', type=int, default=None, help='Minibatch size')
    parser.add_argument('--clip-epsilon', type=float, default=None, help='PPO clip parameter')

    # DQN specific
    parser.add_argument('--buffer-size', type=int, default=None, help='Replay buffer size')
    parser.add_argument('--epsilon-start', type=float, default=None, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=None, help='Final epsilon')
    parser.add_argument('--target-update', type=int, default=None, help='Target network update frequency')

    # LLM data generation
    parser.add_argument('--num-examples', type=int, default=100, help='Number of examples to generate')
    parser.add_argument('--llm-provider', type=str, default='anthropic',
                       choices=['anthropic', 'openai'], help='LLM provider')
    parser.add_argument('--output-file', type=str, default='training_data.json',
                       help='Output file for LLM-generated data')

    # Device
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu, auto-detected if not specified)'
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def train_ppo_agent(args):
    """Train PPO agent."""
    import torch
    from training.env import ScorchedEarthEnv
    from training.ppo import PPOAgent, train_ppo

    print("Initializing PPO training...")

    # Create environment
    render_mode = 'human' if args.render else None
    env = ScorchedEarthEnv(
        num_players=args.num_players,
        opponent_type=args.opponent,
        opponent_difficulty=args.opponent_difficulty,
        max_turns=args.max_turns,
        render_mode=render_mode,
        seed=args.seed
    )

    # Create agent
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = PPOAgent(
        lr=args.lr,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        device=device
    )

    print(f"Training on {device}")
    print(f"Total steps: {args.total_steps}")
    print(f"Opponent: {args.opponent} ({args.opponent_difficulty})")

    # Setup tensorboard logging if available
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    except ImportError:
        print("TensorBoard not available, skipping logging")

    def callback(step, info):
        if writer:
            writer.add_scalar('Episode/reward', info['episode_reward'], step)
            writer.add_scalar('Episode/length', info['episode_length'], step)

    # Train
    try:
        stats = train_ppo(
            env=env,
            agent=agent,
            total_steps=args.total_steps,
            n_steps=args.n_steps,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            callback=callback
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        os.makedirs(args.model_dir, exist_ok=True)
        agent.save(args.model_dir, 'ppo')
        print(f"Model saved to {args.model_dir}")

        if writer:
            writer.close()

        env.close()


def train_dqn_agent(args):
    """Train DQN agent."""
    import torch
    from training.env import ScorchedEarthEnv
    from training.dqn import DQNAgent, train_dqn

    print("Initializing DQN training...")

    # Create environment
    render_mode = 'human' if args.render else None
    env = ScorchedEarthEnv(
        num_players=args.num_players,
        opponent_type=args.opponent,
        opponent_difficulty=args.opponent_difficulty,
        max_turns=args.max_turns,
        render_mode=render_mode,
        seed=args.seed
    )

    # Create agent
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = DQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        target_update_freq=args.target_update,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device
    )

    print(f"Training on {device}")
    print(f"Total steps: {args.total_steps}")
    print(f"Opponent: {args.opponent} ({args.opponent_difficulty})")

    # Setup tensorboard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    except ImportError:
        print("TensorBoard not available")

    def callback(step, info):
        if writer:
            writer.add_scalar('Episode/reward', info['episode_reward'], step)
            writer.add_scalar('Episode/length', info['episode_length'], step)
            writer.add_scalar('Training/epsilon', info['epsilon'], step)

    # Train
    try:
        stats = train_dqn(
            env=env,
            agent=agent,
            total_steps=args.total_steps,
            log_interval=args.log_interval * 100,  # DQN logs per step
            save_interval=args.save_interval,
            callback=callback
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        os.makedirs(args.model_dir, exist_ok=True)
        agent.save(args.model_dir, 'dqn')
        print(f"Model saved to {args.model_dir}")

        if writer:
            writer.close()

        env.close()


def generate_llm_data(args):
    """Generate training data using LLM."""
    from training.env import ScorchedEarthEnv
    from training.llm_data_generator import LLMDataGenerator

    print(f"Generating {args.num_examples} training examples using {args.llm_provider}...")

    # Create environment for generating states
    env = ScorchedEarthEnv(
        num_players=args.num_players,
        opponent_type='heuristic',
        max_turns=args.max_turns,
        seed=args.seed
    )

    try:
        # Create generator
        generator = LLMDataGenerator(provider=args.llm_provider)

        # Generate examples
        examples = generator.generate_training_batch(
            env=env,
            num_examples=args.num_examples,
            save_path=args.output_file
        )

        print(f"Generated {len(examples)} examples")

        # Also generate curriculum suggestions
        print("\nGenerating curriculum suggestions...")
        curriculum = generator.generate_curriculum_stages()
        print("Curriculum stages:")
        for i, stage in enumerate(curriculum):
            print(f"  {i+1}. {stage.get('name', 'Stage')}: {stage}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you have created the API key file (e.g., anthropic.key.txt)")
        sys.exit(1)
    finally:
        env.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("Scorched Earth RL Training")
    print("=" * 60)

    if args.algo == 'ppo':
        train_ppo_agent(args)
    elif args.algo == 'dqn':
        train_dqn_agent(args)
    elif args.algo == 'llm-gen':
        generate_llm_data(args)
    else:
        print(f"Unknown algorithm: {args.algo}")
        sys.exit(1)

    print("Done!")


if __name__ == '__main__':
    main()
