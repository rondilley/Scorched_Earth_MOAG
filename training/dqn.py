# Deep Q-Network (DQN) agent implementation

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.networks import DQNNetwork, obs_to_tensor, batch_obs_to_tensor
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.config import DQN_CONFIG, MODELS_DIR
from training.env import index_to_action, get_total_actions


def format_time(seconds):
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        lr=None,
        gamma=None,
        epsilon_start=None,
        epsilon_end=None,
        epsilon_decay_steps=None,
        target_update_freq=None,
        batch_size=None,
        buffer_size=None,
        learning_starts=None,
        prioritized=True,
        device=None
    ):
        """Initialize DQN agent.

        Args:
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            target_update_freq: Steps between target network updates
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            learning_starts: Steps before starting training
            prioritized: Use prioritized experience replay
            device: Torch device ('cuda' or 'cpu')
        """
        # Use config defaults if not specified
        self.lr = lr or DQN_CONFIG['lr']
        self.gamma = gamma or DQN_CONFIG['gamma']
        self.epsilon_start = epsilon_start or DQN_CONFIG['epsilon_start']
        self.epsilon_end = epsilon_end or DQN_CONFIG['epsilon_end']
        self.epsilon_decay_steps = epsilon_decay_steps or DQN_CONFIG['epsilon_decay_steps']
        self.target_update_freq = target_update_freq or DQN_CONFIG['target_update_freq']
        self.batch_size = batch_size or DQN_CONFIG['batch_size']
        self.buffer_size = buffer_size or DQN_CONFIG['buffer_size']
        self.learning_starts = learning_starts or DQN_CONFIG['learning_starts']
        self.prioritized = prioritized

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer
        if prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Training state
        self.total_steps = 0
        self.num_actions = get_total_actions()

    def get_epsilon(self, step=None):
        """Get current epsilon value for exploration.

        Args:
            step: Current step (uses self.total_steps if None)

        Returns:
            Current epsilon value
        """
        if step is None:
            step = self.total_steps

        if step >= self.epsilon_decay_steps:
            return self.epsilon_end

        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (step / self.epsilon_decay_steps)

    def select_action(self, obs, epsilon=None):
        """Select action using epsilon-greedy policy.

        Args:
            obs: Observation dict (numpy arrays)
            epsilon: Exploration rate (uses schedule if None)

        Returns:
            action_index: Selected action index
            action_dict: Action as dict with angle, power, weapon
        """
        if epsilon is None:
            epsilon = self.get_epsilon()

        if np.random.random() < epsilon:
            action_index = np.random.randint(0, self.num_actions)
        else:
            obs_tensor = obs_to_tensor(obs, self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action_index = q_values.argmax(dim=-1).item()

        action_dict = index_to_action(action_index)
        return action_index, action_dict

    def store_transition(self, obs, action_index, reward, next_obs, done):
        """Store transition in replay buffer.

        Args:
            obs: Current observation
            action_index: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        self.replay_buffer.push(obs, action_index, reward, next_obs, done)

    def update(self):
        """Perform one training update.

        Returns:
            loss: Training loss (None if not enough samples)
        """
        if len(self.replay_buffer) < self.learning_starts:
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from buffer
        if self.prioritized:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, device=self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)

        # Convert to tensors
        states_tensor = batch_obs_to_tensor(states, self.device)
        next_states_tensor = batch_obs_to_tensor(next_states, self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values
        q_values = self.q_network(states_tensor)
        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using online network
            next_q_online = self.q_network(next_states_tensor)
            next_actions = next_q_online.argmax(dim=-1)

            # Evaluate using target network
            next_q_target = self.target_network(next_states_tensor)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        # Compute TD errors
        td_errors = target_q - q_values

        # Huber loss with importance sampling weights
        loss = (weights * nn.functional.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        if self.prioritized:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path=None, name='dqn'):
        """Save model checkpoint.

        Args:
            path: Directory to save to (uses default if None)
            name: Checkpoint name prefix
        """
        if path is None:
            path = MODELS_DIR

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
            }
        }

        filepath = os.path.join(path, f'{name}_step_{self.total_steps}.pt')
        torch.save(checkpoint, filepath)

        # Also save as 'best' for easy loading
        best_path = os.path.join(path, f'{name}_best.pt')
        torch.save(checkpoint, best_path)

        return filepath

    def load(self, path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)

        # Update config if saved
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.lr = config.get('lr', self.lr)
            self.gamma = config.get('gamma', self.gamma)


def train_dqn(
    env,
    agent,
    total_steps=1000000,
    log_interval=1000,
    save_interval=10000,
    callback=None
):
    """Train DQN agent on environment.

    Args:
        env: ScorchedEarthEnv instance
        agent: DQNAgent instance
        total_steps: Total training steps
        log_interval: Steps between logging
        save_interval: Steps between checkpoints
        callback: Optional callback function(step, info)

    Returns:
        Training statistics
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
    }

    # Print training configuration
    print("\n" + "=" * 60)
    print("DQN Training Started")
    print("=" * 60)
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Log interval:      every {log_interval:,} steps")
    print(f"  Save interval:     every {save_interval:,} steps")
    print(f"  Buffer size:       {agent.buffer_size:,}")
    print(f"  Learning starts:   {agent.learning_starts:,} steps")
    print(f"  Epsilon:           {agent.epsilon_start:.2f} -> {agent.epsilon_end:.2f}")
    print(f"  Device:            {agent.device}")
    print("=" * 60)
    print("\nFilling replay buffer...")
    sys.stdout.flush()

    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    episodes_completed = 0
    start_time = time.time()
    last_log_time = start_time
    last_progress_time = start_time
    last_save_steps = 0
    learning_started = False

    for step in range(total_steps):
        # Progress during buffer filling (every 5 seconds)
        if not learning_started and time.time() - last_progress_time >= 5.0:
            buffer_fill = len(agent.replay_buffer) / agent.learning_starts * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (agent.learning_starts - step) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"  Buffer: {buffer_fill:5.1f}% ({len(agent.replay_buffer)}/{agent.learning_starts}) | "
                  f"{steps_per_sec:.1f} steps/s | ETA: {eta:.0f}s | "
                  f"Episodes: {episodes_completed}", end="\r")
            sys.stdout.flush()
            last_progress_time = time.time()

        # Select action
        action_index, action_dict = agent.select_action(obs)

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action_index, reward, next_obs, done)

        # Train
        loss = agent.update()
        if loss is not None:
            stats['losses'].append(loss)
            if not learning_started:
                learning_started = True
                print(" " * 80, end="\r")  # Clear buffer progress line
                elapsed = time.time() - start_time
                print(f"Buffer filled in {elapsed:.1f}s. Training started...")
                sys.stdout.flush()

        episode_reward += reward
        episode_length += 1

        if done:
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            episodes_completed += 1

            if callback:
                callback(step, {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'epsilon': agent.get_epsilon(),
                })

            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Calculate progress
        progress = (step + 1) / total_steps * 100
        elapsed = time.time() - start_time
        steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (total_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0

        # Logging
        if (step + 1) % log_interval == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:]) if stats['episode_rewards'] else 0
            avg_length = np.mean(stats['episode_lengths'][-100:]) if stats['episode_lengths'] else 0
            avg_loss = np.mean(stats['losses'][-100:]) if stats['losses'] else 0
            buffer_fill = len(agent.replay_buffer) / agent.buffer_size * 100

            print(f"\n[{progress:5.1f}%] Step {step+1:,}/{total_steps:,}")
            print(f"  Episodes: {episodes_completed} | Avg reward: {avg_reward:.2f} | Avg length: {avg_length:.1f}")
            print(f"  Loss: {avg_loss:.4f} | Epsilon: {agent.get_epsilon():.3f} | Buffer: {buffer_fill:.0f}%")
            print(f"  Speed: {steps_per_sec:.0f} steps/s | ETA: {format_time(eta)}")
            sys.stdout.flush()
            last_log_time = time.time()

        # Progress indicator between logs (every 10 seconds)
        elif time.time() - last_log_time > 10:
            print(f"  ... {progress:.1f}% ({step+1:,} steps, {episodes_completed} eps, eps={agent.get_epsilon():.3f})", end="\r")
            sys.stdout.flush()
            last_log_time = time.time()

        # Save checkpoint
        if (step + 1) - last_save_steps >= save_interval:
            filepath = agent.save()
            print(f"\n  [Checkpoint saved: {filepath}]")
            sys.stdout.flush()
            last_save_steps = step + 1

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Total episodes:   {episodes_completed}")
    print(f"  Total time:       {format_time(total_time)}")
    print(f"  Avg speed:        {total_steps/total_time:.0f} steps/s")
    if stats['episode_rewards']:
        print(f"  Final avg reward: {np.mean(stats['episode_rewards'][-100:]):.2f}")
    print("=" * 60 + "\n")

    return stats
