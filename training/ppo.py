# Proximal Policy Optimization (PPO) agent implementation

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.networks import ActorCritic, obs_to_tensor, batch_obs_to_tensor
from training.replay_buffer import RolloutBuffer
from training.config import PPO_CONFIG, MODELS_DIR


def format_time(seconds):
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class PPOAgent:
    """PPO agent with actor-critic architecture."""

    def __init__(
        self,
        lr=None,
        gamma=None,
        gae_lambda=None,
        clip_epsilon=None,
        entropy_coef=None,
        value_coef=None,
        max_grad_norm=None,
        n_epochs=None,
        batch_size=None,
        device=None
    ):
        """Initialize PPO agent.

        Args:
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Minibatch size
            device: Torch device
        """
        # Use config defaults if not specified
        self.lr = lr or PPO_CONFIG['lr']
        self.gamma = gamma or PPO_CONFIG['gamma']
        self.gae_lambda = gae_lambda or PPO_CONFIG['gae_lambda']
        self.clip_epsilon = clip_epsilon or PPO_CONFIG['clip_epsilon']
        self.entropy_coef = entropy_coef or PPO_CONFIG['entropy_coef']
        self.value_coef = value_coef or PPO_CONFIG['value_coef']
        self.max_grad_norm = max_grad_norm or PPO_CONFIG['max_grad_norm']
        self.n_epochs = n_epochs or PPO_CONFIG['n_epochs']
        self.batch_size = batch_size or PPO_CONFIG['batch_size']

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Network
        self.network = ActorCritic().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Training state
        self.total_steps = 0

    def select_action(self, obs, deterministic=False):
        """Select action from policy.

        Args:
            obs: Observation dict (numpy arrays)
            deterministic: Use mean action instead of sampling

        Returns:
            action_dict: Action as dict with angle, power, weapon
            log_prob: Log probability of action
            value: Value estimate
        """
        obs_tensor = obs_to_tensor(obs, self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)

        # Convert to numpy/python types
        action_dict = {
            'angle': action['angle'].cpu().item(),
            'power': action['power'].cpu().item(),
            'weapon': action['weapon'].cpu().item(),
        }

        return action_dict, log_prob.cpu().item(), value.cpu().item()

    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in rollout buffer.

        Args:
            obs: Observation
            action: Action dict
            reward: Reward
            value: Value estimate
            log_prob: Log probability
            done: Episode done flag
        """
        self.rollout_buffer.add(obs, action, reward, value, log_prob, done)

    def get_value(self, obs):
        """Get value estimate for observation.

        Args:
            obs: Observation dict

        Returns:
            Value estimate
        """
        obs_tensor = obs_to_tensor(obs, self.device)
        with torch.no_grad():
            value = self.network.get_value(obs_tensor)
        return value.cpu().item()

    def update(self, last_value):
        """Perform PPO update using collected rollouts.

        Args:
            last_value: Value estimate for state after last transition

        Returns:
            Dictionary with training metrics
        """
        # Compute returns and advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get_batches(
                self.batch_size, returns, advantages, self.device
            ):
                obs_batch, actions_batch, old_log_probs, returns_batch, advantages_batch = batch

                # Evaluate current policy
                log_probs, entropy, values = self.network.evaluate_actions(obs_batch, actions_batch)

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns_batch)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Clear buffer
        self.rollout_buffer.clear()

        self.total_steps += len(returns)

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def save(self, path=None, name='ppo'):
        """Save model checkpoint.

        Args:
            path: Directory to save to
            name: Checkpoint name prefix
        """
        if path is None:
            path = MODELS_DIR

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
            }
        }

        filepath = os.path.join(path, f'{name}_step_{self.total_steps}.pt')
        torch.save(checkpoint, filepath)

        # Also save as 'best'
        best_path = os.path.join(path, f'{name}_best.pt')
        torch.save(checkpoint, best_path)

        return filepath

    def load(self, path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)


def train_ppo(
    env,
    agent,
    total_steps=1000000,
    n_steps=2048,
    log_interval=10,
    save_interval=50000,
    callback=None
):
    """Train PPO agent on environment.

    Args:
        env: ScorchedEarthEnv instance
        agent: PPOAgent instance
        total_steps: Total training steps
        n_steps: Steps per rollout before update
        log_interval: Updates between logging
        save_interval: Steps between checkpoints
        callback: Optional callback function

    Returns:
        Training statistics
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
    }

    # Print training configuration
    print("\n" + "=" * 60)
    print("PPO Training Started")
    print("=" * 60)
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Log interval:    every {log_interval} updates")
    print(f"  Save interval:   every {save_interval:,} steps")
    print(f"  Device:          {agent.device}")
    print("=" * 60)
    print("\nCollecting first rollout...")
    sys.stdout.flush()

    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    num_updates = 0
    steps = 0
    episodes_completed = 0
    start_time = time.time()
    last_log_time = start_time
    last_save_steps = 0

    while steps < total_steps:
        rollout_start = time.time()
        rollout_last_print = rollout_start

        # Collect rollout
        for rollout_step in range(n_steps):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            steps += 1

            if done:
                stats['episode_rewards'].append(episode_reward)
                stats['episode_lengths'].append(episode_length)
                episodes_completed += 1

                if callback:
                    callback(steps, {
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                    })

                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

            # Progress during rollout collection (every 5 seconds)
            if time.time() - rollout_last_print >= 5.0:
                rollout_progress = (rollout_step + 1) / n_steps * 100
                rollout_elapsed = time.time() - rollout_start
                rollout_speed = (rollout_step + 1) / rollout_elapsed if rollout_elapsed > 0 else 0
                rollout_eta = (n_steps - rollout_step - 1) / rollout_speed if rollout_speed > 0 else 0
                print(f"  Rollout: {rollout_progress:5.1f}% ({rollout_step+1}/{n_steps}) | "
                      f"{rollout_speed:.1f} steps/s | ETA: {rollout_eta:.0f}s | "
                      f"Episodes: {episodes_completed}", end="\r")
                sys.stdout.flush()
                rollout_last_print = time.time()

            if steps >= total_steps:
                break

        rollout_time = time.time() - rollout_start
        print(" " * 80, end="\r")  # Clear the rollout progress line

        # Compute value for last state
        last_value = agent.get_value(obs) if not done else 0

        # Update policy
        update_start = time.time()
        metrics = agent.update(last_value)
        update_time = time.time() - update_start
        num_updates += 1

        stats['policy_losses'].append(metrics['policy_loss'])
        stats['value_losses'].append(metrics['value_loss'])
        stats['entropies'].append(metrics['entropy'])

        # Calculate progress
        progress = steps / total_steps * 100
        elapsed = time.time() - start_time
        steps_per_sec = steps / elapsed if elapsed > 0 else 0
        eta = (total_steps - steps) / steps_per_sec if steps_per_sec > 0 else 0

        # Logging - always log first update, then every log_interval
        if num_updates == 1 or num_updates % log_interval == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:]) if stats['episode_rewards'] else 0
            avg_length = np.mean(stats['episode_lengths'][-100:]) if stats['episode_lengths'] else 0

            print(f"\n[{progress:5.1f}%] Step {steps:,}/{total_steps:,} | Update {num_updates}")
            print(f"  Episodes: {episodes_completed} | Avg reward: {avg_reward:.2f} | Avg length: {avg_length:.1f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f} | Value loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
            print(f"  Speed: {steps_per_sec:.0f} steps/s | Rollout: {rollout_time:.1f}s | Update: {update_time:.2f}s | ETA: {format_time(eta)}")
            sys.stdout.flush()
            last_log_time = time.time()

        # Progress indicator between logs (every 10 seconds)
        elif time.time() - last_log_time > 10:
            print(f"  ... {progress:.1f}% ({steps:,} steps, {episodes_completed} episodes)", end="\r")
            sys.stdout.flush()
            last_log_time = time.time()

        # Save checkpoint
        if steps - last_save_steps >= save_interval:
            filepath = agent.save()
            print(f"\n  [Checkpoint saved: {filepath}]")
            sys.stdout.flush()
            last_save_steps = steps

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total steps:      {steps:,}")
    print(f"  Total episodes:   {episodes_completed}")
    print(f"  Total time:       {format_time(total_time)}")
    print(f"  Avg speed:        {steps/total_time:.0f} steps/s")
    if stats['episode_rewards']:
        print(f"  Final avg reward: {np.mean(stats['episode_rewards'][-100:]):.2f}")
    print("=" * 60 + "\n")

    return stats
