# Experience replay buffer for DQN

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Simple replay buffer for DQN experience replay."""

    def __init__(self, capacity=100000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state: Observation dict
            action: Action index (int)
            reward: Reward value
            next_state: Next observation dict
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = [t[0] for t in batch]
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = [t[3] for t in batch]
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Samples transitions with probability proportional to their TD error.
    """

    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with prioritized probabilities.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        self.frame += 1

        # Calculate sampling probabilities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize

        # Get transitions
        batch = [self.buffer[i] for i in indices]

        states = [t[0] for t in batch]
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = [t[3] for t in batch]
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors.

        Args:
            indices: Indices of sampled transitions
            td_errors: TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """Rollout buffer for PPO on-policy learning."""

    def __init__(self):
        """Initialize empty rollout buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        """Add a transition to the rollout.

        Args:
            obs: Observation dict
            action: Action dict
            reward: Reward value
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for the state after last transition
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Append last value for bootstrapping
        values = np.append(values, last_value)

        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values[:-1]

        return returns, advantages

    def get_batches(self, batch_size, returns, advantages, device='cpu'):
        """Generate minibatches for training.

        Args:
            batch_size: Size of each minibatch
            returns: Computed returns
            advantages: Computed advantages
            device: Torch device

        Yields:
            Batched data for training
        """
        import torch
        from training.networks import batch_obs_to_tensor

        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Batch observations
            batch_obs = [self.observations[i] for i in batch_indices]
            obs_tensor = batch_obs_to_tensor(batch_obs, device)

            # Batch actions
            batch_actions = {
                'angle': torch.tensor(
                    [self.actions[i]['angle'] for i in batch_indices],
                    dtype=torch.float32, device=device
                ),
                'power': torch.tensor(
                    [self.actions[i]['power'] for i in batch_indices],
                    dtype=torch.float32, device=device
                ),
                'weapon': torch.tensor(
                    [self.actions[i]['weapon'] for i in batch_indices],
                    dtype=torch.long, device=device
                ),
            }

            # Other tensors
            batch_log_probs = torch.tensor(
                [self.log_probs[i] for i in batch_indices],
                dtype=torch.float32, device=device
            )
            batch_returns = torch.tensor(
                returns[batch_indices],
                dtype=torch.float32, device=device
            )
            batch_advantages = torch.tensor(
                advantages[batch_indices],
                dtype=torch.float32, device=device
            )

            yield obs_tensor, batch_actions, batch_log_probs, batch_returns, batch_advantages

    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.observations)
