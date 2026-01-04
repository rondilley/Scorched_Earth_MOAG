# Neural network architectures for RL agents

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from training.config import NETWORK_CONFIG, OBS_CONFIG, ACTION_CONFIG


class TerrainEncoder(nn.Module):
    """1D CNN encoder for terrain height data."""

    def __init__(self, input_size=1024):
        super().__init__()

        channels = NETWORK_CONFIG['terrain_channels']
        kernels = NETWORK_CONFIG['terrain_kernels']
        strides = NETWORK_CONFIG['terrain_strides']

        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=kernels[0], stride=strides[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=kernels[1], stride=strides[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=kernels[2], stride=strides[2])

        # Calculate output size
        def conv_out_size(size, kernel, stride):
            return (size - kernel) // stride + 1

        size = input_size
        size = conv_out_size(size, kernels[0], strides[0])
        size = conv_out_size(size, kernels[1], strides[1])
        size = conv_out_size(size, kernels[2], strides[2])
        self.output_size = size * channels[2]

    def forward(self, x):
        """Forward pass.

        Args:
            x: Terrain tensor of shape (batch, 1024) or (batch, 1, 1024)

        Returns:
            Encoded features of shape (batch, output_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(start_dim=1)


class TankEncoder(nn.Module):
    """MLP encoder for tank state features."""

    def __init__(self):
        super().__init__()

        player_features = OBS_CONFIG['tank_features']
        opponent_features = OBS_CONFIG['max_opponents'] * OBS_CONFIG['opponent_features']
        wind_features = 1
        total_features = player_features + opponent_features + wind_features

        hidden = NETWORK_CONFIG['hidden_size']
        self.fc1 = nn.Linear(total_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.output_size = hidden

    def forward(self, player_tank, opponent_tanks, wind):
        """Forward pass.

        Args:
            player_tank: (batch, 6) player tank features
            opponent_tanks: (batch, max_opponents * 4) opponent features
            wind: (batch, 1) wind value

        Returns:
            Encoded features of shape (batch, hidden_size)
        """
        x = torch.cat([player_tank, opponent_tanks, wind], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with mixed action space.

    Outputs:
        - Continuous: angle (0-180), power (50-500)
        - Discrete: weapon (0-4)
        - Value estimate for critic
    """

    def __init__(self):
        super().__init__()

        # Encoders
        self.terrain_encoder = TerrainEncoder(OBS_CONFIG['terrain_size'])
        self.tank_encoder = TankEncoder()

        # Combined feature size
        combined_size = self.terrain_encoder.output_size + self.tank_encoder.output_size
        hidden = NETWORK_CONFIG['hidden_size']

        # Shared layers
        self.shared_fc1 = nn.Linear(combined_size, hidden)
        self.shared_fc2 = nn.Linear(hidden, hidden)

        # Continuous action heads (angle, power)
        self.angle_mean = nn.Linear(hidden, 1)
        self.angle_log_std = nn.Parameter(torch.zeros(1))
        self.power_mean = nn.Linear(hidden, 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))

        # Discrete action head (weapon)
        self.weapon_logits = nn.Linear(hidden, ACTION_CONFIG['num_weapons'])

        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def _encode(self, obs):
        """Encode observation into features."""
        terrain_features = self.terrain_encoder(obs['terrain'])
        tank_features = self.tank_encoder(
            obs['player_tank'],
            obs['opponent_tanks'],
            obs['wind']
        )
        combined = torch.cat([terrain_features, tank_features], dim=-1)

        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))
        return x

    def forward(self, obs):
        """Forward pass returning all outputs.

        Returns:
            angle_mean, angle_std, power_mean, power_std, weapon_logits, value
        """
        features = self._encode(obs)

        angle_mean = torch.sigmoid(self.angle_mean(features)) * ACTION_CONFIG['angle_max']
        angle_std = self.angle_log_std.exp().expand_as(angle_mean)

        power_range = ACTION_CONFIG['power_max'] - ACTION_CONFIG['power_min']
        power_mean = torch.sigmoid(self.power_mean(features)) * power_range + ACTION_CONFIG['power_min']
        power_std = self.power_log_std.exp().expand_as(power_mean)

        weapon_logits = self.weapon_logits(features)
        value = self.value_head(features)

        return angle_mean, angle_std, power_mean, power_std, weapon_logits, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy.

        Args:
            obs: Observation dict with torch tensors
            deterministic: If True, return mean actions

        Returns:
            action: Dict with angle, power, weapon
            log_prob: Log probability of the action
            value: Value estimate
        """
        angle_mean, angle_std, power_mean, power_std, weapon_logits, value = self.forward(obs)

        if deterministic:
            angle = angle_mean
            power = power_mean
            weapon = weapon_logits.argmax(dim=-1)
            log_prob = torch.zeros(angle.shape[0], device=angle.device)
        else:
            # Sample continuous actions
            angle_dist = Normal(angle_mean, angle_std)
            angle = angle_dist.sample()
            angle_log_prob = angle_dist.log_prob(angle)

            power_dist = Normal(power_mean, power_std)
            power = power_dist.sample()
            power_log_prob = power_dist.log_prob(power)

            # Sample discrete action
            weapon_dist = Categorical(logits=weapon_logits)
            weapon = weapon_dist.sample()
            weapon_log_prob = weapon_dist.log_prob(weapon)

            # Combined log probability
            log_prob = angle_log_prob.squeeze(-1) + power_log_prob.squeeze(-1) + weapon_log_prob

        # Clamp actions to valid ranges
        angle = torch.clamp(angle, ACTION_CONFIG['angle_min'], ACTION_CONFIG['angle_max'])
        power = torch.clamp(power, ACTION_CONFIG['power_min'], ACTION_CONFIG['power_max'])

        action = {
            'angle': angle.squeeze(-1),
            'power': power.squeeze(-1),
            'weapon': weapon,
        }

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """Evaluate log probabilities and entropy for given actions.

        Args:
            obs: Observation dict
            actions: Dict with angle, power, weapon tensors

        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of the policy
            value: Value estimates
        """
        angle_mean, angle_std, power_mean, power_std, weapon_logits, value = self.forward(obs)

        # Continuous action distributions
        angle_dist = Normal(angle_mean, angle_std)
        power_dist = Normal(power_mean, power_std)
        weapon_dist = Categorical(logits=weapon_logits)

        # Log probabilities
        angle_log_prob = angle_dist.log_prob(actions['angle'].unsqueeze(-1)).squeeze(-1)
        power_log_prob = power_dist.log_prob(actions['power'].unsqueeze(-1)).squeeze(-1)
        weapon_log_prob = weapon_dist.log_prob(actions['weapon'])

        log_prob = angle_log_prob + power_log_prob + weapon_log_prob

        # Entropy (for exploration bonus)
        entropy = angle_dist.entropy().mean() + power_dist.entropy().mean() + weapon_dist.entropy().mean()

        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs):
        """Get value estimate only."""
        features = self._encode(obs)
        return self.value_head(features).squeeze(-1)


class DQNNetwork(nn.Module):
    """Dueling DQN network for discretized action space."""

    def __init__(self):
        super().__init__()

        # Encoders
        self.terrain_encoder = TerrainEncoder(OBS_CONFIG['terrain_size'])
        self.tank_encoder = TankEncoder()

        # Combined feature size
        combined_size = self.terrain_encoder.output_size + self.tank_encoder.output_size
        hidden = NETWORK_CONFIG['hidden_size']

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(combined_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Number of discrete actions
        self.num_actions = (ACTION_CONFIG['angle_bins'] *
                          ACTION_CONFIG['power_bins'] *
                          ACTION_CONFIG['num_weapons'])

        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.num_actions),
        )

    def _encode(self, obs):
        """Encode observation into features."""
        terrain_features = self.terrain_encoder(obs['terrain'])
        tank_features = self.tank_encoder(
            obs['player_tank'],
            obs['opponent_tanks'],
            obs['wind']
        )
        combined = torch.cat([terrain_features, tank_features], dim=-1)
        return self.feature_layer(combined)

    def forward(self, obs):
        """Forward pass returning Q-values for all actions.

        Returns:
            Q-values tensor of shape (batch, num_actions)
        """
        features = self._encode(obs)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values

    def get_action(self, obs, epsilon=0.0):
        """Select action using epsilon-greedy policy.

        Args:
            obs: Observation dict
            epsilon: Exploration probability

        Returns:
            action_index: Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)

        with torch.no_grad():
            q_values = self.forward(obs)
            return q_values.argmax(dim=-1).item()


def obs_to_tensor(obs, device='cpu'):
    """Convert observation dict to tensor dict.

    Args:
        obs: Observation dict with numpy arrays
        device: Torch device

    Returns:
        Dict with torch tensors, batch dimension added
    """
    return {
        'terrain': torch.tensor(obs['terrain'], dtype=torch.float32, device=device).unsqueeze(0),
        'player_tank': torch.tensor(obs['player_tank'], dtype=torch.float32, device=device).unsqueeze(0),
        'opponent_tanks': torch.tensor(obs['opponent_tanks'], dtype=torch.float32, device=device).unsqueeze(0),
        'wind': torch.tensor(obs['wind'], dtype=torch.float32, device=device).unsqueeze(0),
    }


def batch_obs_to_tensor(obs_list, device='cpu'):
    """Convert list of observations to batched tensor dict.

    Args:
        obs_list: List of observation dicts
        device: Torch device

    Returns:
        Dict with batched torch tensors
    """
    return {
        'terrain': torch.tensor(
            np.stack([o['terrain'] for o in obs_list]),
            dtype=torch.float32, device=device
        ),
        'player_tank': torch.tensor(
            np.stack([o['player_tank'] for o in obs_list]),
            dtype=torch.float32, device=device
        ),
        'opponent_tanks': torch.tensor(
            np.stack([o['opponent_tanks'] for o in obs_list]),
            dtype=torch.float32, device=device
        ),
        'wind': torch.tensor(
            np.stack([o['wind'] for o in obs_list]),
            dtype=torch.float32, device=device
        ),
    }
