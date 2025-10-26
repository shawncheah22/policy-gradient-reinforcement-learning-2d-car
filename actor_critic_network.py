import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions):
        super(ActorCriticNetwork, self).__init__()
        # 🧠 Convolutional layers to process the image
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # We need to calculate the size of the output from the conv layers
        # A quick way is to pass a dummy tensor through the conv layers
        dummy_input = torch.randn(1, 4, 84, 84)
        conv_out_size = self._get_conv_out(dummy_input)

        # 🧠 Fully connected layers with gradually decreasing size
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # Separate branches for mean and std to allow different feature processing
        self.actor_mean_hidden = nn.Linear(256, 128)
        self.actor_log_std_hidden = nn.Linear(256, 128)
        
        self.actor_mean = nn.Linear(128, n_actions)
        self.actor_log_std = nn.Linear(128, n_actions)
        
        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(1024)
        self.layer_norm2 = nn.LayerNorm(512)
        self.layer_norm3 = nn.LayerNorm(256)

        # Add action bounds as class attributes
        self.action_bounds = {
            'steering': (-1.0, 1.0),    # Left to right
            'gas': (0.0, 1.0),         # No gas to full gas
            'brake': (0.0, 1.0)        # No brake to full brake
        }
        # Critic network layers
        self.critic_hidden = nn.Linear(256, 256)
        self.critic_hidden_2 = nn.Linear(256, 256)
        self.critic_head = nn.Linear(256, 1)


    def _get_conv_out(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))

    def forward(self, x):
        # Pass input through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) 
        
        # Apply fully connected layers with layer normalization and ReLU
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.layer_norm3(self.fc3(x)))
        
        # Separate processing for mean and log_std
        mean_hidden = F.relu(self.actor_mean_hidden(x))
        log_std_hidden = F.relu(self.actor_log_std_hidden(x))
        
        
        mean = self.actor_mean(mean_hidden)
        log_std = self.actor_log_std(log_std_hidden)

        # Prevent log_std from exploding / producing NaNs
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        std = log_std.exp().clamp(min=1e-6)

        distribution = Normal(mean, std)
        raw_action = distribution.rsample()
        log_prob = distribution.log_prob(raw_action).sum(dim=-1, keepdim=True)

        correction = torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6)
        correction = correction.sum(dim=-1, keepdim=True)

        final_log_prob = log_prob - correction

        # Replace the manual scaling with a more explicit bounds handling
        steering = torch.tanh(raw_action[:, 0])  # Already in [-1, 1]
        
        # Scale gas from [-1, 1] to [0, 1]
        gas = torch.sigmoid(raw_action[:, 1])    # Ensures [0, 1] bound
        
        # Scale brake from [-1, 1] to [0, 1]
        brake = torch.sigmoid(raw_action[:, 2])  # Ensures [0, 1] bound
        
        # Stack the bounded actions
        scaled_action = torch.stack([
            steering,
            gas,
            brake
        ], dim=1)

        # Add bounds checking for safety
        assert torch.all(scaled_action[:, 0].ge(-1) & scaled_action[:, 0].le(1)), "Steering out of bounds"
        assert torch.all(scaled_action[:, 1:].ge(0) & scaled_action[:, 1:].le(1)), "Gas/Brake out of bounds"

        critic_hidden = F.relu(self.critic_hidden(x))
        value = self.critic_head(critic_hidden)

        return scaled_action, final_log_prob, value

    def get_action_bounds(self):
        """Returns the action bounds for each action dimension"""
        return self.action_bounds