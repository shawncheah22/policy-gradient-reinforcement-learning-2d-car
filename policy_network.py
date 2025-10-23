import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, n_actions):
        super(PolicyNetwork, self).__init__()
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
        std = log_std.exp()

        distribution = Normal(mean, std)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)

        return action, log_prob