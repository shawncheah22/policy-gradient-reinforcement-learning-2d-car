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

        # 🧠 Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        
        self.actor_mean = nn.Linear(512, n_actions)
        self.actor_log_std = nn.Linear(512, n_actions)

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
        
        x = F.relu(self.fc1(x))
        
        # --- And use the final layer(s) here ---
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        std = log_std.exp()

        distribution = Normal(mean, std)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)

        return action, log_prob