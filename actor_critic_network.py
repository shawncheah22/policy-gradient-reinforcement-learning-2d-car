import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorNetwork(nn.Module):
    """Actor (policy) network for continuous actions.

    - Input: state image tensor (B, C=4, H, W)
    - Output: action tensor (B, n_actions) with tanh squash and
      log_prob tensor (B, 1) (reparameterized sample)
    """
    def __init__(self, n_actions, in_channels=4):
        super(ActorNetwork, self).__init__()

        # CNN trunk
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # determine conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self._get_conv_out(dummy)

        # actor MLP
        self.actor_fc1 = nn.Linear(conv_out, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.actor_mean = nn.Linear(256, n_actions)
        self.actor_log_std = nn.Linear(256, n_actions)

        # initialization
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def _get_conv_out(self, x: torch.Tensor) -> int:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))

    def forward(self, state: torch.Tensor):
        # state: (B, C, H, W)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))

        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick
        dist = Normal(mean, std)
        raw_action = dist.rsample()  # (B, n_actions)

        # log_prob before squash
        log_prob = dist.log_prob(raw_action).sum(dim=1, keepdim=True)  # (B,1)

        # squash with tanh and correct log prob
        action = torch.tanh(raw_action)
        # formula: log_prob = log_prob - sum(log(1 - tanh(u)^2) )
        # add small epsilon for numerical stability
        eps = 1e-6
        log_prob = log_prob - (torch.log(1 - action.pow(2) + eps)).sum(dim=1, keepdim=True)

        return action, log_prob


class CriticNetwork(nn.Module):
    """Single critic network (Q-function) taking state and action.

    - Input: state image (B, C, H, W) and action (B, n_actions)
    - Output: scalar Q-value (B, 1)
    """
    def __init__(self, n_actions, in_channels=4):
        super(CriticNetwork, self).__init__()

        # CNN trunk for state
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = int(np.prod(self._get_conv_activation_size(dummy)))

        # Q-network MLP: state features + action -> q
        self.critic_fc1 = nn.Linear(conv_out + n_actions, 256)
        self.critic_fc2 = nn.Linear(256, 256)
        self.critic_head = nn.Linear(256, 1)

    def _get_conv_activation_size(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.size()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        sa = torch.cat([x, action], dim=1)

        q = F.relu(self.critic_fc1(sa))
        q = F.relu(self.critic_fc2(q))
        q = self.critic_head(q)
        return q
