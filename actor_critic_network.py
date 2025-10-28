import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SACNetwork(nn.Module):
    def __init__(self, n_actions):
        super(SACNetwork, self).__init__()
        
        # 1. --- Shared CNN Trunk (processes the state image) ---
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        dummy_input = torch.randn(1, 4, 84, 84)
        conv_out_size = self._get_conv_out(dummy_input)

        # 2. --- Actor Head (State -> Action) ---
        # This has its own FC layers
        self.actor_fc1 = nn.Linear(conv_out_size, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.actor_mean = nn.Linear(256, n_actions)
        self.actor_log_std = nn.Linear(256, n_actions)

        # 3. --- Critic 1 Head (State + Action -> Q-Value) ---
        # This trunk must combine state (conv_out_size) and action (n_actions)
        self.critic_1_fc1 = nn.Linear(conv_out_size + n_actions, 256)
        self.critic_1_fc2 = nn.Linear(256, 256)
        self.critic_1_head = nn.Linear(256, 1)
        
        # 4. --- Critic 2 Head (State + Action -> Q-Value) ---
        # Identical to Critic 1
        self.critic_2_fc1 = nn.Linear(conv_out_size + n_actions, 256)
        self.critic_2_fc2 = nn.Linear(256, 256)
        self.critic_2_head = nn.Linear(256, 1)

    def _get_conv_out(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))

    # --- We now need three separate forward methods ---

    def forward_actor(self, state):
        """Processes a state and returns an action and log_prob."""
        # Pass state through CNN trunk
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        
        # Pass through Actor's own FC layers
        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))
        
        # ... (rest of the action generation logic: mean, std, sample, log_prob) ...
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        # ... (rest of your logic from before) ...
        
        scaled_action, final_log_prob = ... # (Your tanh squashing logic)
        return scaled_action, final_log_prob

    def forward_critic(self, state, action):
        """Processes a state AND an action, returns Q-values from both critics."""
        # Pass state through CNN trunk
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        
        # Concatenate the flattened state features with the action
        sa = torch.cat([x, action], dim=1) # sa = state-action
        
        # --- Critic 1 ---
        q1 = F.relu(self.critic_1_fc1(sa))
        q1 = F.relu(self.critic_1_fc2(q1))
        q1 = self.critic_1_head(q1)
        
        # --- Critic 2 ---
        q2 = F.relu(self.critic_2_fc1(sa))
        q2 = F.relu(self.critic_2_fc2(q2))
        q2 = self.critic_2_head(q2)
        
        return q1, q2