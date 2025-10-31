import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from actor_critic_network import ActorNetwork, CriticNetwork

# --- We need a Replay Buffer class ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# --- Assume ActorNetwork and CriticNetwork are defined ---
# (Using the separate-trunk design we discussed)

class SACAgent:
    def __init__(self, n_actions, learning_rate=3e-4):
        self.gamma = 0.99
        self.tau = 0.005 # For soft target updates
        
        # --- 1. Actor Network ---
        self.actor = ActorNetwork(n_actions)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # --- 2. Twin Critic Networks ---
        self.critic_1 = CriticNetwork(n_actions)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        
        self.critic_2 = CriticNetwork(n_actions)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        
        # --- 3. Target Networks (slow-moving copies) ---
        self.target_critic_1 = CriticNetwork(n_actions)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        
        self.target_critic_2 = CriticNetwork(n_actions)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # --- 4. Replay Buffer ---
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        
        # --- 5. Learnable Entropy (Alpha) ---
        self.target_entropy = -torch.prod(torch.Tensor(n_actions)).item() # Heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.alpha = self.log_alpha.exp().item()

    def update(self, batch_size):
        # 1. Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # --- Convert to Tensors ---
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # --------------------------------- #
        # --- 2. Calculate Critic Loss --- #
        # --------------------------------- #
        with torch.no_grad(): # We don't need gradients for the target calculation
            # Get next action and log_prob from the ACTOR
            next_actions, next_log_prob = self.actor(next_states)
            
            # Get next Q-values from the TARGET CRITICS
            q1_target = self.target_critic_1(next_states, next_actions)
            q2_target = self.target_critic_2(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # This is the "soft" part
            soft_next_value = min_q_target - self.alpha * next_log_prob
            
            # This is the Bellman equation
            target_q_value = rewards + (1 - dones) * self.gamma * soft_next_value

        # Get current Q-value predictions from the main critics
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        # Calculate the MSE loss
        critic_1_loss = F.mse_loss(current_q1, target_q_value)
        critic_2_loss = F.mse_loss(current_q2, target_q_value)
        critic_loss = critic_1_loss + critic_2_loss

        # Update the Critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # -------------------------------- #
        # --- 3. Calculate Actor Loss --- #
        # -------------------------------- #
        # Get new actions and log_probs from the actor
        new_actions, new_log_prob = self.actor(states)
        
        # Get Q-values for these new actions from the main critics
        q1_pred = self.critic_1(states, new_actions)
        q2_pred = self.critic_2(states, new_actions)
        min_q_pred = torch.min(q1_pred, q2_pred)
        
        # This is the actor's loss from our formula
        actor_loss = (self.alpha * new_log_prob - min_q_pred).mean()
        
        # Update the Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------------- #
        # --- 4. Calculate Alpha Loss --- #
        # ---------------------------------- #
        alpha_loss = -(self.log_alpha * (new_log_prob + self.target_entropy).detach()).mean()
        
        # Update Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item() # Update the alpha value
        
        # --- 5. Soft Update Target Networks ---
        self.soft_update_targets()

    def soft_update_targets(self):
        # This is the "Polyak averaging"
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)