import torch.optim as optim
from actor_critic_network import SACNetwork
import torch

class Agent:
    def __init__(self, n_actions, learning_rate=1e-4):
        # The agent's "brain"
        self.policy_network = SACNetwork(n_actions)
        
        # The optimizer for learning
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Memory for the current episode
        self.log_probs = []
        self.rewards = []
        self.state_values = [] # store critic state values

        self.gamma = 0.99  # Discount factor


    def update(self):
        # 1. Calculate discounted returns (G_t)
        # This part was correct!
        discounted_returns = []
        current_return = 0

        for reward in reversed(self.rewards):
            current_return = reward + self.gamma * current_return
            discounted_returns.insert(0, current_return) 

        # 2. Convert memory lists to tensors
        # We need all three: returns, log_probs, and the critic's state_values
        device = next(self.policy_network.parameters()).device
        
        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=device)
        log_probs_tensor = torch.stack(self.log_probs).to(device)
        
        # .squeeze() removes extra dimensions, e.g., from [batch, 1] to [batch]
        values_tensor = torch.stack(self.state_values).squeeze().to(device)


        # 3. Calculate Advantage (A_t = G_t - V(s_t))
        # This is the core of Actor-Critic
        advantage = returns_tensor - values_tensor
        
        # 4. Calculate Actor Loss (Policy Loss)
        # We want to maximize log_prob * advantage, so we minimize -(log_prob * advantage)
        # .detach() the advantage so that gradients from the actor loss
        # do NOT flow back into the critic. The critic has its own loss.
        actor_loss = -(log_probs_tensor * advantage.detach()).sum()

        # 5. Calculate Critic Loss (Value Loss)
        # This is a simple regression: make the critic's prediction (values_tensor)
        # match the actual observed return (returns_tensor)
        critic_loss = torch.nn.functional.mse_loss(returns_tensor, values_tensor)
        
        # 6. Calculate Total Loss
        # We sum the two losses. You can optionally weight the critic_loss
        # (e.g., total_loss = actor_loss + 0.5 * critic_loss)
        total_loss = actor_loss + critic_loss

        # 7. Perform the update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 8. Clear memory for the next episode
        # *** You were missing this one! ***
        self.log_probs = []
        self.rewards = []
        self.state_values = []