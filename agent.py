import torch.optim as optim
from policy_network import PolicyNetwork
import torch

class Agent:
    def __init__(self, n_actions, learning_rate=1e-4):
        # The agent's "brain"
        self.policy_network = PolicyNetwork(n_actions)
        
        # The optimizer for learning
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Memory for the current episode
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99  # Discount factor


    def update(self):
        # 1. Calculate discounted returns by looping backwards
        discounted_returns = []
        current_return = 0

        for reward in reversed(self.rewards):
            current_return = reward + self.gamma * current_return
            discounted_returns.insert(0, current_return) 

        # 2. Normalize the returns
        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # 3. Calulate Loss

        loss = []
        for log_prob, R in zip(self.log_probs, returns_tensor):
            loss.append(-log_prob * R) # The negative sign is for gradient ascent


        # 4. Perform the update
        self.optimizer.zero_grad()
        # Summing the loss before backprop is more efficient
        loss_tensor = torch.stack(loss).sum()
        loss_tensor.backward()
        self.optimizer.step()
        
        # 5. Clear memory for the next episode
        self.log_probs = []
        self.rewards = []