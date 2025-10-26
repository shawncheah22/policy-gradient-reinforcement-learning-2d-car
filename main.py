from racing_env import CarRacingEnv
from agent import Agent
import torch

# --- 1. Initialization ---
env = CarRacingEnv()

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(n_actions=3, learning_rate=3e-4)  # Increased learning rate for faster learning
# Move the policy network to the chosen device
agent.policy_network.to(device)

total_rewards = []
total_episodes = 1000

# Check if a checkpoint exists and load it onto the correct device
start_episode = 0
try:
    checkpoint = torch.load('car_agent_checkpoint.pth', map_location=device)
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Move optimizer state tensors to device (if any tensors are present)
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    start_episode = checkpoint.get('episode', 0)
    total_rewards = checkpoint.get('total_rewards', [])
    print(f"Resuming training from episode {start_episode}")
except FileNotFoundError:
    print("Starting training from scratch.")

# Put the model in training mode
agent.policy_network.train()

# --- 2. The Main Training Loop ---
for episode in range(start_episode, total_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    episode_reward = 0
    done = False
    
    # --- 3. The Episode Loop ---
    while not done:
        # Convert state to a PyTorch tensor and move to device
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Get action and log_prob from the policy network
        action, log_prob, state_value = agent.policy_network.forward(state_tensor)

        # Detach action from the computation graph and convert to numpy
        # move to CPU before converting to numpy
        action_np = action.detach().squeeze(0).cpu().numpy()

        # Take the action in the environment
        next_state, reward, done = env.step(action_np)

        # Shape the reward
        shaped_reward = reward

        # Penalize excessive steering
        if abs(action_np[0]) > 0.8:
            shaped_reward -= 0.1

        # Small penalty per timestep to encourage faster completion
        shaped_reward -= 0.01

        # Store the shaped reward and log_prob
        agent.rewards.append(shaped_reward)
        agent.log_probs.append(log_prob.sum()) # Sum log_probs for all actions
        agent.state_values.append(state_value) 

        state = next_state
        episode_reward += reward  # Keep original reward for logging
        
    # --- 4. Perform the Update ---
    agent.update()
    
    # --- 5. Logging ---
    total_rewards.append(episode_reward)
    print(f"Episode {episode+1}: Total Reward: {episode_reward:.2f}")

    if (episode + 1) % 10 == 0: # Save every 10 episodes
            torch.save({
                'episode': episode + 1,
                'model_state_dict': agent.policy_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'total_rewards': total_rewards,
            }, 'car_agent_checkpoint.pth')

env.close()