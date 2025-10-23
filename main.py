from racing_env import CarRacingEnv
from agent import Agent
import torch

# --- 1. Initialization ---
env = CarRacingEnv()
agent = Agent(n_actions=3, learning_rate=3e-4)  # Increased learning rate for faster learning
total_rewards = []
total_episodes = 1000

# Check if a checkpoint exists and load it
try:
    checkpoint = torch.load('car_agent_checkpoint.pth')
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    total_rewards = checkpoint['total_rewards']
    print(f"Resuming training from episode {start_episode}")
except FileNotFoundError:
    print("Starting training from scratch.")

# Put the model in training mode
agent.policy_network.train()
# --- 2. The Main Training Loop ---
for episode in range(total_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    episode_reward = 0
    done = False
    
    # --- 3. The Episode Loop ---
    while not done:
        # Convert state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get action and log_prob from the policy network
        action, log_prob = agent.policy_network(state_tensor)
        
        # Detach action from the computation graph and convert to numpy
        action_np = torch.tanh(action).detach().squeeze(0).numpy()  # Use tanh to bound actions to [-1, 1]
        
        # Process the actions for the environment
        # [Steering, Gas, Brake]
        processed_action = action_np.copy()
        processed_action[1] = (action_np[1] + 1) / 2  # Convert gas from [-1,1] to [0,1]
        processed_action[2] = (action_np[2] + 1) / 2  # Convert brake from [-1,1] to [0,1]
        
        # Take the action in the environment
        next_state, reward, done = env.step(processed_action)
        
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