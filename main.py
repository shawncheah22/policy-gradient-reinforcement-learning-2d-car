from racing_env import CarRacingEnv
from agent import Agent
import torch

# --- 1. Initialization ---
env = CarRacingEnv()
agent = Agent(n_actions=3, learning_rate=1e-4)
total_rewards = []
total_episodes = 1000

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
        action_np = action.detach().squeeze(0).numpy()
        action_np[1] += 500
        
        # Take the action in the environment
        next_state, reward, done = env.step(action_np)
        
        # Store the reward and log_prob
        agent.rewards.append(reward)
        agent.log_probs.append(log_prob.sum()) # Sum log_probs for all actions
        
        state = next_state
        episode_reward += reward
        
    # --- 4. Perform the Update ---
    agent.update()
    
    # --- 5. Logging ---
    total_rewards.append(episode_reward)
    print(f"Episode {episode+1}: Total Reward: {episode_reward:.2f}")

    if (episode + 1) % 50 == 0:
        torch.save(agent.policy_network.state_dict(), 'car_agent_weights.pth')
        print(f"--- Model saved at episode {episode+1} ---")

env.close()