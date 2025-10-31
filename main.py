from racing_env import CarRacingEnv
from agent import SACAgent
import torch

# --- 1. Initialization ---
env = CarRacingEnv()

from racing_env import CarRacingEnv
from agent import SACAgent
import torch

# --- 1. Initialization ---
env = CarRacingEnv()

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create agent (uses ActorNetwork and CriticNetwork internally)
agent = SACAgent(n_actions=3, learning_rate=3e-4)

# Move networks to device
agent.actor.to(device)
agent.critic_1.to(device)
agent.critic_2.to(device)
agent.target_critic_1.to(device)
agent.target_critic_2.to(device)

total_rewards = []
total_episodes = 1000
batch_size = 64

# Checkpoint loading (actor + actor optimizer). If keys are missing, continue from scratch.
start_episode = 0
try:
    checkpoint = torch.load('car_agent_checkpoint.pth', map_location=device)
    if 'actor_state_dict' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    if 'actor_optimizer_state_dict' in checkpoint:
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        # Move optimizer tensors to device
        for state in agent.actor_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    start_episode = checkpoint.get('episode', 0)
    total_rewards = checkpoint.get('total_rewards', [])
    print(f"Resuming training from episode {start_episode}")
except FileNotFoundError:
    print("Starting training from scratch.")

# Put the actor in training mode
agent.actor.train()

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

        # Get action and log_prob from the actor
        with torch.no_grad():
            action, log_prob = agent.actor(state_tensor)

        # Detach action and convert to numpy for env
        action_np = action.detach().squeeze(0).cpu().numpy()

        # Take the action in the environment
        next_state, reward, done = env.step(action_np)

        # Shape the reward (example shaping; adapt as needed)
        shaped_reward = reward
        if abs(action_np[0]) > 0.8:
            shaped_reward -= 0.1
        shaped_reward -= 0.01

        # Store transition in replay buffer
        agent.replay_buffer.add(state, action_np, shaped_reward, next_state, done)

        # Update agent if enough samples
        if len(agent.replay_buffer) >= batch_size:
            agent.update(batch_size)

        state = next_state
        episode_reward += reward  # Keep original reward for logging

    # --- 4. Logging & Checkpointing ---
    total_rewards.append(episode_reward)
    print(f"Episode {episode+1}: Total Reward: {episode_reward:.2f}")

    if (episode + 1) % 10 == 0:  # Save every 10 episodes
        torch.save({
            'episode': episode + 1,
            'actor_state_dict': agent.actor.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'total_rewards': total_rewards,
        }, 'car_agent_checkpoint.pth')

env.close()