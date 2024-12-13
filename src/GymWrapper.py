import numpy as np
from config_SimPy import *
from environment import *
from MAAC import *


class GymWrapper:
    """
    Wrapper class to handle the interaction between MAAC and Gym environment

    Args:
        env: Gym environment
        n_agents (int): Number of agents
        action_dim (int): Dimension of action space
        obs_dim (int): Dimension of observation space (default=2: on_hand and in_transit inventory)
        hidden_dim (int): Hidden layer dimension
        buffer_size (int): Replay buffer size
        batch_size (int): Batch size for training
        lr (float): Learning rate
        gamma (float): Discount factor
    """

    def __init__(self, env, n_agents, action_dim, buffer_size, batch_size, lr, gamma, obs_dim=2, hidden_dim=64):
        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.batch_size = batch_size

        # Initialize MAAC components
        self.maac = MAAC(n_agents, obs_dim, action_dim,
                         len(P) + 2, lr, gamma)  # global state dim = WIPs + 2
        self.buffer = ReplayBuffer(buffer_size, obs_dim, n_agents, action_dim)

    def train(self, episodes, eval_interval):
        """
        Train the MAAC system using the Gym environment

        Args:
            episodes: Number of training episodes
            eval_interval: Interval for evaluation and printing results
        """
        best_reward = float('-inf')
        for episode in range(episodes):
            observations = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Extract local observations for each agent
                local_obs = observations['local_obs']

                # Select actions for each agent
                actions = []
                for i in range(self.n_agents):
                    epsilon = max(0.1, 1.0 - episode/500)  # Decreasing epsilon
                    action = self.maac.select_action(local_obs[i], i, epsilon)
                    actions.append(action)

                # Execute actions in environment
                next_observations, reward, done, info = self.env.step(actions)

                # Store transition in buffer
                self.buffer.push(
                    local_obs,
                    observations['global_obs'],
                    np.array(actions),
                    reward,
                    next_observations['local_obs'],
                    next_observations['global_obs'],
                    done
                )

                # Update networks
                self.maac.update(self.batch_size, self.buffer)

                episode_reward += reward
                observations = next_observations

            # Evaluation and logging
            if episode % eval_interval == 0:
                print(f"Episode {episode}")
                print(f"Episode Reward: {episode_reward}")
                print(f"Average Cost: {-episode_reward/self.env.current_day}")
                print("Inventory Levels:", info['inventory_levels'])
                print("-" * 50)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    # Could save best model here

                    model.save(os.path.join(
                        SAVED_MODEL_PATH, SAVED_MODEL_NAME))
                    print(f"{SAVED_MODEL_NAME} is saved successfully")

        # return self.maac

    def evaluate(self, episodes):
        """
        Evaluate the trained MAAC system

        Args:
            episodes: Number of evaluation episodes
        """
        for episode in range(episodes):
            observations = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select actions without exploration
                actions = []
                for i in range(self.n_agents):
                    action = self.maac.select_action(
                        observations['local_obs'][i], i, epsilon=0)
                    actions.append(action)

                observations, reward, done, info = self.env.step(actions)
                episode_reward += reward

                self.env.render()  # Visualize the environment state

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(
                f"Average Daily Cost: {-episode_reward/self.env.current_day}")
            print("-" * 50)