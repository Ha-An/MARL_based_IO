import numpy as np
from config_SimPy import *
from config_MARL import *
from log_MARL import *
from environment import *
from MAAC import *


class GymWrapper:
    """
    Wrapper class to handle the interaction between MAAC and Gym environment

    Args:
        env (gym.Env): Gym environment
        n_agents (int): Number of agents in the environment
        action_dim (int): Dimension of the action space
        state_dim (int): Dimension of the state space
        buffer_size (int): Size of the replay buffer
        batch_size (int): Batch size for training (unit: episodes)
        lr (float): Learning rate for the actor and critic networks
        gamma (float): Discount factor for future rewards
        hidden_dim (int): Hidden dimension for actor and critic networks
    """

    def __init__(self, env, n_agents, action_dim, state_dim, buffer_size, batch_size, lr, gamma, hidden_dim=64):
        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size

        # Initialize MAAC components with correct parameter order
        self.maac = MAAC(
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma
        )
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            n_agents=n_agents,
            action_dim=action_dim,
            state_dim=state_dim
        )

        # Initialize tensorboard logger
        self.logger = TensorboardLogger(n_agents)

        # Log hyperparameters
        self.logger.log_hyperparameters({
            'n_agents': n_agents,
            'action_dim': action_dim,
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'learning_rate': lr,
            'gamma': gamma
        })

    def train(self, num_episodes, eval_interval):
        """
        Train the MAAC system using the Gym environment

        Args:
            episodes: Number of training episodes
            eval_interval: Interval for evaluation and printing results
        """
        best_reward = float('-inf')
        for episode in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0
            done = False
            critic_loss = 0
            actor_losses = [0] * self.n_agents
            epsilon = max(0.1, 1.0 - episode/500)

            while not done:
                # Select actions for each agent
                actions = []
                for i in range(self.n_agents):
                    action = self.maac.select_action(states[i], i, epsilon)
                    actions.append(action)

                # Execute actions in environment
                next_states, reward, done, info = self.env.step(actions)

                # Store transition in buffer
                self.buffer.push(states, np.array(actions),
                                 reward, next_states, done)

                episode_reward += reward
                states = next_states

                # Print simulation events
                if PRINT_SIM_EVENTS:
                    print(info)

            # If we have enough complete episodes, perform training
            if len(self.buffer) >= self.batch_size:
                critic_loss, actor_losses = self.maac.update(
                    self.batch_size, self.buffer)
                # print(f"Episode {episode} - Critic Loss: {critic_loss}")

            # Log training information
            avg_cost = -episode_reward/self.env.current_day
            self.logger.log_training_info(
                episode=episode,
                episode_reward=episode_reward,
                avg_cost=avg_cost,
                inventory_levels=info['inventory_levels'],
                critic_loss=critic_loss,
                actor_losses=actor_losses,
                epsilon=epsilon
            )

            # Evaluation and saving best model
            if episode % eval_interval == 0:
                print(f"Episode {episode}")
                print(f"Episode Reward: {episode_reward}")
                print("-" * 50)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.save_model(episode, episode_reward)

        # Check for device usage warnings
        self.maac.check_device_usage_warnings()

    def evaluate(self, num_episodes):
        """
        Evaluate the trained MAAC system

        Args:
            num_episodes: Number of evaluation episodes
        """
        for episode in range(num_episodes):
            observations = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select actions without exploration
                actions = []
                for i in range(self.n_agents):
                    action = self.maac.select_action(
                        observations[i], i, epsilon=0)
                    actions.append(action)

                observations, reward, done, info = self.env.step(actions)
                episode_reward += reward

                # self.env.render()  # Visualize the environment state

            avg_daily_cost = -episode_reward/self.env.current_day

            # Log evaluation information
            self.logger.log_evaluation_info(
                episode=episode,
                total_reward=episode_reward,
                avg_daily_cost=avg_daily_cost,
                inventory_levels=info['inventory_levels']
            )

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(f"Average Daily Cost: {avg_daily_cost}")
            print(f"Inventory Levels: {info['inventory_levels']}")
            print(f"Order quantities: {info['Order quantities']}")
            print("-" * 50)

        # Check for device usage warnings
        self.maac.check_device_usage_warnings()

    def save_model(self, episode, reward):
        """
        Save the model to the specified path

        Args: 
            episode (int): Current episode number
            reward (float): Current episode reward 
        """
        # Save best model
        model_path = os.path.join(
            MODEL_DIR, f"maac_best_model_episode_{episode}.pt")
        torch.save({
            'episode': episode,
            'best_reward': reward,
            'critic_state_dict': self.maac.critic.state_dict(),
            'actors_state_dict': [actor.state_dict() for actor in self.maac.actors],
            'target_critic_state_dict': self.maac.critic_target.state_dict(),
            'target_actors_state_dict': [target_actor.state_dict() for target_actor in self.maac.actors_target],
            'critic_optimizer_state_dict': self.maac.critic_optimizer.state_dict(),
            'actors_optimizer_state_dict': [optimizer.state_dict() for optimizer in self.maac.actor_optimizers]
        }, model_path)
        # print(f"Saved best model with reward {best_reward} to {model_path}")

    def load_model(self, model_path):
        """
        Load a saved model

        Args:
            model_path (str): Path to the saved model
        """
        checkpoint = torch.load(model_path)

        # Load model states
        self.maac.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.maac.target_critic.load_state_dict(
            checkpoint['target_critic_state_dict'])

        for i, actor_state_dict in enumerate(checkpoint['actors_state_dict']):
            self.maac.actors[i].load_state_dict(actor_state_dict)

        for i, target_actor_state_dict in enumerate(checkpoint['target_actors_state_dict']):
            self.maac.actors_target[i].load_state_dict(target_actor_state_dict)

        # Load optimizer states
        self.maac.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])

        for i, actor_opt_state_dict in enumerate(checkpoint['actors_optimizer_state_dict']):
            self.maac.actor_optimizers[i].load_state_dict(actor_opt_state_dict)

        print(
            f"Loaded model from episode {checkpoint['episode']} with best reward {checkpoint['best_reward']}")

    def __del__(self):
        """Cleanup method to ensure tensorboard writer is closed"""
        if hasattr(self, 'logger'):
            self.logger.close()
