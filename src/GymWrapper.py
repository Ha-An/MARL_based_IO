import numpy as np
from config_SimPy import *
from config_MARL import *
from log_MARL import *
from environment_SimPy import *
from MAAC import *


class GymWrapper:
    """
    Wrapper class to handle the interaction between MAAC and Gym environment 
    Attributes:
        env (InventoryManagementEnv): Gym environment
        num_agents (int): Number of agents in the environment
        joint_action_space_size (spaces.MultiDiscrete): Size of the joint action space
        multi_state_space_size (int): Size of the multi-agent state space
        buffer_size (int): Size of the replay buffer
        batch_size (int): Size of the training batch
        maac (MAAC): Multi-Agent Actor-Critic system
        buffer (ReplayBuffer): Replay buffer for storing experience tuples
        logger (TensorboardLogger): Tensorboard logger for logging training and evaluation metrics
    """

    def __init__(self, env, num_agents, joint_action_space_size, multi_state_space_size, buffer_size, batch_size, lr_actor, lr_critic, gamma, tau, num_heads, hidden_dim):
        """ 
        Args:
            env: Gym environment
            num_agents: Number of agents in the environment
            joint_action_space_size: Size of the joint action space
            multi_state_space_size: Size of the multi-agent state space
            buffer_size: Size of the replay buffer
            batch_size: Size of the training batch
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter
            num_heads: Number of heads in attention mechanism
            hidden_dim: Hidden dimension of the networks
        """
        self.env = env
        self.num_agents = num_agents
        self.joint_action_space_size = joint_action_space_size
        self.multi_state_space_size = multi_state_space_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Initialize MAAC components with correct parameter order
        self.maac = MAAC(
            num_agents=self.num_agents,
            multi_state_space_size=self.multi_state_space_size,
            joint_action_space_size=self.joint_action_space_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            num_heads=num_heads,
            hidden_dim=hidden_dim
        )
        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size
        )

        # Initialize tensorboard logger
        self.logger = TensorboardLogger(num_agents)

    def train(self, num_episodes, eval_interval):
        """
        Train the MAAC system using the Gym environment

        Args:
            num_episodes: Number of training episodes
            eval_interval: Interval for evaluation
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            # Indicators for logging
            critic_loss_val = 0.0
            actor_losses_val = [0.0]*self.num_agents
            td_error_val = 0.0
            mean_q_val = 0.0
            std_q_val = 0.0
            policy_entropy_val = 0.0
            param_norms_val = (0.0, [0.0]*self.num_agents)

            # Epsilon
            if EPSILON_DECAY_TYPE == 'linear':
                fraction = min(float(episode) / float(num_episodes), 1.0)
                epsilon = EPSILON_START + fraction * \
                    (EPSILON_END - EPSILON_START)
            elif EPSILON_DECAY_TYPE == 'exponential':
                epsilon = EPSILON_END + \
                    (EPSILON_START - EPSILON_END) * (DECAY_RATE ** episode)

            # For Action Distribution Logging
            action_distribution = [[] for _ in range(self.num_agents)]
            episode_length = 0

            while not done:
                actions = []
                # Select actions for each agent (optional: with probs)
                for i in range(self.num_agents):
                    action = self.maac.select_action(state, i, epsilon)
                    actions.append(action)
                    action_distribution[i].append(action)

                next_state, reward, done, info = self.env.step(actions)

                # Store in replay buffer
                self.buffer.push((state, actions, reward, next_state, done))

                if len(self.buffer) >= self.batch_size:
                    critic_loss_val, actor_losses_val, td_error_val, mean_q_val, \
                        std_q_val, policy_entropy_val, param_norms_val = \
                        self.maac.update(self.batch_size, self.buffer)

                self.env.total_reward += reward
                state = next_state
                episode_length += 1

            # Log to Tensorboard
            self.logger.log_training_info(
                episode=episode,
                episode_reward=self.env.total_reward,
                critic_loss=critic_loss_val,
                actor_losses=actor_losses_val,
                epsilon=epsilon,
                q_values=(mean_q_val, std_q_val),
                policy_entropy=policy_entropy_val,
                action_distribution=action_distribution,
                kl_divergence=None,
                td_error=td_error_val,
                episode_length=episode_length,
                param_norms=param_norms_val
            )

            # Print info every eval_interval
            if episode % eval_interval == 0:
                print(
                    f"Episode {episode} | Epsilon {epsilon} | Total Reward {self.env.total_reward:.3f}")
                print("-"*50)

    def evaluate(self, num_episodes):
        """
        Evaluate the trained MAAC system

        Args: 
            num_episodes (int): Number of evaluation episodes

        Returns:
            Average reward over the evaluation episodes (float)
        """
        rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select actions without exploration
                actions = []
                for i in range(self.num_agents):
                    action = self.maac.select_action(state, i, epsilon=0)
                    actions.append(action)

                state, reward, done, info = self.env.step(actions)
                episode_reward += reward

                # self.env.render()  # Visualize the environment state

            avg_daily_cost = -episode_reward/self.env.current_day

            # Log evaluation information
            self.logger.log_evaluation_info(
                episode=episode,
                total_reward=episode_reward,
                avg_daily_cost=avg_daily_cost
            )

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(f"Average Daily Cost: {avg_daily_cost}")
            print(f"Order quantities: {info['Order quantities']}")
            print("-" * 50)

            rewards.append(episode_reward)

        return sum(rewards) / num_episodes

    def save_model(self, episode, reward):
        """
        (로직 삽입 보류)
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
