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
    """

    def __init__(self, env, num_agents, joint_action_space_size, multi_state_space_size, buffer_size, batch_size, lr_actor, lr_critic, gamma):
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
            gamma=gamma
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
            episodes: Number of training episodes
            eval_interval: Interval for evaluation and printing results
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            critic_loss = 0
            actor_losses = [0] * self.num_agents
            epsilon = EPSILON_END + \
                (EPSILON_START - EPSILON_END) * (DECAY_RATE ** episode)

            while not done:
                # Select actions for each agent
                actions = []
                for i in range(self.num_agents):
                    action = self.maac.select_action(state, i, epsilon)
                    actions.append(action)

                # Execute actions in environment
                next_state, reward, done, info = self.env.step(actions)

                # print("next_state: ", next_state)
                # print("reward: ", reward)

                # Store an experience in buffer
                self.buffer.push(state, actions,
                                 reward, next_state, done)

                # Update networks
                if len(self.buffer) >= self.batch_size:
                    critic_loss, actor_losses = self.maac.update(
                        self.batch_size, self.buffer)

                episode_reward += reward
                state = next_state

                # Print simulation events
                if PRINT_DAILY_EVENTS:
                    print(info)

                if done and LOG_STATE:
                    print(f"Episode {episode}")
                    print(f"LOG_STATE_REAL")
                    for i in LOG_STATE_REAL:
                        print(i)
                    print(f"LOG_STATE_NOR")
                    for i in LOG_STATE_NOR:
                        print(i)

            # Log training information
            self.logger.log_training_info(
                episode=episode,
                episode_reward=episode_reward,
                critic_loss=critic_loss,
                actor_losses=actor_losses,
                epsilon=epsilon
            )
            # Evaluation and saving best model
            if episode % eval_interval == 0:
                print(f"Episode {episode}")
                print(f"Epsilon {epsilon}")
                print(f"Episode Reward: {episode_reward}")
                print("-" * 50)
                '''
                # 중간에 모델 저장하는 기능
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.save_model(episode, episode_reward)
                '''

    def evaluate(self, num_episodes):
        """
        Evaluate the trained MAAC system

        Args:
            num_episodes: Number of evaluation episodes
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
                avg_daily_cost=avg_daily_cost,
                inventory_levels=info['inventory_levels']
            )

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(f"Average Daily Cost: {avg_daily_cost}")
            print(f"Inventory Levels: {info['inventory_levels']}")
            print(f"Order quantities: {info['Order quantities']}")
            print("-" * 50)

            rewards.append(episode_reward)

        return sum(rewards) / num_episodes

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
