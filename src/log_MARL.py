from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime


# Log daily repots
LIST_STATE_REAL = [[]]  # Real State
''' 
    'On-hand inventory' (list): Product, WIPs, Materials; 
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [10, 5, 20, 30, 40]
    'In-transition inventory' (list): Materials; 
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [3, 5, 2]
    'Remaining demand' (int): Product;
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): 10
'''
LIST_STATE_NOR = [[]]  # Normalized State (0~1)
# STATE_ACTION_REPORT_REAL = [[]]  # Real State
COST_RATIO_HISTORY = []

# Record the cumulative value of each cost component
LIST_TOTAL_COST_COMP = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}


class TensorboardLogger:
    """
    Tensorboard logging utility for MAAC training

    Args:
        log_dir (str): Directory to save tensorboard logs
        n_agents (int): Number of agents in the environment
    """

    def __init__(self, n_agents, log_dir='runs'):
        # Create unique run name with timestamp
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f'MAAC_run_{current_time}'

        # Create log directory
        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.log_dir)
        self.n_agents = n_agents

        print(f"Tensorboard logging to {self.log_dir}")
        print("To view training progress, run:")
        print(f"tensorboard --logdir={log_dir}")

    # def log_training_info(self, episode, episode_reward, critic_loss=None, actor_losses=None, epsilon=None):
    #     """
    #     Log training metrics to tensorboard

    #     Args:
    #         episode (int): Current episode number
    #         episode_reward (float): Total reward for the episode
    #         avg_cost (float): Average cost per day for the episode
    #         inventory_levels (dict): Dictionary of inventory levels for each agent
    #         critic_loss (float, optional): Loss value of the critic network
    #         actor_losses (list, optional): List of loss values for each actor network
    #         epsilon (float, optional): Current exploration rate
    #     """
    #     # Log episode metrics
    #     self.writer.add_scalar('Training/Episode_Reward',
    #                            episode_reward, episode)

    #     # # Log inventory levels for each agent
    #     # for agent_id, level in inventory_levels.items():
    #     #     self.writer.add_scalar(
    #     #         f'Inventory/Agent_{agent_id}', level, episode)

    #     # Log network losses if provided
    #     if critic_loss is not None:
    #         self.writer.add_scalar('Loss/Critic', critic_loss, episode)

    #     if actor_losses is not None:
    #         for i, loss in enumerate(actor_losses):
    #             self.writer.add_scalar(f'Loss/Actor_{i}', loss, episode)

    #     # Log exploration rate if provided
    #     if epsilon is not None:
    #         self.writer.add_scalar('Training/Epsilon', epsilon, episode)
    def log_training_info(self, episode, episode_reward,
                          critic_loss=None, actor_losses=None, epsilon=None,
                          q_values=None, policy_entropy=None,
                          action_distribution=None, kl_divergence=None,
                          td_error=None, episode_length=None,
                          param_norms=None):
        """
        Log extended training metrics to TensorBoard.
        """

        # 기본 에피소드 리워드
        self.writer.add_scalar('Training/Episode_Reward',
                               episode_reward, episode)

        # Critic Loss
        if critic_loss is not None:
            self.writer.add_scalar('Loss/Critic', critic_loss, episode)

        # Actor Losses
        if actor_losses is not None:
            for i, loss in enumerate(actor_losses):
                self.writer.add_scalar(f'Loss/Actor_{i}', loss, episode)

        # Epsilon
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)

        # Q-Value 통계 (평균/표준편차)
        if q_values is not None:
            mean_q, std_q = q_values
            self.writer.add_scalar('Q-Value/Mean', mean_q, episode)
            self.writer.add_scalar('Q-Value/StdDev', std_q, episode)

        # Policy Entropy
        if policy_entropy is not None:
            self.writer.add_scalar('Policy/Entropy', policy_entropy, episode)

        # Action Distribution (히스토그램)
        if action_distribution is not None:
            for i, acts in enumerate(action_distribution):
                # acts는 매 스텝마다 기록된 action이 들어있는 list
                self.writer.add_histogram(
                    f'Actions/Agent_{i}', np.array(acts), episode)

        # KL Divergence
        if kl_divergence is not None:
            self.writer.add_scalar(
                'Policy/KL_Divergence', kl_divergence, episode)

        # TD Error
        if td_error is not None:
            self.writer.add_scalar('Loss/TD_Error', td_error, episode)

        # Episode Length
        if episode_length is not None:
            self.writer.add_scalar(
                'Training/Episode_Length', episode_length, episode)

        # Parameter Norms (Critic, Actors)
        if param_norms is not None:
            critic_norm, actor_norms = param_norms
            self.writer.add_scalar(
                'Parameters/Critic_Norm', critic_norm, episode)
            for i, norm in enumerate(actor_norms):
                self.writer.add_scalar(
                    f'Parameters/Actor_{i}_Norm', norm, episode)

    def log_evaluation_info(self, episode, total_reward, avg_daily_cost):
        """
        Log evaluation metrics to tensorboard

        Args:
            episode (int): Current evaluation episode
            total_reward (float): Total reward for the evaluation episode
            avg_daily_cost (float): Average daily cost during evaluation
            inventory_levels (dict): Dictionary of inventory levels for each agent
        """
        self.writer.add_scalar(
            'Evaluation/Episode_Reward', total_reward, episode)
        self.writer.add_scalar(
            'Evaluation/Average_Daily_Cost', avg_daily_cost, episode)

    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()
