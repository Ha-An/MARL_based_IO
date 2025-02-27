from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime


# Log daily repots
LOG_STATE_REAL = [[]]  # Real State
''' 
    'On-hand inventory' (list): Product, WIPs, Materials; 
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [10, 5, 20, 30, 40]
    'In-transition inventory' (list): Materials; 
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [3, 5, 2]
    'Remaining demand' (int): Product;
        e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): 10
'''
LOG_STATE_NOR = [[]]  # Normalized State (0~1)
LOG_ACTION = [[]]
# STATE_ACTION_REPORT_REAL = [[]]  # Real State
COST_RATIO_HISTORY = []

# Record the cumulative value of each cost component
LOG_TOTAL_COST_COMP = {
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

    def log_training_info(self, episode, episode_reward, critic_loss=None, actor_losses=None, epsilon=None):
        """
        Log training metrics to tensorboard

        Args:
            episode (int): Current episode number
            episode_reward (float): Total reward for the episode
            avg_cost (float): Average cost per day for the episode
            inventory_levels (dict): Dictionary of inventory levels for each agent
            critic_loss (float, optional): Loss value of the critic network
            actor_losses (list, optional): List of loss values for each actor network
            epsilon (float, optional): Current exploration rate
        """
        # Log episode metrics
        self.writer.add_scalar('Training/Episode_Reward',
                               episode_reward, episode)

        # # Log inventory levels for each agent
        # for agent_id, level in inventory_levels.items():
        #     self.writer.add_scalar(
        #         f'Inventory/Agent_{agent_id}', level, episode)

        # Log network losses if provided
        if critic_loss is not None:
            self.writer.add_scalar('Loss/Critic', critic_loss, episode)

        if actor_losses is not None:
            for i, loss in enumerate(actor_losses):
                self.writer.add_scalar(f'Loss/Actor_{i}', loss, episode)

        # Log exploration rate if provided
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)

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
