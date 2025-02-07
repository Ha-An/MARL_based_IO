import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_MARL import *
from environment import *
from log_SimPy import *
from log_MARL import *


class InventoryManagementEnv(gym.Env):
    """
    Gym environment for multi-agent inventory management system
    Handles the simulation of inventory management with multiple procurement agents
    """

    def __init__(self):
        super(InventoryManagementEnv, self).__init__()
        self.scenario = {"DEMAND": DEMAND_SCENARIO,
                         "LEADTIME": LEADTIME_SCENARIO}
        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.cur_episode = 1  # Current episode
        self.cur_outer_loop = 1  # Current outer loop
        self.cur_inner_loop = 1  # Current inner loop
        self.scenario_batch_size = 99999  # Initialize the scenario batch size
        self.current_day = 0  # Initialize the current day
        self.n_agents = MAT_COUNT  # Number of agents

        # Record the cumulative value of each cost
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Define action space
        """
        Action space is a MultiDiscrete space where each agent can choose an order quantity
        """
        action_dims = []
        for i in range(len(I)):
            if I[i]["TYPE"] == "Material":
                action_dims.append(ACTION_MAX)
        self.action_space = spaces.MultiDiscrete(action_dims)
        self.action_dim_size = ACTION_MAX + 1

        # Define unified observation space
        '''
        State space is a MultiDiscrete space where each agent observes the following:
            On-hand inventory level for each item: len(I)
            In-transition inventory level for each material: MAT_COUNT
            Remaining demand: 1  
        '''
        obs_dims = []
        # On-hand inventory levels for all items
        for _ in range(len(I)):
            obs_dims.append(INVEN_LEVEL_MAX - INVEN_LEVEL_MIN + 1)
        # In-transition inventory levels for material items
        for _ in range(MAT_COUNT):
            obs_dims.append(ACTION_MAX - ACTION_MIN + 1)
        # Remaining demand
        obs_dims.append(ACTION_MAX - ACTION_MIN + 1)
        # Define observation space as MultiDiscrete
        self.observation_space = spaces.MultiDiscrete(obs_dims)
        self.state_dim_size = len(obs_dims)

        # Initialize simulation environment
        self.reset()

    def reset(self):
        """
        Reset the environment to initial state

        Returns:
            states: Initial state array for each agent
        """
        # Initialize the total reward for the episode
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Create new SimPy environment and components
        self.simpy_env, self.inventory_list, self.procurement_list, self.production_list, \
            self.sales, self.customer, self.supplier_list, self.daily_events = create_env(
                I, P, LOG_DAILY_EVENTS
            )

        # Initialize simulation processes
        scenario = {
            "DEMAND": DEMAND_SCENARIO,
            "LEADTIME": LEADTIME_SCENARIO
        }
        simpy_event_processes(
            self.simpy_env, self.inventory_list, self.procurement_list,
            self.production_list, self.sales, self.customer, self.supplier_list,
            self.daily_events, I, scenario
        )
        update_daily_report(self.inventory_list)

        self.current_day = 0
        self.total_reward = 0
        self.shortages = 0

        return self._get_observations()

    def step(self, actions):
        """
        Execute one time step (1 day) in the environment

        Args:
            actions: Array of order quantities for each material agent

        Returns:
            observations: State array for each agent
            reward: Negative total cost for the day
            done: Whether the episode has ended
            info: Additional information for debugging
        """
        # Set order quantities for each material agent
        for i, action in enumerate(actions):
            # print(f"Material_{i} order quantity: {action}")
            I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"] = int(action)

        # Run simulation for one day
        STATE_ACTION_REPORT_REAL[-1].append(actions)
        self.simpy_env.run(until=(self.current_day + 1) * 24)
        self.current_day += 1
        update_daily_report(self.inventory_list)

        # Get next observations
        next_states = self._get_observations()

        # Calculate reward (a negative value of the daily total cost)
        reward = -Cost.update_cost_log(self.inventory_list)
        # Update LOG_TOTAL_COST_COMP
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] += DAILY_COST[key]
        Cost.clear_cost()

        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        # Check if episode is done
        done = self.current_day >= SIM_TIME

        # Additional info for debugging
        info = {
            'Day': self.current_day,
            'Daily cost': -reward,
            'Total cost': -self.total_reward,
            'inventory_levels': {
                f"Material_{i}": inv.on_hand_inventory
                for i, inv in enumerate(self.inventory_list)
                if I[inv.item_id]['TYPE'] == "Material"
            },
            'Order quantities': actions
        }

        return next_states, reward, done, info

    def _get_observations(self):
        """
        Construct unified state observation array

        Returns:
            numpy array with shape [n_agents, self.state_dim_size]
        """
        # Initialize single state array
        state = np.zeros(self.state_dim_size, dtype=np.int32)
        state_idx = 0

        # Add on-hand inventory levels for all items
        for inv in self.inventory_list:
            state[state_idx] = np.clip(
                inv.on_hand_inventory,
                INVEN_LEVEL_MIN,
                INVEN_LEVEL_MAX
            )
            state_idx += 1

        # Add in-transit inventory levels for material items
        for inv in self.inventory_list:
            if I[inv.item_id]["TYPE"] == "Material":
                state[state_idx] = np.clip(
                    inv.in_transition_inventory,
                    ACTION_MIN,
                    ACTION_MAX
                )
                state_idx += 1

        # Add remaining demand
        remaining_demand = I[0]['DEMAND_QUANTITY'] - \
            self.inventory_list[0].on_hand_inventory
        state[state_idx] = np.clip(
            remaining_demand,
            ACTION_MIN,
            ACTION_MAX
        )

        # Copy the same state for all agents
        states = np.tile(state, (self.n_agents, 1))

        return states

    def render(self, mode='human'):
        """
        Render the environment's current state
        Currently just prints basic information
        """
        if mode == 'human':
            print(f"\nDay: {self.current_day}")
            print("\nInventory Levels:")
            for inv in self.inventory_list:
                print(f"{I[inv.item_id]['NAME']}: {inv.on_hand_inventory} "
                      f"(In Transit: {inv.in_transition_inventory})")

    def close(self):
        """Clean up environment resources"""
        pass
