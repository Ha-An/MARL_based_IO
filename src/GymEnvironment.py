import gym
import numpy as np
from config_SimPy import *
from config_MARL import *
from environment_SimPy import *
from log_SimPy import *
from log_MARL import *


class InventoryManagementEnv(gym.Env):
    """
    Gym environment for multi-agent inventory management system
    Handles the simulation of inventory management with multiple procurement agents 

    Attributes:
        scenario (dict): Dictionary of demand and lead time scenarios
        shortages (int): Number of shortages in the current episode
        total_reward_over_episode (list): Total reward over the current episode
        total_reward (float): Total reward over the current episode
        cur_episode (int): Current episode number
        cur_outer_loop (int): Current outer loop number --> Not used. This is for Meta-RL
        cur_inner_loop (int): Current inner loop number --> Not used. This is for Meta-RL
        scenario_batch_size (int): Scenario batch size --> Not used. This is for Meta-RL
        current_day (int): Current day in the simulation
        num_agents (int): Number of agents
        joint_action_space_size (spaces.MultiDiscrete): Size of the joint action space
        multi_state_space_size (int): Size of the multi-agent state space
        simpy_env (simpy.Environment): SimPy simulation environment
        inventory_list (list): List of inventory objects
        procurement_list (list): List of procurement objects
        production_list (list): List of production objects
        sales (Sales): Sales object
        customer (Customer): Customer object
        supplier_list (list): List of supplier objects
        daily_events (list): List of daily events
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
        self.num_agents = NUM_AGENTS  # Number of agents

        # Define action space
        self.joint_action_space_size = JOINT_ACTION_SPACE_SIZE

        # Define state space
        self.multi_state_space_size = MULTI_STATE_SPACE_SIZE

        # Initialize simulation environment
        self.reset()

    def reset(self):
        """
        Reset the environment to initial state

        Returns:
            observations: State array for each agent
        """
        # Initialize log data
        LIST_DAILY_EVENTS.clear()
        LIST_LOG_COST.clear()
        LIST_LOG_DAILY_REPORTS.clear()
        LIST_LOG_STATE_DICT.clear()
        DICT_DAILY_COST = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }

        # Initialize the total reward for the episode
        for key in DICT_DAILY_COST.keys():
            LIST_TOTAL_COST_COMP[key] = 0

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

        # print("LIST_DAILY_EVENTS: ", len(LIST_DAILY_EVENTS))
        # print("LOG_COST: ", len(LOG_COST))
        # print("LOG_DAILY_REPORTS: ", len(LOG_DAILY_REPORTS))
        # print("LOG_STATE_DICT: ", len(LOG_STATE_DICT))
        # print("DICT_DAILY_COST: ", DICT_DAILY_COST)
        # print("GRAPH_LOG: ", len(GRAPH_LOG))

        # print("\nLIST_STATE_REAL: ", LIST_STATE_REAL)
        # print("LIST_STATE_NOR: ", LIST_STATE_NOR)
        # print("COST_RATIO_HISTORY: ", COST_RATIO_HISTORY)
        # print("LIST_TOTAL_COST_COMP: ", LIST_TOTAL_COST_COMP)

        return self._get_observations()

    def step(self, actions):
        """
        Execute one time step (1 day) in the environment

        Args:
            actions: Array of order quantities for each material agent

        Returns:
            next_states: Next state array for each agent
            reward: Reward for the current time step
            done: Boolean indicating if the episode is done
            info: Additional information for debugging
        """
        # Set order quantities for each material agent
        # print("actions: ", actions)
        if USE_SQPOLICY:
            for i, action in enumerate(actions):
                if self.inventory_list[self.procurement_list[i].item_id].on_hand_inventory <= SQPAIR['Reorder']:
                    I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"] = SQPAIR['Order']
                else:
                    I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"] = 0
        else:
            for i, action in enumerate(actions):
                # print(f"Material_{i} order quantity: {action}")
                I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"] = int(
                    action)
        # for i, action in enumerate(actions):
        #     print("Action (", i, "): ",
        #           I[self.procurement_list[i].item_id]["LOT_SIZE_ORDER"])

        # Run simulation for one day (24 hours)
        self.simpy_env.run(until=(self.current_day + 1) * 24)
        self.current_day += 1
        update_daily_report(self.inventory_list)

        # Get next observations
        next_states = self._get_observations()

        # Calculate reward (a negative value of the daily total cost)
        reward = -Cost.update_cost_log(self.inventory_list)

        ''' Reward scaling '''
        reward = reward / 1000

        # Update LIST_TOTAL_COST_COMP
        for key in DICT_DAILY_COST.keys():
            LIST_TOTAL_COST_COMP[key] += DICT_DAILY_COST[key]
        Cost.clear_cost()

        # Update total reward and shortages
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

    def _normalize_state(self, state):
        """Normalize state to range [0,1]"""

        # Normalize state to range [0,1]
        normalized_state = (state - STATE_MINS) / \
            (STATE_MAXS - STATE_MINS + 1e-8)

        # Clip values to [0,1]
        normalized_state = np.clip(normalized_state, 0, 1)

        return normalized_state

    def _get_observations(self):
        """
        Returns the fully observable state of the system for all agents.
        Since this is a fully observable system, all agents receive the same global state.

        Returns:
            state: Normalized state array for each agent
                'On-hand inventory' (list): Product, WIPs, Materials; 
                    e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [10, 5, 20, 30, 40]
                'In-transition inventory' (list): Materials; 
                    e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): [3, 5, 2]
                'Remaining demand' (int): Product;
                    e.g. AP2 (1 Prodct, 1 WIP, 3 Materials): 10
        """

        # Make State for RL
        state = []
        for i in range(len(I)):
            # On-hand inventory levels for all items
            state.append(
                LIST_LOG_STATE_DICT[-1][f"On_Hand_{I[i]['NAME']}"])
            # In-transition inventory levels for material items
            if I[i]["TYPE"] == "Material":
                # append Intransition inventory
                state.append(
                    LIST_LOG_STATE_DICT[-1][f"In_Transit_{I[i]['NAME']}"])
        # Remaining demand
        rem_demand = I[0]["DEMAND_QUANTITY"] - \
            self.inventory_list[0].on_hand_inventory
        if rem_demand < 0:
            raise ValueError("Error: Negative remaining demand detected")
        else:
            state.append(rem_demand)

        # Normalize state
        state = np.array(state)
        norm_state = self._normalize_state(state)

        # Update logs (LIST_STATE_REAL, LIST_STATE_NOR)
        if LOG_STATE:
            LIST_STATE_REAL.append([i for i in state])
            LIST_STATE_NOR.append([i for i in norm_state])
        return state

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
