import gym
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

    Attributes: 
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

        # Record the cumulative value of each cost
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

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

        # Run simulation for one day
        LOG_ACTION[-1].append(actions)
        self.simpy_env.run(until=(self.current_day + 1) * 24)
        self.current_day += 1
        update_daily_report(self.inventory_list)

        # Get next observations
        next_states = self._get_observations()

        # Calculate reward (a negative value of the daily total cost)
        reward = -Cost.update_cost_log(self.inventory_list)

        ''' Reward scaling '''
        reward = reward / 1000

        # Update LOG_TOTAL_COST_COMP
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] += DAILY_COST[key]
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
        # Min-Max 정규화 공식 적용
        normalized_state = (state - STATE_MINS) / \
            (STATE_MAXS - STATE_MINS + 1e-8)

        # 0~1 범위를 벗어나는 경우 보정 (숫자 오차 방지)
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
                LOG_STATE_DICT[-1][f"On_Hand_{I[i]['NAME']}"])
            # In-transition inventory levels for material items
            if I[i]["TYPE"] == "Material":
                # append Intransition inventory
                state.append(
                    LOG_STATE_DICT[-1][f"In_Transit_{I[i]['NAME']}"])
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

        # Update logs (LOG_STATE_REAL, LOG_STATE_NOR)
        if LOG_STATE:
            LOG_STATE_REAL.append([i for i in state])
            LOG_STATE_NOR.append([i for i in norm_state])
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
