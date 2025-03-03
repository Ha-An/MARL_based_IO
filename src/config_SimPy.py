import random  # For random number generation
import os
import numpy as np
import shutil

#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Material, WIP;
# NAME: Item's name or model;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to suppliers [days]
# INIT_LEVEL: Initial inventory level [units]
# DEMAND_QUANTITY: Demand quantity for the final product [units] -> THIS IS UPDATED EVERY 24 HOURS (Default: 0)
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# REMOVE## LOT_SIZE_ORDER: Lot-size for the order of materials (Q) [units] -> THIS IS AN AGENT ACTION THAT IS UPDATED EVERY 24 HOURS
# HOLD_COST: Holding cost of the items [$/unit*day]
# PURCHASE_COST: Purchase cost of the materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# ORDER_COST_TO_SUP: Ordering cost for the materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]
# DUE_DATE: Term of customer order to delivered [days]
# SHORTAGE_COST: Backorder cost of products [$/unit]

#### Processes #####################################################################
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day] (Production rate must be a positive number between 1 and 24.)
# INPUT_TYPE_LIST: List of types of input materials or WIPs
# QNTY_FOR_INPUT_ITEM: Quantity of input materials or WIPs [units]
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/unit]
# PROCESS_STOP_COST: Penalty cost for stopping the process [$/unit]


# Assembly Process 1

I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 7,
         "INIT_LEVEL": 0,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 7,
         "SHORTAGE_COST_PRO": 50},
     1: {"ID": 1, "TYPE": "Material", "NAME": "MATERIAL 1",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0}}
P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [
    1], "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}

'''
# Assembly Process 2
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PROD",
         "CUST_ORDER_CYCLE": 7,
         "INIT_LEVEL": 0,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 7,
         "SHORTAGE_COST_PRO": 50},
     1: {"ID": 1, "TYPE": "Material", "NAME": "MAT 1",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     2: {"ID": 2, "TYPE": "Material", "NAME": "MAT 2",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     3: {"ID": 3, "TYPE": "Material", "NAME": "MAT 3",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     4: {"ID": 4, "TYPE": "WIP", "NAME": "WIP 1",
         "INIT_LEVEL": 1,
         "HOLD_COST": 1}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1], I[2]], "QNTY_FOR_INPUT_ITEM": [1, 1],
         "OUTPUT": I[4], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[2], I[3], I[4]], "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
         "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}


# Assembly Process 3
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PROD",
         "CUST_ORDER_CYCLE": 7,
         "INIT_LEVEL": 0,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 7,
         "SHORTAGE_COST_PRO": 50},
     1: {"ID": 1, "TYPE": "Material", "NAME": "MAT 1",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     2: {"ID": 2, "TYPE": "Material", "NAME": "MAT 2",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     3: {"ID": 3, "TYPE": "Material", "NAME": "MAT 3",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     4: {"ID": 4, "TYPE": "Material", "NAME": "MAT 4",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     5: {"ID": 5, "TYPE": "Material", "NAME": "MAT 5",
         "MANU_ORDER_CYCLE": 1,
         "INIT_LEVEL": 2,
         "SUP_LEAD_TIME": 2,  # SUP_LEAD_TIME must be an integer
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "ORDER_COST_TO_SUP": 1,
         "LOT_SIZE_ORDER": 0},
     6: {"ID": 6, "TYPE": "WIP", "NAME": "WIP 1",
         "INIT_LEVEL": 1,
         "HOLD_COST": 1},
     7: {"ID": 7, "TYPE": "WIP", "NAME": "WIP 2",
         "INIT_LEVEL": 1,
         "HOLD_COST": 1}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1], I[2]], "QNTY_FOR_INPUT_ITEM": [1, 1],
         "OUTPUT": I[6], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[2], I[3], I[6]], "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
         "OUTPUT": I[7], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2},
     2: {"ID": 2, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[4], I[5], I[7]], "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
         "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}
'''

# State space
# if this is not 0, the length of state space of demand quantity is not identical to INVEN_LEVEL_MAX
INVEN_LEVEL_MIN = 0
INVEN_LEVEL_MAX = 20  # Capacity limit of the inventory [units]

# Simulation
SIM_TIME = 100  # 200 [days] per episode

# Count for material inventory
MAT_COUNT = 0
for id in I.keys():
    if I[id]["TYPE"] == "Material":
        MAT_COUNT += 1

# Scenario about Demand and leadtime
DEMAND_SCENARIO = {"Dist_Type": "UNIFORM",
                   "min": 14,
                   "max": 14}

LEADTIME_SCENARIO = {"Dist_Type": "UNIFORM",
                     "min": 1,
                     "max": 1}
# Example of Gaussian case
"""
DEMAND_SCENARIO = {"Dist_Type": "GAUSSIAN",
                    "mean": 11.5, 
                    "std": 2}
 
LEADTIME_SCENARIO = {"Dist_Type": "GAUSSIAN",
                     "mean": 3,
                     "std": 1}
"""


def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    else:
        folder_name = os.path.join(folder_name, "Train_1")
    os.makedirs(folder_name)
    return folder_name


def save_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    # Create a new folder
    os.makedirs(path)
    return path


def DEMAND_QTY_FUNC(scenario):
    # Uniform distribution
    if scenario["Dist_Type"] == "UNIFORM":
        return random.randint(scenario['min'], scenario["max"])
    # Gaussian distribution
    elif scenario["Dist_Type"] == "GAUSSIAN":
        # Gaussian distribution
        demand = round(np.random.normal(scenario['mean'], scenario['std']))
        if demand < 0:
            return 1
        elif demand > INVEN_LEVEL_MAX:
            return INVEN_LEVEL_MAX
        else:
            return demand


def SUP_LEAD_TIME_FUNC(lead_time_dict):
    if lead_time_dict["Dist_Type"] == "UNIFORM":
        # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
        # SUP_LEAD_TIME must be an integer
        return random.randint(lead_time_dict['min'], lead_time_dict['max'])
    elif lead_time_dict["Dist_Type"] == "GAUSSIAN":
        mean = lead_time_dict['mean']
        std = lead_time_dict['std']
        # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
        lead_time = np.random.normal(mean, std)
        if lead_time < 0:
            lead_time = 0
        elif lead_time > 7:
            lead_time = 7
        # SUP_LEAD_TIME must be an integer
        return int(round(lead_time))


# Ordering rules : Reorder point (S) and Order quantity (Q)
# USE_SQPOLICY = True  : When using SQpolicy (DRL is NOT used)
# USE_SQPOLICY = False  : When NOT using SQpolicy (DRL is used)
USE_SQPOLICY = False
# SQPAIR = {'Reorder': 5,
#           'Order': 4}
SQPAIR = {'Reorder': 2,
          'Order': 1}

# Log simulation events
LOG_DAILY_EVENTS = False

# Print logs
PRINT_GRAPH_RECORD = False
PRINT_DAILY_EVENTS = False
PRINT_DAILY_COST = False
PRINT_LOG_REPORTS = False

# Cost model
# If False, the total cost is calculated based on the inventory level for every 24 hours.
# Otherwise, the total cost is accumulated every hour.
HOURLY_COST_MODEL = True
'''
VISUALIAZTION = [1, 1, 1]  # PRINT RAW_MATERIAL, WIP, PRODUCT
'''
TIME_CORRECTION = 0.0001
