from config_SimPy import *
from log_SimPy import *
import environment as env
from visualization_SimPy import *
import time

# Start timing the computation
start_time = time.time()

# Define the scenario
scenario = {"DEMAND": DEMAND_SCENARIO, "LEADTIME": LEADTIME_SCENARIO}

# Create environment
simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events = env.create_env(
    I, P, LIST_DAILY_EVENTS)

env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, supplierList, daily_events, I, scenario)


if PRINT_DAILY_EVENTS:
    print(f"============= Initial Inventory Status =============")
    for inventory in inventoryList:
        print(
            f"{I[inventory.item_id]['NAME']} Inventory: {inventory.on_hand_inventory} units")

    print(f"============= SimPy Simulation Begins =============")

for x in range(SIM_TIME):
    print(f"\nDay {(simpy_env.now) // 24+1} Report:")
    simpy_env.run(until=simpy_env.now+24)  # Run the simulation for 24 hours

    # Print the simulation log every 24 hours (1 day)
    if PRINT_DAILY_EVENTS:
        for log in daily_events:
            print(log)
    if LOG_DAILY_EVENTS:
        daily_events.clear()

    env.update_daily_report(inventoryList)

    env.Cost.update_cost_log(inventoryList)
    # Print the daily cost
    if PRINT_DAILY_COST:
        for key in DAILY_COST.keys():
            print(f"{key}: {DAILY_COST[key]}")
        print(f"Daily Total Cost: {LOG_COST[-1]}")
    print(f"Cumulative Total Cost: {sum(LOG_COST)}")
    env.Cost.clear_cost()

if PRINT_GRAPH_RECORD:
    viz_sq()

if PRINT_LOG_REPORTS:
    print("\nLOG_DAILY_REPORTS ================")
    for report in LOG_DAILY_REPORTS:
        print(report)
    print("\nLOG_STATE_DICT ================")
    for record in LOG_STATE_DICT:
        print(record)

# Calculate computation time and print it
end_time = time.time()
print(f"\nComputation time (s): {(end_time - start_time):.2f} seconds")
print(f"Computation time (m): {(end_time - start_time)/60:.2f} minutes")
