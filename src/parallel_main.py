# import time
# from parallel_env import ParallelInventoryManagementEnv
# # 기존에 사용하던 MAAC / GymWrapper 관련 모듈들 import
# from GymWrapper import GymWrapper
# from config_SimPy import *
# from config_MARL import *

# if __name__ == "__main__":
#     start_time = time.time()

#     # 원하는 병렬 환경 개수 설정
#     num_envs = NUM_PARALLEL_ENVS  # config_MARL.py 등에 설정된 값 사용

#     # 기존 InventoryManagementEnv 대신 병렬 환경 생성
#     parallel_env = ParallelInventoryManagementEnv(num_envs=num_envs)

#     # GymWrapper는 그대로 재사용 (단, 내부적으로 single env가 아닌 parallel_env를 사용)
#     wrapper = GymWrapper(
#         env=parallel_env,
#         num_agents=MAT_COUNT,
#         joint_action_space_size=JOINT_ACTION_SPACE_SIZE,
#         multi_state_space_size=MULTI_STATE_SPACE_SIZE,
#         buffer_size=BUFFER_SIZE,
#         batch_size=BATCH_SIZE,
#         lr_actor=LEARNING_RATE_ACTOR,
#         lr_critic=LEARNING_RATE_CRITIC,
#         gamma=GAMMA
#     )

#     # 모델 로드 여부에 따라 분기
#     if LOAD_MODEL:
#         print(f"Loading model from {MODEL_PATH}")
#         try:
#             wrapper.load_model(MODEL_PATH)
#             print("Model loaded successfully")

#             # 로드한 모델로 평가
#             training_end_time = time.time()
#             wrapper.evaluate(N_EVAL_EPISODES)
#         except FileNotFoundError:
#             print(f"No saved model found at {MODEL_PATH}")
#     else:
#         # 병렬 학습 예시 (GymWrapper에 train_parallel이 구현되어 있다고 가정)
#         print("Starting parallel training of new model...")
#         wrapper.train_parallel(N_TRAIN_EPISODES, EVAL_INTERVAL)

#         training_end_time = time.time()

#         # 학습 완료 후 평가
#         print("\nStarting evaluation...")
#         wrapper.evaluate(N_EVAL_EPISODES)

#     end_time = time.time()

#     print("\nTime Analysis:")
#     print(f"Total computation time: {(end_time - start_time)/60:.2f} minutes")
#     if not LOAD_MODEL:
#         print(
#             f"Training time: {(training_end_time - start_time)/60:.2f} minutes")
#     print(f"Evaluation time: {(end_time - training_end_time)/60:.2f} minutes")


from config_SimPy import *
from log_SimPy import *
import environment_SimPy as env
from visualization_SimPy import *
import time
import threading
# Start timing the computation
start_time = time.time()


def simulate():

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
        # Run the simulation for 24 hours
        simpy_env.run(until=simpy_env.now+24)

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
    # Log simulation events

    LIST_DAILY_EVENTS.clear()

    # Stores the daily total cost incurred each day
    LOG_COST.clear()

    # Log daily repots: Inventory level for each item; Remaining demand (demand - product level)
    LOG_DAILY_REPORTS.clear()
    LOG_STATE_DICT.clear()

    # Dictionary to temporarily store the costs incurred over a 24-hour period
    DAILY_COST = {
        'Holding cost': 0,
        'Process cost': 0,
        'Delivery cost': 0,
        'Order cost': 0,
        'Shortage cost': 0
    }
    print(f"\nComputation time (s): {(end_time - start_time):.2f} seconds")
    print(f"\nComputation time (m): {(end_time - start_time)/60:.2f} minutes")


# 스레드 실행
threads = []
thread1 = threading.Thread(target=simulate)
thread2 = threading.Thread(target=simulate)
thread3 = threading.Thread(target=simulate)
thread4 = threading.Thread(target=simulate)

thread1.start()
thread2.start()
thread3.start()
thread4.start()

threads = [thread1, thread2, thread3, thread4]
for thread in threads:
    thread.join()

# for _ in range(2):
#     simulate()

end_time = time.time()
print(f"\nComputation time (s): {(end_time - start_time):.2f} seconds")
