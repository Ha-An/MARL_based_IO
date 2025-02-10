import time
from GymWrapper import *
from GymEnvironment import *
from config_SimPy import *
from config_MARL import *

num_experiment = 5
data = []

for i in range(num_experiment):
    # Start timing the computation
    start_time = time.time()

    # Create environment
    env = InventoryManagementEnv()

    # Initialize wrapper
    wrapper = GymWrapper(
        env=env,
        n_agents=MAT_COUNT,
        action_dim=env.action_dim_size,  # 0-5 units order quantity
        state_dim=env.state_dim_size,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA
    )

    # Train new model
    print("Starting training of new model...")
    wrapper.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
    training_end_time = time.time()

    # Evaluate the trained model
    print("\nStarting evaluation...")
    avg_reward = wrapper.evaluate(N_EVAL_EPISODES)

    # Calculate computation time and print it
    end_time = time.time()

    data.append({"Avg. Reward": avg_reward,
                "total_time": (end_time - start_time)/60})

for i in range(num_experiment):
    print(f"Experiment {i+1}:")
    print(data[i])


# tensorboard --logdir=runs
