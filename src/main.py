import time
from GymWrapper import *
from GymEnvironment import *
from config_SimPy import *
from config_MARL import *

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

if LOAD_MODEL:
    # Load the saved model and evaluate
    print(f"Loading model from {MODEL_PATH}")
    try:
        wrapper.load_model(MODEL_PATH)
        print("Model loaded successfully")

        # Evaluate the loaded model
        training_end_time = time.time()
        wrapper.evaluate(N_EVAL_EPISODES)
    except FileNotFoundError:
        print(f"No saved model found at {MODEL_PATH}")
        exit()
else:
    # Train new model
    print("Starting training of new model...")
    wrapper.train(N_TRAIN_EPISODES, EVAL_INTERVAL)
    training_end_time = time.time()

    # Evaluate the trained model
    print("\nStarting evaluation...")
    wrapper.evaluate(N_EVAL_EPISODES)

# Calculate computation time and print it
end_time = time.time()
print("\nTime Analysis:")
print(f"Total computation time: {(end_time - start_time)/60:.2f} minutes")
if not LOAD_MODEL:
    print(f"Training time: {(training_end_time - start_time)/60:.2f} minutes")
print(f"Evaluation time: {(end_time - training_end_time)/60:.2f} minutes")


# tensorboard --logdir=runs
