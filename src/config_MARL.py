import os
from config_SimPy import *
from gym import spaces

# The number of parallel environments
# NUM_PARALLEL_ENVS = 4

# The number of agents
NUM_AGENTS = MAT_COUNT

########## The range of action space ##########
'''
Define action space: Action space is a MultiDiscrete space where each agent can choose an order quantity 
    e.g. action space for each agent = [0, 1, 2, 3, 4, 5]
        action_space_size for each agent = 5 - 0 + 1 = 6
        joint_action_space_size = [6, 6, 6, 6] (4 agents)
'''
ACTION_MIN = 0
ACTION_MAX = 5  # ACTION_SPACE = [0, 1, 2, 3, 4, 5]
ACTION_SPACE_SIZE = ACTION_MAX - ACTION_MIN + 1
JOINT_ACTION_SPACE_SIZE = spaces.MultiDiscrete([ACTION_SPACE_SIZE]*NUM_AGENTS)


########## The range of state space ##########
'''
State space is a MultiDiscrete space (fully observable) where each agent can observe:
    On-hand inventory level for each item
        max: INVEN_LEVEL_MAX
        min: INVEN_LEVEL_MIN
        Number of dimensions (number of items): len(I)
        Size of each dimension: INVEN_LEVEL_MAX - INVEN_LEVEL_MIN + 1
    In-transition inventory level for each material
        max: ACTION_MAX
        min: 0
        Number of dimensions (number of materials): MAT_COUNT
        Size of each dimension: ACTION_MAX - ACTION_MIN + 1
    Remaining demand (demand - on-hand inventory) for the first item (Product) -> negative values are clipped to 0
        max: DEMAND_SCENARIO["max"]
        min: 0
        Size of dimension: DEMAND_SCENARIO["max"] - DEMAND_SCENARIO["min"] + 1

    e.g. AP1: state_dims = [21, 21, 6, 6] 
'''
STATE_MINS = []
STATE_MAXS = []
# On-hand inventory levels for all items
for _ in range(len(I)):
    STATE_MINS.append(INVEN_LEVEL_MIN)
    STATE_MAXS.append(INVEN_LEVEL_MAX)
# In-transition inventory levels for material items
for _ in range(MAT_COUNT):
    STATE_MINS.append(0)
    STATE_MAXS.append(ACTION_MAX*7)
# Remaining demand
STATE_MINS.append(0)
STATE_MAXS.append(DEMAND_SCENARIO["max"])
# Convert to numpy arrays
STATE_MINS = np.array(STATE_MINS, dtype=np.int32)
STATE_MAXS = np.array(STATE_MAXS, dtype=np.int32)
# Define state space
MULTI_STATE_SPACE_SIZE = spaces.MultiDiscrete(STATE_MAXS - STATE_MINS + 1)

# Log daily repots
LOG_STATE = False

BUFFER_SIZE = 100000  # Size of the replay buffer (default: 100000)
# Batch size for training (Default: 256; Sample unit: transition)
BATCH_SIZE = 256  # Default: 256
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-5
GAMMA = 0.95

# Soft update parameter for the target network
TAU = 0.01

# Number of attention heads for the actor network (default: 4)
NUM_HEADS = 4
# Hidden dimension for the actor and critic networks (default: 64)
HIDDEN_DIM = 128

# Epsilon-greedy exploration: Exponential Decay
EPSILON_DECAY_TYPE = "linear"  # "exponential", "linear"
EPSILON_START = 1.0
EPSILON_END = 0.1
DECAY_RATE = 0.997  # 감소율 (0.9 ~ 0.999 사이 값 사용)

# Training
'''
N_TRAIN_EPISODES: Number of training episodes (Default=1000)
EVAL_INTERVAL: Interval for evaluation and printing results (Default=10)
'''
N_TRAIN_EPISODES = 10
EVAL_INTERVAL = 10

# Evaluation
'''
N_EVAL_EPISODES: Number of evaluation episodes (Default=100) 
'''
N_EVAL_EPISODES = 10


# Find minimum Delta
PRODUCT_OUTGOING_CORRECTION = 0
for key in P:
    PRODUCT_OUTGOING_CORRECTION = max(P[key]["PRODUCTION_RATE"] *
                                      max(P[key]['QNTY_FOR_INPUT_ITEM']), INVEN_LEVEL_MAX)
# maximum production


# Configuration for model loading/saving
LOAD_MODEL = False  # Set to True to load a saved model
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "maac_best_model.pt")  # 불러올 모델 경로
