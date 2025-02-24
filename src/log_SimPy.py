# Log simulation events
LOG_DAILY_EVENTS = []

# Stores the daily total cost incurred each day
LOG_COST = []

# Log daily repots: Inventory level for each item; Remaining demand (demand - product level)
LOG_DAILY_REPORTS = []
LOG_STATE_DICT = []

# Dictionary to temporarily store the costs incurred over a 24-hour period
DAILY_COST = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}

GRAPH_LOG = {}
