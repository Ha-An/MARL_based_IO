# Log simulation events
LIST_DAILY_EVENTS = []

# Stores the daily total cost incurred each day
LIST_LOG_COST = []

# Log daily repots: Inventory level for each item; Remaining demand (demand - product level)
LIST_LOG_DAILY_REPORTS = []
LIST_LOG_STATE_DICT = []

# Dictionary to temporarily store the costs incurred over a 24-hour period
DICT_DAILY_COST = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}

GRAPH_LOG = {}
