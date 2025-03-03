import simpy
from config_SimPy import *
from log_SimPy import *
from visualization_SimPy import *


class Inventory:
    """
    Class for managing inventory for each item (material, WIP, or product) in the simulation.
    The class contains methods for updating inventory levels, calculating costs, and generating daily reports.

    Attributes:
        env (simpy.Environment): SimPy environment
        item_id (int): ID of the item
        on_hand_inventory (int): On-hand inventory level
        in_transition_inventory (int): In-transition inventory level
        capacity_limit (int): Maximum capacity of the inventory
        daily_inven_report (list): Daily inventory report
        unit_holding_cost (float): Unit holding cost per hour
        holding_cost_last_updated (float): Last time holding cost was updated
    """

    def __init__(self, env, item_id, holding_cost):
        """
        Args:
            env (simpy.Environment): SimPy environment
            item_id (int): ID of the item
            holding_cost (float): Unit holding cost per hour
        """

        # Initialize inventory object
        self.env = env
        self.item_id = item_id  # 0: product; others: WIP or material
        # Initial inventory level
        self.on_hand_inventory = I[self.item_id]['INIT_LEVEL']
        # Inventory in transition (e.g., being delivered)
        self.in_transition_inventory = 0
        self.capacity_limit = INVEN_LEVEL_MAX  # Maximum capacity of the inventory
        # Daily inventory report template:
        '''
        Day / Inventory_Name / Inventory_Type / Inventory at the start of the day / Income_Inventory(Onhand) / Outgoing_inventory(Onhand) / Intransit_Inventory / Inventory at the end of the day
        '''
        self.daily_inven_report = [f"Day {self.env.now // 24+1}", I[self.item_id]['NAME'],
                                   I[self.item_id]['TYPE'], self.on_hand_inventory, 0, 0, 0, 0]
        # Unit holding cost per hour
        self.unit_holding_cost = holding_cost / 24
        self.holding_cost_last_updated = 0.0  # Last time holding cost was updated

    def update_demand_quantity(self, daily_events):
        """
        Update the demand quantity and log the event.

        Args:
            daily_events (list): List to store daily events
        """
        if LOG_DAILY_EVENTS:
            daily_events.append(
                f"{present_daytime(self.env.now)}: Customer order of {I[0]['NAME']}                                 : {I[0]['DEMAND_QUANTITY']} units ")

    def update_inven_level(self, quantity_of_change, inven_type, daily_events):
        """
        Update the inventory level based on the quantity of change and log the event.

        Args:
            quantity_of_change (int): Quantity of change in inventory
            inven_type (str): Type of inventory (ON_HAND or IN_TRANSIT)
            daily_events (list): List to store daily events
        """
        if inven_type == "ON_HAND":
            # Update on-hand inventory
            Cost.cal_cost(self, "Holding cost")
            self.on_hand_inventory += quantity_of_change

            # Check if inventory exceeds capacity limit
            if self.on_hand_inventory > self.capacity_limit:
                if LOG_DAILY_EVENTS:
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: Due to the upper limit of the inventory, {I[self.item_id]['NAME']} is wasted: {self.on_hand_inventory - self.capacity_limit}")
                self.on_hand_inventory = self.capacity_limit

            self.holding_cost_last_updated = self.env.now

        elif inven_type == "IN_TRANSIT":
            # Update in-transition inventory
            self.in_transition_inventory += quantity_of_change
        self._update_report(quantity_of_change, inven_type)

    def _update_report(self, quantity_of_change, inven_type):
        """
        [Protected]
        Update the daily inventory report based on the quantity of change.

        Args:
            quantity_of_change (int): Quantity of change in inventory
            inven_type (str): Type of inventory (ON_HAND or IN_TRANSIT)            
        """
        if inven_type == "ON_HAND":
            if quantity_of_change > 0:
                # Income Inventory
                self.daily_inven_report[4] += quantity_of_change

            else:
                # Outgoing Invnetory
                self.daily_inven_report[5] -= quantity_of_change

        elif inven_type == "IN_TRANSIT":
            # In_Transit Inventory
            self.daily_inven_report[6] += quantity_of_change


class Supplier:
    """
    Class for managing suppliers in the simulation.
    The class contains methods for delivering materials to the manufacturer after a certain lead time.

    Attributes:
        env (simpy.Environment): SimPy environment
        name (str): Name of the supplier
        item_id (int): ID of the item
    """

    def __init__(self, env, name, item_id):
        """
        Attributes = Args
        """
        # Initialize supplier object
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver_to_manufacturer(self, procurement, material_qty, material_inventory, daily_events, lead_time_dict):
        """
        Deliver materials to the manufacturer after a certain lead time.

        Args:
            procurement (Procurement): Procurement object
            material_qty (int): Quantity of materials to deliver
            material_inventory (Inventory): Material inventory
            daily_events (list): List to store daily events
            lead_time_dict (dict): Dictionary containing lead times for different suppliers
        """
        I[self.item_id]["SUP_LEAD_TIME"] = SUP_LEAD_TIME_FUNC(lead_time_dict)
        lead_time = I[self.item_id]["SUP_LEAD_TIME"]
        # Log the delivery event with lead time
        if LOG_DAILY_EVENTS:
            daily_events.append(
                f"{self.env.now}: {I[self.item_id]['NAME']} will be delivered at {lead_time} days after         : {material_qty} units")

        # Wait for the lead time
        yield self.env.timeout(lead_time * 24)

        # Receive materials by calling the receive_materials method of procurement
        procurement.receive_materials(
            material_qty, material_inventory, daily_events)


class Procurement:
    """
    Class for managing procurement in the simulation.
    The class contains methods for ordering materials and processing the receipt of materials.

    Attributes:
        env (simpy.Environment): SimPy environment
        item_id (int): ID of the item
        unit_purchase_cost (float): Unit purchase cost
        unit_setup_cost (float): Unit setup cost
        order_qty (int): Order quantity
    """

    def __init__(self, env, item_id, purchase_cost, setup_cost):
        """
        Attributes = Args
        """
        self.env = env
        self.item_id = item_id
        self.unit_purchase_cost = purchase_cost
        self.unit_setup_cost = setup_cost
        self.order_qty = 0  # Initialize order quantity

    def receive_materials(self, material_qty, material_inventory, daily_events):
        """
        Process the receipt of materials and update inventory.

        Args:
            material_qty (int): Quantity of materials received
            material_inventory (Inventory): Material inventory
            daily_events (list): List to store daily events
        """
        if LOG_DAILY_EVENTS:
            daily_events.append(
                f"==============={I[self.item_id]['NAME']} Delivered ===============")

        # Update in-transition inventory (reduce it)
        material_inventory.update_inven_level(
            -material_qty, "IN_TRANSIT", daily_events)
        # Update on-hand inventory (increase it)
        material_inventory.update_inven_level(
            material_qty, "ON_HAND", daily_events)
        if LOG_DAILY_EVENTS:
            daily_events.append(
                f"{present_daytime(self.env.now)}: {I[self.item_id]['NAME']} has delivered                             : {material_qty} units ")  # Record when Material provide

    def order_material(self, supplier, inventory, daily_events, lead_time_dict):
        """
        Place orders for materials to the supplier.

        Args:
            supplier (Supplier): Supplier object
            inventory (Inventory): Material inventory
            daily_events (list): List to store daily events
            lead_time_dict (dict): Dictionary containing lead times for different suppliers
        """
        while True:
            if LOG_DAILY_EVENTS:
                daily_events.append(
                    f"==============={I[self.item_id]['NAME']}\'s Inventory ===============")

            # Set the order size based on LOT_SIZE_ORDER and reorder level
            if USE_SQPOLICY:
                if inventory.on_hand_inventory <= SQPAIR['Reorder']:
                    order_size = SQPAIR['Order']
                else:
                    order_size = 0
            else:
                # I[self.item_id]["LOT_SIZE_ORDER"] = ORDER_QTY
                order_size = I[self.item_id]["LOT_SIZE_ORDER"]
            if order_size > 0:
                if LOG_DAILY_EVENTS:
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: The Procurement ordered {I[self.item_id]['NAME']}: {order_size}  units  ")
                self.order_qty = order_size
                # Update in-transition inventory
                inventory.update_inven_level(
                    order_size, "IN_TRANSIT", daily_events)
                # Calculate and log order cost
                Cost.cal_cost(self, "Order cost")
                # Initiate the delivery process by calling deliver_to_manufacturer method of the supplier
                self.env.process(supplier.deliver_to_manufacturer(
                    self, order_size, inventory, daily_events, lead_time_dict))
                if LOG_DAILY_EVENTS:
                    # Record in_transition_inventory
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: {I[self.item_id]['NAME']}\'s In_transition_inventory                    : {inventory.in_transition_inventory} units ")
                    # Record inventory
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: {I[self.item_id]['NAME']}\'s Total_Inventory                            : {inventory.in_transition_inventory+inventory.on_hand_inventory} units  ")
            # Wait for the next order cycle
            yield self.env.timeout(I[self.item_id]["MANU_ORDER_CYCLE"] * 24)


class Production:
    """
    Class for managing production processes in the simulation.
    The class contains methods for simulating the production process and updating inventory levels.

    Attributes:
        env (simpy.Environment): SimPy environment
        name (str): Name of the production process
        process_id (int): ID of the process
        production_rate (int): Production rate
        output (dict): Output item
        input_inventories (list): List of input inventories
        qnty_for_input_item (list): List of quantities for input items
        output_inventory (Inventory): Output inventory
        processing_time (float): Processing time for each unit
        unit_processing_cost (float): Unit processing cost per hour
        print_stop (bool): Flag to print stop event
        print_limit (bool): Flag to print limit event
    """

    def __init__(self, env, name, process_id, production_rate, output, input_inventories, qnty_for_input_item, output_inventory, processing_cost, process_stop_cost):
        """
        Attributes = Args
        """
        # Initialize production process
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.qnty_for_input_item = qnty_for_input_item
        self.output_inventory = output_inventory
        self.processing_time = 24 / self.production_rate
        self.unit_processing_cost = processing_cost
        self.print_stop = True
        self.print_limit = True

    def process_items(self, daily_events):
        """
        Simulate the production process.

        Args:
            daily_events (list): List to store daily events
        """
        while True:
            # Check if there's a shortage of input materials or WIPs
            shortage_check = False
            for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                if inven.on_hand_inventory < input_qnty:
                    shortage_check = True
                    # early stop
                    break

            # Check if the output inventory is full
            inven_upper_limit_check = False
            if self.output_inventory.on_hand_inventory >= self.output_inventory.capacity_limit:
                inven_upper_limit_check = True

            if shortage_check:
                if self.print_stop:
                    if LOG_DAILY_EVENTS:
                        daily_events.append(
                            "===============Process Phase===============")

                        daily_events.append(
                            f"{present_daytime(self.env.now)}: Stop {self.name} due to a shortage of input materials or WIPs")
                self.print_stop = False

                # yield self.env.timeout(1)  # Check upper limit every hour
                yield self.env.timeout(self.processing_time)
            elif inven_upper_limit_check:
                if self.print_limit:
                    if LOG_DAILY_EVENTS:
                        daily_events.append(
                            "===============Process Phase===============")
                        daily_events.append(
                            f"{present_daytime(self.env.now)}: Stop {self.name} due to the upper limit of the inventory. The output inventory is full")
                self.print_limit = False
                # yield self.env.timeout(1)  # Check upper limit every hour
                yield self.env.timeout(self.processing_time)
            else:
                if LOG_DAILY_EVENTS:
                    daily_events.append(
                        "===============Process Phase===============")
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: Process {self.process_id} begins")

                # Consume input materials or WIPs
                for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                    inven.update_inven_level(-input_qnty,
                                             "ON_HAND", daily_events)
                # Process items (consume time)
                Cost.cal_cost(self, "Process cost")
                # Time correction
                yield self.env.timeout(self.processing_time-TIME_CORRECTION)
                if LOG_DAILY_EVENTS:
                    daily_events.append(
                        "===============Result Phase================")
                # Cost Update Time Correction
                self.output_inventory.holding_cost_last_updated -= TIME_CORRECTION
                # Update the inventory level for the output item
                self.output_inventory.update_inven_level(
                    1, "ON_HAND", daily_events)
                # Cost Update Time Correction
                self.output_inventory.holding_cost_last_updated += TIME_CORRECTION
                if LOG_DAILY_EVENTS:
                    daily_events.append(
                        f"{self.env.now+TIME_CORRECTION}: {self.output['NAME']} has been produced                         : 1 units")
                self.print_limit = True
                self.print_limit = True
                yield self.env.timeout(TIME_CORRECTION)  # Time correction


class Sales:
    """
    Class for managing sales processes in the simulation.
    The class contains methods for delivering products to customers and handling shortages.

    Attributes:
        env (simpy.Environment): SimPy environment
        item_id (int): ID of the item
        due_date (int): Due date for the product
        unit_delivery_cost (float): Unit delivery cost
        unit_setup_cost (float): Unit setup cost
        unit_shortage_cost (float): Unit shortage cost
        delivery_item (int): Delivery item
        num_shortages (int): Number of shortages
    """

    def __init__(self, env, item_id, delivery_cost, setup_cost, shortage, due_date):
        """
        Attributes = Args
        """
        # Initialize sales process
        self.env = env
        self.item_id = item_id
        self.due_date = due_date
        self.unit_delivery_cost = delivery_cost
        self.unit_setup_cost = setup_cost
        self.unit_shortage_cost = shortage
        self.delivery_item = 0
        self.num_shortages = 0

    def _deliver_to_cust(self, demand_size, product_inventory, daily_events):
        """
        [Protected]
        Deliver products to customers and handle shortages if any.

        Args:
            demand_size (int): Demand size
            product_inventory (Inventory): Product inventory
            daily_events (list): List to store daily events
        """
        yield self.env.timeout(I[self.item_id]["DUE_DATE"] * 24-TIME_CORRECTION/2)  # Time Correction
        product_inventory.holding_cost_last_updated -= TIME_CORRECTION / \
            2  # Cost Update Time Correction
        # Calculate the shortage
        self.num_shortages = max(
            0, demand_size - product_inventory.on_hand_inventory)
        # Calculate the delivery item
        self.delivery_item = min(
            demand_size, product_inventory.on_hand_inventory)
        if LOG_DAILY_EVENTS:
            daily_events.append(
                f"DEMAND: {demand_size}, DELIVERY:{self.delivery_item}")
        # If there are some products available, deliver them first
        if self.delivery_item > 0:
            if LOG_DAILY_EVENTS:
                daily_events.append(
                    f"{self.env.now+TIME_CORRECTION/2}: PRODUCT have been delivered to the customer       : {self.delivery_item} units ")
            # Update inventory
            product_inventory.update_inven_level(
                -self.delivery_item, 'ON_HAND', daily_events)
            Cost.cal_cost(self, "Delivery cost")

        # Calculate and log shortage cost
        if self.num_shortages > 0:
            Cost.cal_cost(self, "Shortage cost")
            if LOG_DAILY_EVENTS:
                daily_events.append(
                    f"{present_daytime(self.env.now+TIME_CORRECTION/2)}: Unable to deliver {self.num_shortages} units to the customer due to product shortage")

        # Cost Update Time Correction
        product_inventory.holding_cost_last_updated += TIME_CORRECTION
        yield self.env.timeout(TIME_CORRECTION/2)  # Time Correction

    def receive_demands(self, demand_qty, product_inventory, daily_events):
        """
        Receive demands from customers and initiate the delivery process.

        Args:
            demand_qty (int): Demand quantity
            product_inventory (Inventory): Product inventory
            daily_events (list): List to store daily events
        """
        # Update demand quantity in inventory
        product_inventory.update_demand_quantity(daily_events)
        # Initiate delivery process
        self.env.process(self._deliver_to_cust(
            demand_qty, product_inventory, daily_events))


class Customer:
    """
    Class for managing customers in the simulation.
    The class contains methods for placing orders for products and initiating the sales process.

    Attributes:
        env (simpy.Environment): SimPy environment
        name (str): Name of the customer
        item_id (int): ID of the item
    """

    def __init__(self, env, name, item_id):
        """
        Attributes = Args
        """
        # Initialize customer object
        self.env = env
        self.name = name
        self.item_id = item_id

    def order_product(self, sales, product_inventory, daily_events, scenario):
        """
        Place orders for products to the sales process.

        Args:
            sales (Sales): Sales object
            product_inventory (Inventory): Product inventory
            daily_events (list): List to store daily events
            scenario (dict): Scenario for demand quantity
        """
        while True:
            # Generate a random demand quantity
            I[0]["DEMAND_QUANTITY"] = DEMAND_QTY_FUNC(scenario)
            # Receive demands and initiate delivery process
            sales.receive_demands(
                I[0]["DEMAND_QUANTITY"], product_inventory, daily_events)
            # Wait for the next order cycle
            yield self.env.timeout(I[0]["CUST_ORDER_CYCLE"] * 24)


class Cost:
    """
    Class for managing costs in the simulation.
    The class contains methods for calculating and logging different types of costs.
    """
    # Class for managing costs in the simulation
    def cal_cost(instance, cost_type):
        """
        Calculate and log different types of costs.

        Args:
            instance: Instance of the class (Inventory, Procurement, Production, Sales)
            cost_type (str): Type of cost (Holding cost, Process cost, Delivery cost, Order cost, Shortage cost)
        """

        if cost_type == "Holding cost":
            # Calculate holding cost
            DICT_DAILY_COST[cost_type] += instance.unit_holding_cost * instance.on_hand_inventory * (
                instance.env.now - instance.holding_cost_last_updated)
        elif cost_type == "Process cost":
            # Calculate processing cost
            DICT_DAILY_COST[cost_type] += instance.unit_processing_cost
        elif cost_type == "Delivery cost":
            # Calculate delivery cost
            DICT_DAILY_COST[cost_type] += instance.unit_delivery_cost * \
                instance.delivery_item + instance.unit_setup_cost
        elif cost_type == "Order cost":
            # Calculate order cost
            DICT_DAILY_COST[cost_type] += instance.unit_purchase_cost * \
                instance.order_qty + instance.unit_setup_cost
        elif cost_type == "Shortage cost":
            # Calculate shortage cost
            DICT_DAILY_COST[cost_type] += instance.unit_shortage_cost * \
                instance.num_shortages

    def update_cost_log(inventoryList):
        """
        Update the cost log at the end of each day.

        Args:
            inventoryList (list): List of inventory objects

        Returns:
            float: Daily total cost
        """
        LIST_LOG_COST.append(0)
        # Update holding cost
        for inven in inventoryList:
            DICT_DAILY_COST['Holding cost'] += inven.unit_holding_cost * inven.on_hand_inventory * (
                inven.env.now - inven.holding_cost_last_updated)
            inven.holding_cost_last_updated = inven.env.now

        # Update daily total cost
        for key in DICT_DAILY_COST.keys():
            LIST_LOG_COST[-1] += DICT_DAILY_COST[key]

        return LIST_LOG_COST[-1]

    def clear_cost():
        """
        Clear the daily cost report.
        """
        # Clear daily report
        for key in DICT_DAILY_COST.keys():
            DICT_DAILY_COST[key] = 0


def record_inventory(env, inventoryList):
    """
    Record inventory at every hour

    Args:
        env (simpy.Environment): SimPy environment
        inventoryList (list): List of inventory objects
    """
    record_graph(I)
    while (True):
        for inven in inventoryList:
            GRAPH_LOG[I[inven.item_id]['NAME']].append(inven.on_hand_inventory)
            if I[inven.item_id]['TYPE'] == 'Material':
                GRAPH_LOG[f"{I[inven.item_id]['NAME']}_in_transition_inventory"].append(
                    inven.in_transition_inventory)
        yield env.timeout(1)


def create_env(I, P, daily_events):
    """
    Create the simulation environment and necessary objects.

    Args:
        I (dict): Dictionary containing information about items
        P (dict): Dictionary containing information about processes
        daily_events (list): List to store daily events

    Returns:
        simpy_env(simpy.Environment): SimPy environment object
        inventoryList (list): List of inventory objects
        procurementList (list): List of procurement objects
        productionList (list): List of production objects
        sales (Sales): Sales object
        customer (Customer): Customer object
        supplierList (list): List of supplier objects
        daily_events (list): List to store daily events 
    """
    # Function to create the simulation environment and necessary objects
    simpy_env = simpy.Environment()  # Create a SimPy environment

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[i]["HOLD_COST"]))

    # Create stakeholders (Customer, Suppliers)
    customer = Customer(simpy_env, "CUSTOMER", I[0]["ID"])

    supplierList = []
    procurementList = []
    for i in I.keys():
        if I[i]["TYPE"] == 'Material':
            supplierList.append(Supplier(simpy_env, "SUPPLIER_" + str(i), i))
            procurementList.append(Procurement(
                simpy_env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["ORDER_COST_TO_SUP"]))

    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"], I[0]["SHORTAGE_COST_PRO"], I[0]["DUE_DATE"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_TYPE_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_" + str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, P[i]["QNTY_FOR_INPUT_ITEM"], output_inventory, P[i]["PROCESS_COST"], P[i]["PROCESS_STOP_COST"]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events


def simpy_event_processes(simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events, I, scenario):
    """
    Define the event processes for the simulation.

    Args:
        simpy_env (simpy.Environment): SimPy environment
        inventoryList (list): List of inventory objects
        procurementList (list): List of procurement objects
        productionList (list): List of production objects
        sales (Sales): Sales object
        customer (Customer): Customer object
        supplierList (list): List of supplier objects
        daily_events (list): List to store daily events
        I (dict): Dictionary containing information about items
        scenario (dict): Dictionary containing scenario information    
    """

    simpy_env.process(customer.order_product(
        sales, inventoryList[I[0]["ID"]], daily_events, scenario["DEMAND"]))
    for production in productionList:
        simpy_env.process(production.process_items(daily_events))
    for i in range(len(supplierList)):
        simpy_env.process(procurementList[i].order_material(
            supplierList[i], inventoryList[supplierList[i].item_id], daily_events, scenario["LEADTIME"]))
    simpy_env.process(record_inventory(simpy_env, inventoryList))


def update_daily_report(inventoryList):
    """
    Update daily reports for inventory.

    Args:
        inventoryList (list): List of inventory objects 
    """
    # Update daily reports for inventory
    day_list = []
    day_dict = {}
    for inven in inventoryList:
        inven.daily_inven_report[-1] = inven.on_hand_inventory
        day_list = day_list+(inven.daily_inven_report)

        day_dict[f"On_Hand_{I[inven.item_id]['NAME']}"] = inven.on_hand_inventory
        if I[inven.item_id]["TYPE"] == "Material":
            day_dict[f"In_Transit_{I[inven.item_id]['NAME']}"] = inven.daily_inven_report[6]
    LIST_LOG_DAILY_REPORTS.append(day_list)
    LIST_LOG_STATE_DICT.append(day_dict)
    # Reset report
    for inven in inventoryList:
        inven.daily_inven_report = [f"Day {inven.env.now//24+1}", I[inven.item_id]['NAME'], I[inven.item_id]['TYPE'],
                                    inven.on_hand_inventory, 0, 0, inven.in_transition_inventory, 0]  # inventory report


def present_daytime(env_now):
    """
    Present the current time in the simulation.

    Args:
        env_now (float): Current time in the simulation

    Returns:
        str: Current time in the simulation
    """
    fill_length = len(str(SIM_TIME * 24))
    return str(int(env_now)).zfill(fill_length)
