import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, exp
import os
from functools import lru_cache
import time

projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Compatibility fix


# Customer class
class Customer:
    def __init__(self, id: int, x: int, y: int, demand: int, ready_time: int, due_date: int, service_time: int):
        """
        Initialize a Customer object with the given attributes for the VRPTW problem.
        :param id: int unique identifier for the customer
        :param x: int x-coordinate of the customer location
        :param y: int y-coordinate of the customer location
        :param demand: int demand of the customer
        :param ready_time: int earliest time the customer can be served
        :param due_date: int latest time the customer can be served
        :param service_time: int time it takes to serve the customer
        """
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time


def validate_time_window(self, customers: list[Customer]):
    """
    Validate the time window constraints for the truck's route.
    :param customers: List of Customer objects.
    :return: True if the route is valid, otherwise False.
    """
    current_time = 0
    for i in range(len(self.route) - 1):
        from_customer = customers[self.route[i]]
        to_customer = customers[self.route[i + 1]]

        # Travel time
        travel_time = calculate_distance(from_customer, to_customer)
        arrival_time = current_time + travel_time

        # Check time window constraint
        if arrival_time < to_customer.ready_time:
            # Truck arrives early, wait until ready_time
            current_time = to_customer.ready_time
        elif arrival_time > to_customer.due_date:
            # Time window violated
            return False
        else:
            # Valid arrival time
            current_time = arrival_time

        # Add service time
        current_time += to_customer.service_time

    return True

# Truck class
class Truck:
    def __init__(self, id: int, capacity: int, start_location: int):
        """
        Initialize a Truck object with the given attributes for the VRPTW problem.
        :param id: int truck unique identifier
        :param capacity: int truck capacity
        :param start_location: int starting location of the truck (ID of customer 0 -> depot)
        """
        self.id = id
        self.capacity = capacity
        self.route = [start_location]
        self.remaining_capacity = capacity
        self.current_time = 0
        self.total_distance = 0

    def calculate_route_distance(self, customers: list[Customer]):
        """
        Calculate the total distance of the route for the truck
        :param customers: list of Customer objects
        :return: int total distance of the route for the truck
        """
        distance = 0
        for i in range(len(self.route) - 1):
            distance += calculate_distance(customers[self.route[i]], customers[self.route[i + 1]])
        return distance


@lru_cache(maxsize=None)
def calculate_distance(c1: Customer, c2: Customer):
    """
    Calculate the Euclidean distance between two customers
    :param c1: customer 1 object
    :param c2: customer 2 object
    :return: float Euclidean distance between the two customers
    """
    return sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

# Solution Class
class VRPTWSolution:
    def __init__(self, customers: list[Customer], truck_capacity: int, max_trucks: int, depot: Customer):
        """
        Initialize a VRPTWSolution object with the given attributes for the VRPTW problem.
        :param customers: list[Customer] list of Customer objects
        :param truck_capacity: int capacity of the trucks
        :param max_trucks: int maximum number of trucks that can be used
        :param depot: int ID of the depot location (customer 0)
        """
        self.customers = customers
        self.truck_capacity = truck_capacity
        self.max_trucks = max_trucks
        self.depot = depot
        self.routes = self.generate_initial_solution()

    def generate_initial_solution(self):
        """
        Generate an initial solution for the VRPTW problem using a greedy heuristic
        :return: list of Truck objects
        """
        unvisited = set(self.customers[1:])
        trucks = []
        for truck_id in range(self.max_trucks):
            truck = Truck(truck_id, self.truck_capacity, self.depot.id)
            while unvisited:
                nearest_customer, nearest_distance = None, float('inf')
                for customer in unvisited:
                    distance = calculate_distance(self.customers[truck.route[-1]], customer)
                    if truck.remaining_capacity >= customer.demand and distance < nearest_distance:
                        nearest_customer, nearest_distance = customer, distance
                if nearest_customer:
                    truck.route.append(nearest_customer.id)
                    truck.remaining_capacity -= nearest_customer.demand
                    unvisited.remove(nearest_customer)
                else:
                    break
            if len(truck.route) > 1:  # Only add if used
                truck.route.append(self.depot.id)
                trucks.append(truck)
        return trucks

    def calculate_cost(self, alpha=1, beta=1000):
        """
        Calculate the cost of the solution using a linear combination of the number of trucks and total distance.
        This allows us to minimize the number of trucks used while minimizing the total distance traveled.
        The weight of the trucks are higher than the weight of the distance to minimize the number of trucks used.
        :param alpha: INT weight for the number of trucks
        :param beta: INT weight for the total distance
        :return: INT cost of the solution
        """
        num_trucks = len(self.routes)
        total_distance = sum(truck.calculate_route_distance(self.customers) for truck in self.routes)
        return alpha * num_trucks + beta * total_distance

    def calculate_total_distance(self):
        """
        Calculate the total distance of the solution by summing the distance of each truck's route.
        :return: int total distance of the solution
        """
        return sum(truck.calculate_route_distance(self.customers) for truck in self.routes)

    def neighbor_solution(self):
        """
        Generate a neighbor solution by attempting to merge two routes together or applying a 2-Opt move to improve a route.
        Ensures truck capacity constraints are respected.
        :return: list[Truck] list of Truck objects
        """
        # Copy current routes for modification
        new_routes = [Truck(t.id, t.capacity, t.route[0]) for t in self.routes]
        for i, truck in enumerate(self.routes):
            new_routes[i].route = truck.route[:]
            new_routes[i].remaining_capacity = truck.remaining_capacity  # Maintain capacity info

        # Attempt to relocate a customer from one truck to another
        if len(new_routes) > 1:
            truck1, truck2 = random.sample(new_routes, 2)
            if len(truck2.route) > 2:  # Truck 2 has customers to relocate
                customer_to_move = random.choice(truck2.route[1:-1])  # Exclude depot
                customer_demand = self.customers[customer_to_move].demand
                if truck1.remaining_capacity >= customer_demand:
                    # Perform relocation
                    truck2.route.remove(customer_to_move)
                    truck2.remaining_capacity += customer_demand
                    truck1.route.insert(-1, customer_to_move)  # Before returning to depot
                    truck1.remaining_capacity -= customer_demand

        # Apply a 2-Opt move to a single route
        selected_truck = random.choice(new_routes)
        if len(selected_truck.route) > 3:  # At least 3 points for 2-Opt
            i, j = sorted(random.sample(range(1, len(selected_truck.route) - 1), 2))
            selected_truck.route[i:j] = reversed(selected_truck.route[i:j])

        # Validate all routes for capacity compliance
        for truck in new_routes:
            used_capacity = sum(self.customers[stop].demand for stop in truck.route[1:-1])  # Exclude depot
            if used_capacity > truck.capacity:
                raise ValueError(f"Truck {truck.id} exceeds capacity after neighbor operation.")

        return new_routes

# Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, initial_solution: VRPTWSolution, customers: list[Customer], depot: Customer, initial_temperature=1000, cooling_rate=0.995, min_temperature=1e-3, max_passes=10):
        """
        Initialize the Simulated Annealing algorithm with the given parameters.
        :param initial_solution: VRPTWSolution object representing the initial solution
        :param customers: list[Customer] list of Customer objects
        :param depot: int ID of the depot location (customer 0)
        :param initial_temperature: int initial temperature of the system
        :param cooling_rate: float cooling rate of the system
        :param min_temperature: float minimum temperature of the system
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.customers = customers
        self.depot = depot
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_passes = max_passes

    def accept_probability(self, delta, temperature):
        """
        Calculate the probability of accepting a worse solution based on the temperature and the change in cost (delta).
        :param delta: change in cost between the new and current solution
        :param temperature: current temperature of the system (higher temperature allows for worse solutions to be accepted)
        :return: int / float probability of accepting the worse solution
        """
        if delta < 0:
            return 1
        return exp(-delta / temperature)

    def optimize(self):
        """
        Optimize the VRPTW problem using Simulated Annealing to minimize the number of trucks used and the total distance traveled.
        :return: VRPTWSolution object representing the best solution found
        """
        for pass_num in range(self.max_passes):
            start_time = time.time()
            self.temperature = 1000  # Reset the temperature for each pass
            while self.temperature > self.min_temperature:
                try:
                    # Generate a new neighbor solution
                    new_routes = self.current_solution.neighbor_solution()
                    new_solution = VRPTWSolution(self.customers, self.current_solution.truck_capacity,
                                                 self.current_solution.max_trucks, self.depot)
                    new_solution.routes = new_routes

                    # Validate capacity constraints
                    for truck in new_solution.routes:
                        used_capacity = sum(self.customers[stop].demand for stop in truck.route[1:-1])  # Exclude depot
                        if used_capacity > truck.capacity:
                            raise ValueError(f"Truck {truck.id} exceeds capacity after optimization step.")

                    # Calculate costs
                    current_cost = self.current_solution.calculate_cost()
                    new_cost = new_solution.calculate_cost()

                    # Accept or reject the new solution
                    if self.accept_probability(new_cost - current_cost, self.temperature) > random.random():
                        self.current_solution = new_solution
                        if new_cost < self.best_solution.calculate_cost():
                            self.best_solution = new_solution

                    self.temperature *= self.cooling_rate


                except ValueError as e:
                    # Log and skip invalid solutions
                    print(f"Skipping invalid solution: {e}")

            print(f"Pass {pass_num + 1}/{self.max_passes} [{time.time() - start_time:.2f}s]")

        return self.best_solution

# Visualization
def plot_routes(solution: VRPTWSolution, customers: list[Customer], filename: str, initial_cost: float, optimized_cost: float, maxpasses: int, opttime: float):
    """
    Plot the optimized truck routes on a 2D graph with the customers and depot locations shown.
    :param solution: VRPTWSolution object representing the optimized solution
    :param customers: list of Customer objects representing the customers
    :return:
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    plt.figure(figsize=(10, 10))

    # Generate distinct colors for routes
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution.routes)))

    for truck, color in zip(solution.routes, colors):
        # Calculate used capacity before starting
        used_capacity = sum(customers[stop].demand for stop in truck.route[1:-1])  # Exclude depot

        # Get coordinates for the route
        route_coords = [(customers[stop].x, customers[stop].y) for stop in truck.route]
        route_coords = np.array(route_coords)

        # Plot the route with a unique color
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', color=color, label=f"Truck {truck.id} [{used_capacity}/{truck.capacity}]")
        # Plot the customers with the same color
        for stop in truck.route[1:-1]:  # Exclude depot (start and end)
            plt.scatter(customers[stop].x, customers[stop].y, color=color, zorder=5)


    # Overlay an SVG for the depot
    depot_image = plt.imread(os.path.join(projectRoot, "src", "warehouse-10-512.png"))
    imagebox = OffsetImage(depot_image, zoom=0.04)  # Adjust zoom for size
    ab = AnnotationBbox(imagebox, (customers[0].x, customers[0].y), frameon=False)
    plt.gca().add_artist(ab)

    # Add the initial cost, the optimized cost and the total number of passes to the plot as text below the graph & legend
    plt.text(0.5, -0.1, f"Initial Cost: {initial_cost:.2f}\nOptimized Cost [{maxpasses} passes - {opttime:.2f}s]: {optimized_cost:.2f} ", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # Add legend and labels
    plt.legend()
    plt.title(f"Optimized Truck Routes for {filename.strip('.txt')}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Load data
def load_customers(file_path: str):
    """
    Load customer data from a file and return a list of Customer objects representing the customers.
    :param file_path: str file path to the customer data file
    :return: list[Customer] list of Customer objects
    """
    customers = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[8:]  # Skip header
        for line in lines:
            if line.strip():
                parts = list(map(float, line.split()))
                customers.append(Customer(int(parts[0]), parts[1], parts[2], int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])))
    return customers

def load_conditions(file_path: str):
    """
    Load conditions data from a file and return a list of conditions.
    :param file_path: str file path to the conditions data file
    :return: list of conditions [Max Trucks: int, Truck Capacity: int]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    return list(map(int, lines[4].split()))

# Main Execution
if __name__ == "__main__":
    filename = "rc201.txt"
    maxpasses = 1

    customers = load_customers(os.path.join(projectRoot, "solomon_instances", filename))  # Load customer data
    conditions = load_conditions(os.path.join(projectRoot, "solomon_instances", filename))  # Load conditions data
    initial_solution = VRPTWSolution(customers, truck_capacity=conditions[1], max_trucks=conditions[0], depot=customers[0])  # Initialize solution
    sa = SimulatedAnnealing(initial_solution, customers, customers[0], max_passes=maxpasses)  # Initialize Simulated Annealing
    start_time = time.time()
    best_solution = sa.optimize()  # Optimize solution
    end_time = time.time()
    plot_routes(best_solution, customers, filename, initial_solution.calculate_total_distance(), best_solution.calculate_total_distance(), maxpasses, (end_time-start_time))  # Plot optimized routes