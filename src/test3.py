import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, exp
import os

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

    def calculate_cost(self, alpha=1000, beta=1):
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

    def neighbor_solution(self):
        """
        Generate a neighbor solution by attempting to merge two routes together
        :return: list[Truck] list of Truck objects
        """
        new_routes = [Truck(t.id, t.capacity, t.route[0]) for t in self.routes]
        for i, truck in enumerate(self.routes):
            new_routes[i].route = truck.route[:]

        # Attempt to merge two routes
        if len(new_routes) > 1:
            truck1, truck2 = random.sample(new_routes, 2)
            if len(truck2.route) > 2:  # Truck 2 has customers to merge
                customer_to_move = random.choice(truck2.route[1:-1])  # Exclude depot
                if truck1.remaining_capacity >= self.customers[customer_to_move].demand:
                    truck2.route.remove(customer_to_move)
                    truck1.route.insert(-1, customer_to_move)  # Before returning to depot
                    truck1.remaining_capacity -= self.customers[customer_to_move].demand
                    truck2.remaining_capacity += self.customers[customer_to_move].demand

        return new_routes

# Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, initial_solution: VRPTWSolution, customers: list[Customer], depot: Customer, initial_temperature=1000, cooling_rate=0.995, min_temperature=1e-3):
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
        while self.temperature > self.min_temperature:
            new_routes = self.current_solution.neighbor_solution()
            new_solution = VRPTWSolution(self.customers, self.current_solution.truck_capacity, self.current_solution.max_trucks, self.depot)
            new_solution.routes = new_routes
            current_cost = self.current_solution.calculate_cost()
            new_cost = new_solution.calculate_cost()

            if self.accept_probability(new_cost - current_cost, self.temperature) > random.random():
                self.current_solution = new_solution
                if new_cost < self.best_solution.calculate_cost():
                    self.best_solution = new_solution

            self.temperature *= self.cooling_rate
        return self.best_solution

# Visualization
def plot_routes(solution: VRPTWSolution, customers: list[Customer]):
    """
    Plot the optimized truck routes on a 2D graph with the customers and depot locations shown.
    :param solution: VRPTWSolution object representing the optimized solution
    :param customers: list of Customer objects representing the customers
    :return:
    """
    plt.figure(figsize=(10, 10))

    # Generate distinct colors for routes
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution.routes)))

    for truck, color in zip(solution.routes, colors):
        # Get coordinates for the route
        route_coords = [(customers[stop].x, customers[stop].y) for stop in truck.route]
        route_coords = np.array(route_coords)

        # Plot the route with a unique color
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', color=color, label=f"Truck {truck.id}")

        # Plot the customers with the same color
        for stop in truck.route[1:-1]:  # Exclude depot (start and end)
            plt.scatter(customers[stop].x, customers[stop].y, color=color, zorder=5)

    # Plot the depot in black
    plt.scatter(customers[0].x, customers[0].y, color='black', zorder=10, label="Depot")

    plt.legend()
    plt.title("Optimized Truck Routes (Minimized Trucks)")
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
    projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Compatibility fix
    customers = load_customers(os.path.join(projectRoot, "solomon_instances", "rc102.txt"))  # Load customer data
    conditions = load_conditions(os.path.join(projectRoot, "solomon_instances", "rc102.txt"))  # Load conditions data
    initial_solution = VRPTWSolution(customers, truck_capacity=conditions[1], max_trucks=conditions[0], depot=customers[0])  # Initialize solution
    sa = SimulatedAnnealing(initial_solution, customers, customers[0])  # Initialize Simulated Annealing
    best_solution = sa.optimize()  # Optimize solution
    plot_routes(best_solution, customers)  # Plot optimized routes
