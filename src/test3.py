import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, exp
import os
from functools import lru_cache
import time
import concurrent.futures

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

    def can_visit(self, customer: Customer, current_location: Customer):
        """
        Check if the truck can visit the customer within time and capacity constraints.
        :param customer: The customer to visit next.
        :param current_location: The truck's current location.
        :return: True if visit is feasible, otherwise False.
        """
        arrival_time = self.calculate_arrival_time(current_location, customer)

        # Check capacity constraint
        if self.remaining_capacity < customer.demand:
            return False

        # Check time window constraint
        if arrival_time > customer.due_date:
            return False

        return True

    def visit(self, customer: Customer, current_location: Customer):
        """
        Update the truck's state after visiting a customer.
        :param customer: The customer being visited.
        :param current_location: The truck's current location.
        """
        travel_time = calculate_distance(current_location, customer)
        arrival_time = self.current_time + travel_time
        # Wait if arriving early
        self.current_time = max(arrival_time, customer.ready_time) + customer.service_time
        self.remaining_capacity -= customer.demand
        self.total_distance += travel_time

    def calculate_arrival_time(self, current_location: Customer, customer: Customer):
        """
        Calculates the estimated arrival time at a customer location.
        :param current_location: The current location of the truck (Customer object).
        :param customer: The customer to visit next (Customer object).
        :return: int estimated arrival time at the customer location.
        """
        travel_time = calculate_distance(current_location, customer)
        return self.current_time + travel_time


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
        :param depot: Customer object representing the depot location (customer 0)
        """
        self.customers = customers
        self.truck_capacity = truck_capacity
        self.max_trucks = max_trucks
        self.depot = depot
        self.routes = self.generate_initial_solution()

    def generate_initial_solution(self):
        """
        Generate an initial solution by respecting constraints on capacity and time windows.
        """
        unvisited = set(self.customers[1:])  # Exclude the depot
        trucks = []

        for truck_id in range(self.max_trucks):
            truck = Truck(truck_id, self.truck_capacity, self.depot.id)
            while unvisited:
                current_location = self.customers[truck.route[-1]]
                feasible_customers = [
                    customer for customer in unvisited if truck.can_visit(customer, current_location)
                ]
                if not feasible_customers:
                    break

                # Prioritize harder-to-visit customers (e.g., close due date, higher demand)
                feasible_customers.sort(key=lambda c: (c.due_date, -c.demand))
                best_customer = feasible_customers[0]

                truck.visit(best_customer, current_location)
                truck.route.append(best_customer.id)
                unvisited.remove(best_customer)

            if len(truck.route) > 1:
                truck.route.append(self.depot.id)  # Return to depot
                trucks.append(truck)
            if not unvisited:
                break  # All customers assigned

        return trucks

    def neighbor_solution(self):
        """
        Generates a neighbor solution with strategies for adding/removing trucks and balancing routes.
        Ensures all generated routes are feasible.
        """
        new_routes = [Truck(t.id, t.capacity, t.route[0]) for t in self.routes]
        for i, truck in enumerate(self.routes):
            new_routes[i].route = truck.route[:]
            new_routes[i].remaining_capacity = truck.remaining_capacity

        # Choose a strategy for neighbor generation
        strategy = random.choice([
            "merge_routes", "move_customer", "swap_between_routes",
            "reorder_within_route", "add_truck", "remove_empty_trucks"
        ])
        depot_id = self.depot.id

        if strategy == "merge_routes" and len(new_routes) > 1:
            # Merge two routes
            truck1, truck2 = random.sample(new_routes, 2)
            combined_route = truck1.route[:-1] + truck2.route[1:-1]  # Merge routes, excluding depots

            # Check feasibility
            if self.is_feasible_route(combined_route):
                truck1.route = combined_route + [depot_id]
                truck2.route = [depot_id]  # Empty truck2
                truck2.remaining_capacity = truck2.capacity

        elif strategy == "move_customer" and len(new_routes) > 1:
            # Move a customer from one route to another
            truck1, truck2 = random.sample(new_routes, 2)
            if len(truck1.route) > 2:  # Ensure truck1 has customers to move
                customer = random.choice(truck1.route[1:-1])  # Exclude depot
                truck1.route.remove(customer)
                truck2.route.insert(-1, customer)  # Add to truck2

                # Validate both routes
                if not (self.is_feasible_route(truck1.route) and self.is_feasible_route(truck2.route)):
                    # Revert changes if invalid
                    truck2.route.remove(customer)
                    truck1.route.insert(-1, customer)

        elif strategy == "swap_between_routes" and len(new_routes) > 1:
            # Swap customers between routes
            truck1, truck2 = random.sample(new_routes, 2)
            if len(truck1.route) > 2 and len(truck2.route) > 2:  # Ensure both have customers
                customer1 = random.choice(truck1.route[1:-1])  # Exclude depots
                customer2 = random.choice(truck2.route[1:-1])

                # Temporarily swap customers
                truck1.route.remove(customer1)
                truck2.route.remove(customer2)
                truck1.route.insert(-1, customer2)
                truck2.route.insert(-1, customer1)

                # Validate both routes
                if not (self.is_feasible_route(truck1.route) and self.is_feasible_route(truck2.route)):
                    # Revert changes if invalid
                    truck1.route.remove(customer2)
                    truck2.route.remove(customer1)
                    truck1.route.insert(-1, customer1)
                    truck2.route.insert(-1, customer2)

        elif strategy == "reorder_within_route":
            # Reorder customers within a single route (2-opt)
            truck = random.choice(new_routes)
            if len(truck.route) > 3:  # At least 3 nodes for meaningful reordering
                i, j = sorted(random.sample(range(1, len(truck.route) - 1), 2))  # Exclude depots
                new_route = truck.route[:i] + truck.route[i:j + 1][::-1] + truck.route[j + 1:]

                # Validate the reordered route
                if self.is_feasible_route(new_route):
                    truck.route = new_route

        elif strategy == "add_truck" and len(new_routes) < self.max_trucks:
            # Add a new truck if needed
            new_truck = Truck(len(new_routes), self.truck_capacity, depot_id)
            # Attempt to split the longest route
            longest_route = max(new_routes, key=lambda t: t.calculate_route_distance(self.customers))
            if len(longest_route.route) > 3:  # At least 2 deliveries to split
                split_point = len(longest_route.route) // 2
                new_truck.route = longest_route.route[split_point:]
                longest_route.route = longest_route.route[:split_point] + [depot_id]

                # Validate routes after split
                if self.is_feasible_route(new_truck.route) and self.is_feasible_route(longest_route.route):
                    new_routes.append(new_truck)
                else:
                    # Revert split if invalid
                    longest_route.route += new_truck.route[1:]
                    new_truck.route = [depot_id]

        elif strategy == "remove_empty_trucks":
            # Remove empty trucks
            new_routes = [truck for truck in new_routes if len(truck.route) > 2]

        # Ensure all routes are feasible before returning
        for truck in new_routes:
            if not self.is_feasible_route(truck.route):
                return self.routes  # Return current solution if invalid neighbor is generated

        return new_routes

    def is_feasible_route(self, route):
        """
        Check if a route is feasible with respect to time windows and capacity constraints.
        """
        current_capacity = 0
        current_time = 0
        current_customer = self.depot

        for customer_id in route[1:]:  # Skip the depot
            customer = self.customers[customer_id]
            current_capacity += customer.demand
            travel_time = calculate_distance(current_customer, customer)
            arrival_time = current_time + travel_time

            if current_capacity > self.truck_capacity or arrival_time > customer.due_date:
                return False

            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_customer = customer

        return True

    def calculate_cost(self, alpha=100, beta=1000, gamma=100):  # Factors might need a little fine tuning
        """
        Calculate the cost of the solution with weights on minimizing trucks, distance, and underutilized trucks.
        :param alpha: INT weight for the number of trucks.
        :param beta: INT weight for the total distance.
        :param gamma: INT weight for underutilized trucks.
        :return: INT cost of the solution.
        """
        num_trucks = len([truck for truck in self.routes if len(truck.route) > 2])  # Exclude empty trucks
        total_distance = sum(truck.calculate_route_distance(self.customers) for truck in self.routes)

        # Penalize underutilized trucks
        underutilized_trucks = sum(1 for truck in self.routes if len(truck.route) <= 3)  # Depot + 1 or 2 customers
        underutilization_penalty = gamma * underutilized_trucks

        return alpha * num_trucks + beta * total_distance + underutilization_penalty

    def calculate_distance(self):
        return sum(truck.calculate_route_distance(self.customers) for truck in self.routes)

# Simulated Annealing
class SimulatedAnnealing:
    def __init__(self, initial_solution: VRPTWSolution, initial_temperature=1000.0, cooling_rate=0.995, min_temperature=1e-3):
        """
        Initialize the Simulated Annealing algorithm.
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

    def accept_probability(self, delta, temperature):
        """
        Calculate the probability of accepting a worse solution.
        """
        if delta < 0:
            return 1
        return exp(-delta / temperature)

    def generate_and_evaluate_neighbor(self, current_cost):
        """
        Generate a neighbor solution and evaluate its cost.
        :param current_cost: INT Current solution cost.
        :return: (VRPTWSolution, cost)
        """
        neighbor_routes = self.current_solution.neighbor_solution()
        new_solution = VRPTWSolution(
            self.current_solution.customers,
            self.current_solution.truck_capacity,
            self.current_solution.max_trucks,
            self.current_solution.depot,
        )
        new_solution.routes = neighbor_routes
        new_cost = new_solution.calculate_cost()
        return new_solution, new_cost

    def optimize(self, max_iterations=1000, num_workers=16):
        """
        Optimize the VRPTW problem using Simulated Annealing with parallel neighbor evaluation.
        :param max_iterations: INT Maximum iterations per temperature level.
        :param num_workers: INT Number of parallel workers.
        """
        iteration = 0
        while self.temperature > self.min_temperature:
            neighbors = []
            costs = []
            current_cost = self.current_solution.calculate_cost()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Generate neighbors in parallel
                futures = [executor.submit(self.generate_and_evaluate_neighbor, current_cost) for _ in
                           range(max_iterations)]
                for future in concurrent.futures.as_completed(futures):
                    neighbor, cost = future.result()
                    neighbors.append(neighbor)
                    costs.append(cost)

            # Select the best neighbor
            best_idx = costs.index(min(costs))
            best_neighbor = neighbors[best_idx]
            best_cost = costs[best_idx]

            # Accept or reject the best neighbor
            if self.accept_probability(best_cost - current_cost, self.temperature) > random.random():
                self.current_solution = best_neighbor
                if best_cost < self.best_solution.calculate_cost():
                    self.best_solution = best_neighbor

            # Cool down
            self.temperature *= self.cooling_rate
            iteration += 1
            active_trucks = len([truck for truck in self.best_solution.routes if len(truck.route) > 2])
            print(
                f"Iteration {iteration}: Temperature {self.temperature:.4f}, Best Cost {self.best_solution.calculate_cost()}, Active Trucks: {active_trucks}")

        return self.best_solution


# Visualization
def plot_routes(solution: VRPTWSolution, customers: list[Customer], filename: str, inicost: float, optcost: float, opttime: float):
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
    plt.text(0.5, -0.1,
             f"Initial Cost: {inicost:.2f}\nOptimized Cost [{opttime:.2f}s]: {optcost:.2f} ",
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

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

def isBestSolutionValid(best_solution: VRPTWSolution):
    """
    Check if the best solution is valid by ensuring all routes are feasible.
    :param best_solution: VRPTWSolution object representing the best solution
    :return: bool True if the solution is valid, otherwise False
    """
    for truck in best_solution.routes:
        if not best_solution.is_feasible_route(truck.route):
            return False
    return True

# Main Execution
if __name__ == "__main__":
    filename = ("c109.txt")

    customers = load_customers(os.path.join(projectRoot, "solomon_instances", filename))  # Load customer data
    conditions = load_conditions(os.path.join(projectRoot, "solomon_instances", filename))  # Load conditions data
    initial_solution = VRPTWSolution(customers, truck_capacity=conditions[1], max_trucks=conditions[0], depot=customers[0])  # Initialize solution
    sa = SimulatedAnnealing(
        initial_solution=initial_solution,
        initial_temperature=1000.0,  # Higher -> More exploration, but risk of accepting worse solutions
        cooling_rate=0.99,  # Higher -> Faster convergence, but risk of local minima
        min_temperature=0.001  # Lower -> More iterations, but better results

    )
    start_time = time.time()
    best_solution = sa.optimize(max_iterations=50)  # Optimize solution
    for i in range(len(best_solution.routes)):
        print(f"Truck {i}: {best_solution.routes[i].route}")
    end_time = time.time()
    print(f"Solution is valid: {isBestSolutionValid(best_solution)}")
    plot_routes(best_solution, customers, filename, initial_solution.calculate_distance(), best_solution.calculate_distance(), (end_time-start_time))  # Plot optimized routes
