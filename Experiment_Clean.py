import random
import numpy as np

apriori_list = [0, 3, 4, 1, 2, 0]
"""imported Apriori-route"""
data = {}
"""Stores the data for the problem."""
data['distance_matrix'] = [
    [
        0, 548, 776, 696, 582,
    ],
    [
        548, 0, 684, 308, 194
    ],
    [
        776, 684, 0, 992, 878
    ],
    [
        696, 308, 992, 0, 114
    ],
    [
        582, 194, 878, 114, 0
    ],
]
"""Create Lists and variables"""
customer_list = []
avg_distance = []


def create_customer():
    """Creates Customer_List with random demands & position from Apriori_List"""
    customer_list.clear()
    for y in apriori_list:
        customer_x = Customer(random.randrange(1, 12, 1), y)
        if customer_x.position == 0:
            customer_x.demand = 0
        customer_list.append(customer_x)
    return customer_list


class Service:
    """For the vehicle and tracking distance"""

    def __init__(self, position, capacity, distance):
        self.capacity = capacity
        self.position = position
        self.distance = distance

    def position_update(self, next_customer):
        """Update vehicle position"""
        self.position = next_customer
        return self.position

    def distance_update(self, target):
        """Update total vehicle distance travelled"""
        self.distance += data['distance_matrix'][self.position][target]
        return self.distance

    def serve_regular(self, current_step, customer):
        """Vehicle can serve fully. regular workflow"""
        self.capacity = self.capacity - customer_list[current_step].demand
        customer_list[current_step].demand = 0
        return customer_list[current_step].demand, self.capacity

    def refill(self):
        """Failure 0: Refill vehicle in case capacity == 0 after serving a customer fully"""
        self.distance += data['distance_matrix'][self.position][0]
        self.position = 0
        self.capacity = 15
        return self.distance, self.position, self.capacity

    def server_partially(self, current_step, customer):  # TODO
        """Failure 1: serve partially -> Go To depot -> Go back -> Fulfill serving"""
        self.position_update(customer)
        customer_list[current_step].demand = customer_list[current_step].demand - self.capacity
        self.capacity = 0
        self.distance += data['distance_matrix'][self.position][0]
        self.position = 0
        self.capacity = 15
        self.distance += data['distance_matrix'][self.position][customer]
        self.serve_regular(current_step, customer)
        return self.position, self.capacity, self.distance, customer_list[current_step].demand


class Customer:
    """Customers with demand and position"""

    def __init__(self, demand, position):
        self.demand = demand
        self.position = position

    def set_demand(self, demand):
        """Change Customer Demand"""
        self.demand = demand


if __name__ == "__main__":

    for i in range(10):
        """Performing simulation multiple times and resetting for next cycle"""
        create_customer()
        vehicle = Service(0, 15, 0)
        current_customer = 0
        step = 0
        for x in apriori_list:
            """Serve every customer in order of Apriori_List"""
            if vehicle.capacity == 0:
                """Failure0: Vehicle capacity = 0 after last FULLY serving a customer"""
                vehicle.refill()
            """Continue to next customer"""
            current_customer = apriori_list[step]
            vehicle.distance_update(current_customer)

            if customer_list[step].demand > vehicle.capacity:
                """Failure 1: Vehicle must serve partially and perform detour"""
                vehicle.server_partially(step, current_customer)
            else:
                """Serve fully with enough capacity"""
                vehicle.serve_regular(step, current_customer)
            """Update vehicle position & step for next cycle"""
            vehicle.position_update(current_customer)
            step = step + 1
        """Add cycle's resulting total distance to list for avg"""
        avg_distance.append(vehicle.distance)
    """End simulation and show results"""
    print("\nSimulation is done\n", avg_distance)
    print(np.mean(avg_distance))
