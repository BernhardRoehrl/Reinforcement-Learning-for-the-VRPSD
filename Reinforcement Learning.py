import random
import numpy as np

random.seed(9001)
apriori_list = [0, 3, 4, 1, 2, 11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 0]
"""imported Apriori-route"""
data = {}
"""Stores the data for the problem."""
data['distance_matrix'] = [
    [
        0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354,
        468, 776, 662
    ],
    [
        548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
        1016, 868, 1210
    ],
    [
        776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164,
        1130, 788, 1552, 754
    ],
    [
        696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
        1164, 560, 1358
    ],
    [
        582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
        1050, 674, 1244
    ],
    [
        274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628,
        514, 1050, 708
    ],
    [
        502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856,
        514, 1278, 480
    ],
    [
        194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320,
        662, 742, 856
    ],
    [
        308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662,
        320, 1084, 514
    ],
    [
        194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388,
        274, 810, 468
    ],
    [
        536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764,
        730, 388, 1152, 354
    ],
    [
        502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114,
        308, 650, 274, 844
    ],
    [
        388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194,
        536, 388, 730
    ],
    [
        354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0,
        342, 422, 536
    ],
    [
        468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536,
        342, 0, 764, 194
    ],
    [
        776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274,
        388, 422, 764, 0, 798
    ],
    [
        662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730,
        536, 194, 798, 0
    ],
]

"""Create Lists and variables"""
customer_list = []
avg_distance = []
avg_list = []
capacity = 20  # Vehicle Capacity Parameter


def create_customer():
    """Creates Customer_List with random demands & position from Apriori_List"""
    customer_list.clear()
    for y in apriori_list:
        customer_x = Customer(random.randrange(1, 11, 1), y)
        if customer_x.position == 0:
            customer_x.demand = 0
        customer_list.append(customer_x)
    return customer_list


class Service:
    """For the vehicle and tracking distance"""
    def __init__(self, position, capacity, distance, step_distance):
        self.exploration_rate = exploration_rate  #  AUFPASSEN HIER WEGEN EXPLORATION RATE RESET
        self.capacity = capacity
        self.step = 0
        self.state = 0
        self.old_capacity = 0
        self.rewards_current_episodes = 0
        self.position = position
        self.distance = distance
        self.step_distance = step_distance
        self.exploration_counter = 0

    def reset_variables(self, capacity):
        """Resetting for outer loop"""
        create_customer()
        self.position = 0
        self.capacity = capacity
        self.distance = 0
        self.step_distance = 0
        self.step = 0
        self.state = 0
        self.old_capacity = 0
        self.rewards_current_episodes = 0
        return self, customer_list

    def position_update(self, next_customer):
        """Update vehicle position"""
        self.position = next_customer
        return self.position

    def distance_update(self, target):
        """Update total vehicle distance travelled"""
        self.distance += data['distance_matrix'][self.position][target]
        self.step_distance += data['distance_matrix'][self.position][target]
        return self.distance, self.step_distance

    def refill(self, current_step, customer):
        """The Agent chooses to refill. This means driving back and to the new customer"""
        self.distance += data['distance_matrix'][self.position][0]
        self.step_distance = data['distance_matrix'][self.position][0]
        self.position = 0
        self.capacity = capacity
        self.distance += data['distance_matrix'][self.position][customer]
        self.step_distance += data['distance_matrix'][self.position][customer]
        self.position_update(customer)
        self.capacity = self.capacity - customer_list[current_step].demand
        customer_list[current_step].demand = 0
        return self.position, self.capacity, self.distance, customer_list[current_step].demand, self.step_distance

    def serve(self, current_step, customer):
        if self.capacity < customer_list[current_step].demand:
            """Failure 1: serve partially -> Go To depot -> Go back -> Fulfill serving"""
            """Hinfahren1 und customer partially bedienen"""
            self.distance += data['distance_matrix'][self.position][customer]
            self.step_distance += data['distance_matrix'][self.position][customer]
            self.position_update(customer)
            customer_list[current_step].demand = customer_list[current_step].demand - self.capacity
            self.capacity = 0
            """Zurückfahren Vom partially bedienten Customer zum Depot"""
            self.distance += data['distance_matrix'][self.position][0]
            self.step_distance += data['distance_matrix'][self.position][0]
            self.position = 0
            self.capacity = capacity
            """Hinfahren2 Zurück zum Customer und rest bedienen"""
            self.distance += data['distance_matrix'][self.position][customer]
            self.step_distance += data['distance_matrix'][self.position][customer]
            self.position_update(customer)
            self.capacity = self.capacity - customer_list[current_step].demand
            customer_list[current_step].demand = 0
            return self.position, self.capacity, self.distance, customer_list[current_step].demand, self.step_distance
        else:
            """Vehicle can serve fully. regular workflow"""
            self.capacity = self.capacity - customer_list[current_step].demand
            customer_list[current_step].demand = 0
            self.distance_update(customer)
            self.position_update(customer)
            return customer_list[current_step].demand, self.capacity, self.position, self.distance, self.step_distance

    def execute_episode(self):
        self.step_distance = 0
        self.old_capacity = self.capacity
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < self.exploration_rate:
            action = random.choice(action_space_size)
            if action == 1:
                """Agent chooses action 1: Vehicle must serve"""
                self.serve(self.step, apriori_list[self.step])
                self.exploration_counter += 1
            else:
                """Agent chooses action 2: refilling"""
                self.refill(self.step, apriori_list[self.step])
                self.exploration_counter += 1
        else:
            action = np.argmin(q_table[self.old_capacity, self.state, :])
            if action == 1:
                self.serve(self.step, apriori_list[self.step])
            elif action == 0:
                self.refill(self.step, apriori_list[self.step])
        reward = self.step_distance / 100
        new_state = self.step + 1
        old_value = q_table[self.old_capacity, self.state, action]
        best_expected_value = np.min(q_table[self.capacity, new_state, :])
        bellman_term = (reward + discount_rate * best_expected_value - old_value)
        q_table[self.old_capacity, self.state, action] = old_value + learning_rate * bellman_term
        self.rewards_current_episodes += reward
        self.step = self.step + 1
        self.state = new_state
        return self, new_state, q_table,

    def post_episode_calculation(self):
        avg_distance.append(self.distance)
        self.exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * episode)
        rewards_all_episodes.append(self.rewards_current_episodes)
        for divider in list_q_table:
            if divider == episode:
                print(divider, "The Simulation explored: ", self.exploration_counter)
        return avg_distance, self, rewards_all_episodes,




class Customer:
    """Customers with demand and position"""
    def __init__(self, demand, position):
        self.demand = demand
        self.position = position

    def set_demand(self, demand):
        """Change Customer Demand"""
        self.demand = demand

def print_final():
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
    avg_distances_per_thousand_episodes = np.split(np.array(avg_distance), num_episodes / 1000)
    count = 1000
    print("\n*********Average reward per thousand episodes******\n")
    for r in rewards_per_thousand_episodes:
        print(count, ":", str(sum(r / 1000)))
        count += 1000
    count = 1000
    print("\n********Average Distance per thousand episodes******\n")
    for d in avg_distances_per_thousand_episodes:
        print(count, ":", str(sum(d / 1000)))
        count += 1000
    """End simulation and show results"""
    # print("\nSimulation is done!!!\n", avg_distance,"\n\n", avg_list)
    # print("\nAVG_Distance = ", np.average(avg_distance))
    print("LEN OF Avg_distance = ", len(avg_distance))
    # Print updated Q-table
    # print("\n**********Q-Table**************\n", q_table)
    print("The Simulation explored: ", vehicle.exploration_counter)
    print(q_table)

"""Q-Learning"""
# Q_Table define actions and states
action_refill = 0
action_serve = 1
action_space_size = [action_refill, action_serve]

state_customer_list = len(apriori_list)
q_table = np.zeros((capacity + 1, state_customer_list + 1, len(action_space_size)))

num_episodes = 15000
max_steps_per_episode = 99
# List of rewards
rewards_all_episodes = []

learning_rate = 0.04  # Learning rate
discount_rate = 0.75  # Discounting rate

# Exploration parameters
exploration_rate = 1.0  # Exploration rate
max_exploration_rate = 1.0  # Exploration probability at start
min_exploration_rate = 0.001  # Minimum exploration probability
exploration_decay_rate = 0.05  # Exponential decay rate for exploration prob
exploration_counter = 0
list_q_table = [300]

if __name__ == "__main__":

    vehicle = Service(0, 15, 0, 0)
    for episode in range(num_episodes):
        vehicle.reset_variables(capacity)
        for x in apriori_list:
            vehicle.execute_episode()
        vehicle.post_episode_calculation()
    print_final()


