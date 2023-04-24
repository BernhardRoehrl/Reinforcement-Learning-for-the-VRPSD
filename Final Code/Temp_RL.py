import random
import numpy as np
from Apriori_Revisited import apriori_list
from Distance_Matrix import distance_matrix
import csv

# random.seed(9001)
apriori_list = apriori_list
"""imported Apriori-route"""
data = {}
"""Stores the data for the problem."""
data['distance_matrix'] = distance_matrix

"""Create Lists and Parameters"""
customer_list = []
avg_distance = []
capacity = 20  # Vehicle Capacity Parameter

"""Additional Parameters for QLearning"""
action_refill = 0
action_serve = 1
action_space_size = [action_refill, action_serve]

num_episodes = 16000
max_steps_per_episode = 99
rewards_all_episodes = []  # List of rewards

learning_rate = 0.04  # Learning rate
discount_rate = 0.75  # Discounting rate

# Exploration parameters
exploration_rate = 1.0  # Exploration rate
max_exploration_rate = 1.0  # Exploration probability at start
min_exploration_rate = 0.001  # Minimum exploration probability
exploration_decay_rate = 0.05  # Exponential decay rate for exploration prob
exploration_counter = 0
list_q_table = [300]


def create_customer():
    """Creates Customer_List with random demands & position from Apriori_List"""
    customer_list.clear()
    for y in apriori_list:
        customer_x = Customer(random.randrange(10, 20, 1), y)
        if customer_x.position == 0:
            customer_x.demand = 0
        customer_list.append(customer_x)
    return customer_list


class Service:
    """For the vehicle and tracking distance"""

    def __init__(self, position, capacity, distance, step_distance, refillcounter, failurecounter):
        self.capacity = capacity
        self.position = position
        self.distance = distance
        self.step_distance = step_distance
        self.refillcounter = refillcounter
        self.failurecounter = failurecounter

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
        if episode > 15000:
            self.refillcounter += 1
        return self.position, self.capacity, self.distance, customer_list[
            current_step].demand, self.step_distance, self.refillcounter

    def serve(self, current_step, customer):
        if self.capacity < customer_list[current_step].demand:
            """Failure 1: serve partially -> Go To depot -> Go back -> Fulfill serving"""
            """Hinfahren1 und customer partially bedienen"""
            if episode > 15000:
                self.failurecounter += 1
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
            return self.position, self.failurecounter, self.capacity, self.distance, customer_list[
                current_step].demand, self.step_distance
        else:
            """Vehicle can serve fully. regular workflow"""
            self.capacity = self.capacity - customer_list[current_step].demand
            customer_list[current_step].demand = 0
            self.distance_update(customer)
            self.position_update(customer)
            return customer_list[current_step].demand, self.capacity, self.position, self.distance, self.step_distance


class Customer:
    """Customers with demand and position"""

    def __init__(self, demand, position):
        self.demand = demand
        self.position = position

    def set_demand(self, demand):
        """Change Customer Demand"""
        self.demand = demand


class QTable:
    """A class for the q_table and bellman stuff saving as csv afterwards"""

    def __init__(self, capacity, apriori_list, action_space_size):
        self.q_table = np.zeros((capacity + 1, len(apriori_list) + 1, len(action_space_size)))
        self.capacity = capacity
        self.apriori_list = apriori_list
        self.action_space_size = action_space_size

    def __getitem__(self, key):
        """Apparently needed to access my q_table later by [operator"""
        return self.q_table[key]

    def get_value(self, capacity, state, action):
        return self.q_table[capacity, state, action]

    def set_value(self, capacity, state, action, value):
        self.q_table[capacity, state, action] = value

    def save_csv(self, file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Capacity', 'Action Space'] + [f'State {i}' for i in list(range(self.apriori_list + 1))])

            for i, row in enumerate(self.q_table):
                writer.writerow([f'Customer {i}'] + list(row))

    def load_csv(self, file_path):
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)
            self.capacity = int(headers[0])
            self.action_space_size = list(map(int, headers[1].split(', ')))
            self.apriori_list = len(headers) - 2
            self.q_table = np.zeros((self.apriori_list + 1, len(self.action_space_size)))
            for i, row in enumerate(reader):
                for j, value in enumerate(row[1:]):
                    self.q_table[i, j] = float(value)


if __name__ == "__main__":

    q_table = QTable(capacity, apriori_list, action_space_size)
    """This is the q_table as a class member with the parameters from the start"""
    for episode in range(num_episodes):
        """Performing simulation multiple times and resetting for next cycle"""
        create_customer()
        vehicle = Service(0, capacity, 0, 0, 0, 0)  # care for defs refilling capacity
        next_customer = 0
        step = 0
        rewards_current_episodes = 0
        state = 0
        old_capacity = 0
        for x in apriori_list:
            vehicle.step_distance = 0  # Reset step_distance for reward
            next_customer = apriori_list[step]  # Continue to next customer
            old_capacity = vehicle.capacity  # Set current Capacity for state/new_state
            exploration_rate_threshold = random.uniform(0, 1)  # Choosing Exploration or Exploitation (e-greedy method)
            # Exploration part choosing actions on random
            if exploration_rate_threshold < exploration_rate or episode < 15000:
                action = random.choice(action_space_size)
                if action == 1:  # Agent chooses action 1: Vehicle must serve
                    vehicle.serve(step, next_customer)
                    exploration_counter += 1
                else:  # Agent chooses action 2: refilling
                    vehicle.refill(step, next_customer)
                    exploration_counter += 1

            # Exploitation (taking the smallest Q value for this state)
            else:
                action = np.argmin(q_table[old_capacity, state, :])  # Returns 1 or 0
                if action == 1:  # Assigning the corresponding actions
                    vehicle.serve(step, next_customer)
                elif action == 0:
                    vehicle.refill(step, next_customer)
            reward = vehicle.step_distance / 100  # Set the reward based on the distance travelled in this step
            new_state = step + 1  # update new_state for bellman equation

            if episode == 15999:  # print out the best possible result we can have in the last episode
                Last_Episode = (
                    "The last episode does: x", x, "with action: ", action, "and refillcounter = ",
                    vehicle.refillcounter,
                    "and failurecounter = ", vehicle.failurecounter)
                print(Last_Episode)
                data[Last_Episode] = Last_Episode
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma* max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            if episode < 15000:  # stop learning after 15000 episodes
                old_value = q_table[old_capacity, state, action]
                best_expected_value = np.min(q_table[vehicle.capacity, new_state, :])
                bellman_term = (reward + discount_rate * best_expected_value - old_value)
                q_table.set_value(old_capacity, state, action, old_value + learning_rate * bellman_term)

            rewards_current_episodes += reward

            """Begin next cycle"""
            step = step + 1
            state = new_state
        """Add cycle's resulting total distance to list for avg"""
        avg_distance.append(vehicle.distance)
        # Reduce epsilon (for less exploration later on)
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episodes)
        """Print Q-Table after 1000 Iterations"""
        for teiler in list_q_table:
            if teiler == episode:
                # print("**********Q-Table after ", teiler, "Iterations**************\n", q_table)
                print(teiler, "The Simulation explored: ", exploration_counter)

    # Calculate and print the average reward and Distance per thousand episodes
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
    print("LEN OF Avg_distance = ", len(avg_distance))
    print("The Simulation explored: ", exploration_counter)
    np.save("../q_table.npy", q_table)
