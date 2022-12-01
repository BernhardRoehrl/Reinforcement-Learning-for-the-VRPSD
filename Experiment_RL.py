import random
import numpy as np

simulations = []
for i in range(100):
    simulations.append(i)
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
        customer_x = Customer(random.randrange(1, 11, 1), y)
        if customer_x.position == 0:
            customer_x.demand = 0
        customer_list.append(customer_x)
    return customer_list


class Service:
    """For the vehicle and tracking distance"""

    def __init__(self, position, capacity, distance, step_distance):
        self.capacity = capacity
        self.position = position
        self.distance = distance
        self.step_distance = step_distance

    def print_info(self, current_step, customer):
        """Prints out the general information for every step"""
        print("\nFrom Position", self.position, "To Customer", customer,
              "which has the remaining demand of: ", customer_list[current_step].demand, ", Distance for this = ",
              data["distance_matrix"][self.position][customer],
              "\nTotal Distance travelled", self.distance, ", Vehicle Position = ",
              self.position, ", next_customer = ", customer, ", step = ", current_step)

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
        self.print_info(current_step, customer)
        print("The Agent chooses to refill in anticipation. Therefore the Vehicle will now return to the Depot" \
              "refill extra distance1 will be = ", data['distance_matrix'][self.position][0])
        self.distance += data['distance_matrix'][self.position][0]
        self.step_distance = data['distance_matrix'][self.position][0]
        self.position = 0
        self.capacity = 15
        self.distance += data['distance_matrix'][self.position][customer]
        self.step_distance += data['distance_matrix'][self.position][customer]
        self.print_info(current_step, customer)
        self.position_update(customer)
        self.capacity = self.capacity - customer_list[current_step].demand
        customer_list[current_step].demand = 0
        return self.position, self.capacity, self.distance, customer_list[current_step].demand, self.step_distance

    def refill_automatic(self):
        """Failure 0: Refill vehicle in case capacity == 0 after serving a customer fully"""
        self.distance += data['distance_matrix'][self.position][0]
        self.step_distance += data['distance_matrix'][self.position][0]
        print("\nFailure0: Vehicle is empty after last customer and needs refill. Travelling from: ", self.position,
              "to the depot extra distance: ", data['distance_matrix'][self.position][0], "total distance: ",
              self.distance)
        self.position = 0
        self.capacity = 15
        print("\nVehicle is at depot and refilled")
        return self.distance, self.position, self.capacity, self.step_distance

    def serve(self, current_step, customer):
        print("The Agent chooses to serve if possible")
        if self.capacity < customer_list[current_step].demand:
            """Failure 1: serve partially -> Go To depot -> Go back -> Fulfill serving"""
            print("\n *****Serving will not be fully!!******")
            """Hinfahren1 und customer partially bedienen"""
            self.print_info(current_step, customer)
            self.distance += data['distance_matrix'][self.position][customer]
            self.step_distance += data['distance_matrix'][self.position][customer]
            self.position_update(customer)
            customer_list[current_step].demand = customer_list[current_step].demand - self.capacity
            self.capacity = 0
            print("\nFailure1: This customer couldn't be served fully!"
                  "Detour will be performed and customer will be revisited to",
                  "fulfill remaining demand of ", customer_list[current_step].demand, "but Vehicle has ",
                  self.capacity, "capacity left, current distance = ", self.distance)
            """Vom partially bedienten Customer zum Depot"""
            print("The self has traveled to the depot to refill extra distance1 = ",
                  data['distance_matrix'][customer][0])
            self.distance += data['distance_matrix'][self.position][0]
            self.step_distance += data['distance_matrix'][self.position][0]
            self.position = 0
            self.capacity = 15
            """Zurück zum Customer und rest bedienen"""
            self.distance += data['distance_matrix'][self.position][customer]
            self.step_distance += data['distance_matrix'][self.position][customer]
            self.print_info(current_step, customer)
            self.position_update(customer)
            self.capacity = self.capacity - customer_list[current_step].demand
            customer_list[current_step].demand = 0
            return self.position, self.capacity, self.distance, customer_list[current_step].demand, self.step_distance
        else:
            """Vehicle can serve fully. regular workflow"""
            print("\n The Customer can be served fully\n")
            self.print_info(current_step, customer)
            self.capacity = self.capacity - customer_list[current_step].demand
            customer_list[current_step].demand = 0
            self.distance_update(customer)
            self.position_update(customer)
            return customer_list[current_step].demand, self.capacity, self.position, self.distance, self.step_distance

    def perform_choosen_action(self, current_step, customer, action, step):
        """Perform action chosen by Agent"""
        self.action = action
        if self.action == [0, 1]:
            self.refill()
        if self.action == [1, 0]:
            self.serve()
        reward = self.step_distance * -10
        return step+1, reward

    def __str__(self):
        """Print out vehicle properties"""
        return "\nVehicle Capacity: " + str(self.capacity) + "\nVehicle Position: " + str(
            self.position) + "\nVehicle Distance: " + str(self.distance) + "\nVehicle Step Distance: " \
               + str(self.step_distance)


class Customer:
    """Customers with demand and position"""

    def __init__(self, demand, position):
        self.demand = demand
        self.position = position

    def __str__(self):
        """Print out the Customer_List or specifics"""
        if self.position == 0:
            return "\nDepot at Position: " + str(self.position) + "\nDemand: " + str(self.demand)
        else:
            return "\nName+Position of Customer: " + str(self.position) + "\nDemand: " + str(self.demand)

    def __repr__(self):
        """Print out all of customer_list[]"""
        return self.__str__()

    def set_demand(self, demand):
        """Change Customer Demand"""
        self.demand = demand


"""Q-Learning"""
# Q_Table define actions and states
action_refill = [0, 1]
action_serve = [1, 0]
action_space_size = [action_refill, action_serve]

state_space_size = 7
#state_space_size = [Service.capacity, Service.position, step]

q_table = np.zeros((state_space_size, len(action_space_size)))

num_episodes = 1
# List of rewards
rewards_all_episodes = []

num_episodes = len(simulations)  # Total simulations
max_steps_per_simulations = 5  # Max steps per simulation = redundant cause of fixed list

learning_rate = 0.09  # Learning rate
discount_rate = 0.95  # Discounting rate

# Exploration parameters
exploration_rate = 1.0  # Exploration rate
max_exploration_rate = 1.0  # Exploration probability at start
min_exploration_rate = 0.0001  # Minimum exploration probability
exploration_decay_rate = 0.002  # Exponential decay rate for exploration prob

if __name__ == "__main__":

    for episode in range(num_episodes):
        for i in simulations:
            """Performing simulation multiple times and resetting for next cycle"""
            print("\n\nSimulation: ", simulations[i])
            create_customer()
            vehicle = Service(0, 15, 0, 0)  # care for defs refilling capacity
            next_customer = 0
            step = 0
            rewards_current_episodes = 0
            state = 0

            for x in apriori_list:
                """Reset step_distance for reward"""
                vehicle.step_distance = 0
                """Serve every customer in order of Apriori_List"""
                print("\n\n\nStep: ", step, "\n", vehicle)
                print("Customer Status:\n", customer_list)
                if vehicle.capacity == 0:
                    """Failure0: Vehicle capacity = 0 after last FULLY serving a customer"""
                    vehicle.refill_automatic()
                """Continue to next customer"""
                next_customer = apriori_list[step]
                # 3. Choose an action a in the current world state (s)
                # First randomize a number
                exploration_rate_threshold = random.uniform(0, 1)
                # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exploration_rate_threshold < exploration_rate:
                    action = np.argmax(q_table[state, :])
                # Else doing a random choice --> exploration
                else:
                    action = random.choice(action_space_size)
                    if action == [1, 0]:
                        """Agent chooses action 1: Vehicle must serve"""
                        vehicle.serve(step, next_customer)
                    else:
                        """Agent chooses action 2: refilling"""
                        vehicle.refill(step, next_customer)
                # Take the action and observe the outcome state(s') and reward (r)
                new_state = step+1
                reward = vehicle.step_distance/-10
                print("\n\n\n***************new_state, reward, =", new_state, reward, "***************")

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma* max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                q_table[state, action] = q_table[state, action] + learning_rate * (
                        reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])

                rewards_current_episodes += reward

                """Begin next cycle"""
                step = step + 1
                state = new_state
                print("\nStep finished: ", vehicle)
            """Add cycle's resulting total distance to list for avg"""
            avg_distance.append(vehicle.distance)
            # Reduce epsilon (for less exploration later on)
            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
                -exploration_decay_rate * simulations[i])
            rewards_all_episodes.append(rewards_current_episodes)
        """End simulation and show results"""
        print("\nSimulation is done!!!\n", avg_distance)
        print(np.mean(avg_distance))

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 50)
    count = 50
    print("\n*********Average reward per thousand episodes******\n")
    for r in rewards_per_thousand_episodes:
        print(count, ":", str(sum(r / 1000)))
        count += 50

    # Print updated Q-table
    print("\n**********Q-Table**************\n", q_table)
