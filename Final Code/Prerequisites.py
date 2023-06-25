import numpy as np

"""Profile"""
solomon_id = 0  # Contains Unique ID of Instance and C/R in the beginning for Clustered or Random Customer Spread
capacity = 100  # Maximum Capacity H of the Vehicle
demand_bottom = 10  # Minimum possible customer demands
demand_top = 70  # Maximum possible customer demands
demand_arr = [demand_top, demand_bottom]  # Array of Bounds
demand_mean = np.mean(demand_arr)  # Mean of Demands for Simple Policy


class Instance:

    def __init__(self, filename):
        self.filename = filename
        self.solomon_id, self.customers = self.parse_file()
        self.distance_matrix = self.create_distance_matrix(self.customers)
        self.demand_top = demand_top
        self.demand_bottom = demand_bottom
        self.demand_mean = np.mean([self.demand_top, self.demand_bottom])
        self.capacity = capacity

    def parse_file(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()
            solomon_id = lines[0].strip()  # the instance id is always in the first line
            customer_lines = lines[8:]  # remaining lines with relevant information
            customers = []
            for line in customer_lines:
                details = line.strip().split()
                if len(details) >= 3:
                    customer = {
                        "id": int(details[0]),
                        "x": int(details[1]),
                        "y": int(details[2]),
                    }
                    customers.append(customer)
        return solomon_id, customers

    def euclidean_distance(self, x1, y1, x2, y2):
        """calculate distances"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def create_distance_matrix(self, customers):
        n = len(customers)
        distance_matrix = np.zeros((n, n))  # init distance matrix with zeros
        for i in range(n):  # Go through data
            for j in range(i + 1, n):
                x1, y1 = customers[i]['x'], customers[i]['y']
                x2, y2 = customers[j]['x'], customers[j]['y']
                distance = self.euclidean_distance(x1, y1, x2, y2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        distance_matrix = distance_matrix.astype(int)
        return distance_matrix

    def reduce_customer_size(self, new_size):
        """only an example so far Should do the rear end cutting to the new customer size."""
        # self.customers = self.customers[:new_size]
        # self.distance_matrix = self.distance_matrix[:new_size, :new_size]
        # return self.customers, self.distance_matrix
        pass

    def main(self):
        self.create_distance_matrix(self.customers)
        print("1: Distance Matrix Created")
        print(self.customers)
        print(self.distance_matrix)


if __name__ == "__main__":
    solomon_c108 = Instance('input_string.txt')
    print("Customers: ", solomon_c108.customers)
    print("\nDemand Mean: ", solomon_c108.demand_mean)
    print("\nDM: ", solomon_c108.distance_matrix)
