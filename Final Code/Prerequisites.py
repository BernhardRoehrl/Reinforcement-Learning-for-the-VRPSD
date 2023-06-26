import numpy as np
import json
import os

"""Profile"""
solomon_id = 0  # Contains Unique ID of Instance and C/R in the beginning for Clustered or Random Customer Spread
capacity = 100  # Maximum Capacity H of the Vehicle
demand_bottom = 10  # Minimum possible customer demands
demand_top = 70  # Maximum possible customer demands
demand_arr = [demand_top, demand_bottom]  # Array of Bounds
demand_mean = np.mean(demand_arr)  # Mean of Demands for Simple Policy


class Instance:

    def __init__(self, filename, size, capacity, demand_bottom, demand_top):
        self.filedir = r'D:\Code\VRP OR\Solomon Instances'
        self.filename = os.path.join(self.filedir, filename if filename.endswith('.txt') else filename + ".txt")
        self.size = size
        self.demand_top = demand_top
        self.demand_bottom = demand_bottom
        self.capacity = capacity
        self.demand_mean = np.mean([self.demand_top, self.demand_bottom])
        self.solomon_id, self.customers = self.parse_file()
        self.distance_matrix, self.customers = self.create_distance_matrix()
        self.capacity_check()
        self.profile = {
            "customer_spread": self.solomon_id[0],
            "solomon_id": self.solomon_id[1:],
            "N": int(len((self.customers)))-1,
            "capacity": self.capacity,
            "Demand_Top": self.demand_top,
            "Demand_Bottom": self.demand_bottom,
            "%": self.capacity/self.demand_mean,
        }
        self.instance_name = f"{self.profile['customer_spread']}{self.profile['solomon_id']}_N{self.profile['N']}_H{self.profile['capacity']}_D{self.profile['Demand_Bottom']}-{self.profile['Demand_Top']}_%_{round(self.profile['%'], 2)}"


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

    def capacity_check(self):
        if self.capacity < self.demand_top:
            raise Exception("!!!ILLEGAL PROFILE CAPCITY HAS TO BE EQUAL OR BIGGER THAN DEMAND TOP!!!")

        if self.demand_bottom > self.demand_top:
            raise Exception("!!!ILLEGAL PROFILE Demand Bottom HAS TO BE EQUAL OR smaller THAN DEMAND TOP!!!")


    def euclidean_distance(self, x1, y1, x2, y2):
        """calculate distances"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def create_distance_matrix(self):
        self.customers = [customer for customer in self.customers if customer['id'] <= self.size]

        n = len(self.customers)
        distance_matrix = np.zeros((n, n))  # init distance matrix with zeros
        for i in range(n):  # Go through data
            for j in range(i + 1, n):
                x1, y1 = self.customers[i]['x'], self.customers[i]['y']
                x2, y2 = self.customers[j]['x'], self.customers[j]['y']
                distance = self.euclidean_distance(x1, y1, x2, y2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        distance_matrix = distance_matrix.astype(int)
        return distance_matrix, self.customers


if __name__ == "__main__":
    solomon_flex = Instance('C101', size=5, capacity=19, demand_bottom=20, demand_top=19)
    print("Customers: ", solomon_flex.customers)
    print("\nDemand Mean: ", solomon_flex.demand_mean)
    print("\nDM: ", solomon_flex.distance_matrix)
    print(len(solomon_flex.customers))
    print(solomon_flex.distance_matrix)
    print(solomon_flex.instance_name)


