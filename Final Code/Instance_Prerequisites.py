import numpy as np
import os


class Instance:
    # Initialize the instance with given parameters
    def __init__(self, filename, size, capacity, demand_bottom, demand_top):
        self.filedir = r'D:\Code\VRP OR\Solomon Instances'  # Define directory where instances are stored
        self.filename = os.path.join(self.filedir, filename if filename.endswith('.txt') else filename + ".txt")
        # Join the directory and filename to get full file path
        self.size = size  # The size of the instance
        if size <= 20:  # Dynamic Sizing Category
            self.instance_size_category = "small"
        elif 30 < size <= 60:
            self.instance_size_category = "medium"
        else:
            self.instance_size_category = "large"
        self.demand_top = demand_top
        self.demand_bottom = demand_bottom
        self.capacity = capacity
        self.demand_mean = np.mean([self.demand_top, self.demand_bottom])
        self.solomon_id, self.customers = self.parse_file()
        self.distance_matrix, self.customers = self.create_distance_matrix()
        self.capacity_check()  # make sure no illegal moves are done
        self.profile = {  # save the profile for instance naming
            "customer_spread": self.solomon_id[0],
            "solomon_id": self.solomon_id[1:],
            "N": int(len((self.customers))) - 1,
            "capacity": self.capacity,
            "Demand_Top": self.demand_top,
            "Demand_Bottom": self.demand_bottom,
            "Capacity Stress": (self.capacity / self.demand_mean),
            "InstanceSize": self.instance_size_category
        }
        self.instance_name = f"{self.profile['customer_spread']}{self.profile['solomon_id']}_N{self.profile['N']}_H{self.profile['capacity']}_D{self.profile['Demand_Bottom']}-{self.profile['Demand_Top']}"
        # One Way Print out of the whole important configuration

    def parse_file(self):
        """Read in the Solomon File's Information and keep the relvant"""
        with open(self.filename, 'r') as file:
            lines = file.readlines()
            solomon_id = lines[0].strip()  # the instance id is always in the first line
            customer_lines = lines[8:]  # remaining lines with relevant information
            customers = []  # init an empty list
            for line in customer_lines:
                details = line.strip().split()
                if len(details) >= 3:  # create a dic with the data we extract
                    customer = {
                        "id": int(details[0]),
                        "x": int(details[1]),
                        "y": int(details[2]),
                    }
                    customers.append(customer)
        return solomon_id, customers

    def capacity_check(self):
        """Check for Illegal Instance Configuration"""
        if self.capacity < self.demand_top:
            raise Exception("!!!ILLEGAL PROFILE CAPCITY HAS TO BE EQUAL OR BIGGER THAN DEMAND TOP!!!")

        if self.demand_bottom > self.demand_top:
            raise Exception("!!!ILLEGAL PROFILE Demand Bottom HAS TO BE EQUAL OR smaller THAN DEMAND TOP!!!")

    def euclidean_distance(self, x1, y1, x2, y2):
        """calculate distances"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def create_distance_matrix(self):
        """Cut the extracted data to desired size and calculate distance matrix"""
        """Cutting"""
        self.customers = [customer for customer in self.customers if customer['id'] <= self.size]

        """Distance Matrix"""
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


"""Testing Purposes"""

if __name__ == "__main__":
    solomon_flex = Instance('C101', size=5, capacity=19, demand_bottom=20, demand_top=19)
    print("Customers: ", solomon_flex.customers)
    print("\nDemand Mean: ", solomon_flex.demand_mean)
    print("\nDM: ", solomon_flex.distance_matrix)
    print(len(solomon_flex.customers))
    print(solomon_flex.distance_matrix)
    print(solomon_flex.instance_name)
