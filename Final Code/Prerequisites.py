import numpy as np

"""Profile"""
customer_spread = 'C108'  # Name of Solomon Instance R = Random, C = Clustered
capacity = 100  # Maximum Capacity H of the Vehicle
demand_bottom = 10  # Minimum possible customer demands
demand_top = 70  # Maximum possible customer demands
demand_arr = [demand_top, demand_bottom]  # Array of Bounds
demand_mean = np.mean(demand_arr)  # Mean of Demands for Simple Policy

"""Solomon Instance Cutter"""
desired_list = []  # init list
with open('input_string.txt', 'r') as file:
    # Read in and cut input for distance calculations
    output_string = ''
    for line in file:
        values = line.strip().split()
        if len(values) >= 3:
            new_line = ",".join(values[:3]) + "\n"
            output_string += new_line

with open('cut_string.txt', 'w') as file:
    # Write the cut string
    file.write(output_string)

with open('cut_string.txt', 'r') as file:
    # Open and transform cut string into desired_list
    for line in file:
        values = line.strip().split(",")
        if len(values) >= 3:
            position = int(values[0])
            values = values[1:3]
            desired_list.append((int(values[0]), int(values[1])))


def euclidean_distance(x1, y1, x2, y2):
    """calculate distances"""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


data = np.loadtxt("cut_string.txt", delimiter=",")  # Load data from file

"""Create Distance Matrix"""

n = len(data)
distance_matrix = np.zeros((n, n))  # init distance_matrix
for i in range(n):  # Go through data
    for j in range(i + 1, n):
        x1, y1 = data[i, 1], data[i, 2]
        x2, y2 = data[j, 1], data[j, 2]
        distance = euclidean_distance(x1, y1, x2, y2)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance
distance_matrix = distance_matrix.astype(int)
print("1: Distance Matrix Created")

if __name__ == "__main__":
    """Print distance matrix"""
    print(desired_list)
    print(distance_matrix)
    print(demand_mean)
