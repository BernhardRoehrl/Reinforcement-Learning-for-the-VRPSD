import numpy as np

"""Profile"""
customer_spread = 'R'
depot_position = '0M_'
capacity = 100
demand_bottom = 10
demand_top = 70
demand_arr = [demand_top, demand_bottom]
demand_mean = np.mean(demand_arr)

# Load data from file
data = np.loadtxt("cutted_string.txt", delimiter=",")
def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
# Compute distance matrix
n = len(data)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        x1, y1 = data[i, 1], data[i, 2]
        x2, y2 = data[j, 1], data[j, 2]
        distance = euclidean_distance(x1, y1, x2, y2)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

distance_matrix = distance_matrix.astype(int)
# Print distance matrix
print(distance_matrix)
print(demand_mean)