import os
from Prerequisites import Instance
import numpy as np

solomon_dir = r'D:\Code\VRP OR\Solomon Instances'

solomon_files = [f for f in os.listdir(solomon_dir) if os.path.isfile(os.path.join(solomon_dir, f))]

customer_sizes = [5, 10, 15, 100]

demand_configurations = [
    {"demand_top": 1, "demand_bottom": 1, "capacity": 1},
    {"demand_top": 10, "demand_bottom": 5, "capacity": 15}
]

testlist = []


for solomon_file in solomon_files:
    file_name = solomon_file
    for customer_size in customer_sizes:
        size = customer_size
        for demand_configuration in demand_configurations:
            demand_top = demand_configuration["demand_top"]
            demand_bottom = demand_configuration["demand_bottom"]
            capacity = demand_configuration["capacity"]
            instance = Instance(file_name, size, capacity, demand_top, demand_bottom)
            testlist.append(instance.instance_name)

with open('loop_results_instances.txt', 'w') as file:
        file.write('\n'.join(testlist))


# for solomon_file in solomon_files:
#     instance = Instance(os.path.join(solomon_dir, solomon_file))
#
#     for customer_size in customer_sizes:
#         instance.reduce_customer_size()
#
#         for demand_top, demand_bottom, capacity in demand_top_bottom_capacity_combinations:
#             instance.demand_top = demand_top
#             instance.demand_bottom = demand_bottom
#             instance.capacity = capacity
#             instance.demand_mean = np.mean([demand_top, demand_bottom])
#
#             reinforcement_learning.py
#             simple_policy.py