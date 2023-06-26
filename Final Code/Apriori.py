import re
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from Prerequisites import Instance

class Apriori:

    @staticmethod
    def create_apriori_list(instance):
        distance_matrix = instance.distance_matrix

        def create_data_model():
            data = {}
            data['distance_matrix'] = distance_matrix
            data['num_vehicles'] = 1
            data['depot'] = 0
            return data

        data = create_data_model()
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        def print_solution(data, manager, routing, solution):
                """Prints solution on console."""
                max_route_distance = 0
                for vehicle_id in range(data['num_vehicles']):
                    index = routing.Start(vehicle_id)
                    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                    route_distance = 0
                    while not routing.IsEnd(index):
                        plan_output += ' {} -> '.format(manager.IndexToNode(index))
                        previous_index = index
                        index = solution.Value(routing.NextVar(index))
                        route_distance += routing.GetArcCostForVehicle(
                            previous_index, index, vehicle_id)
                    plan_output += '{}\n'.format(manager.IndexToNode(index))
                    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
                    global apriori_list
                    apriori_list = plan_output
                    apriori_list = apriori_list.split("\n", 1)[1]
                    apriori_list = apriori_list.rsplit("\n", 3)[0]
                    apriori_list = re.sub(r'\(\d+\)|Load|->', '', apriori_list)
                    apriori_list = apriori_list.split()
                    for i in range(0, len(apriori_list)):
                        apriori_list[i] = int(apriori_list[i])
                    max_route_distance = max(route_distance, max_route_distance)
                return apriori_list




        """Entry point of the program."""
        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            return print_solution(data, manager, routing, solution)
        else:
            print('No solution found !')
            return None
        #apriori_list = apriori_list