import time
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from mamp.envs import env
from mamp.tools.utils import l2norm, path_length
from draw.plt2d import plt_visulazation
from mamp.configs.config import envs_type
from mata.ta_config.general_config import build_agents_and_tasks, read_json
from mata.ta_config.general_config import build_obj_pos_large, build_cost_path_matrix


# Compute the Euclidean distance between two points
def distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5


def create_data_model(robs_info, tasks_pos, dist_u2t, dist_t2t, dist_t2u):
    """Create the data model for the problem."""
    num_targets = len(tasks_pos)
    num_robots = len(robs_info)
    data = {}

    # Generate depot coordinate points
    coords = []
    for rob in robs_info:
        coords.append((rob.ta_pos_[0], rob.ta_pos_[1]))
    # coords = [(50, 50), (20, 80), (80, 80), (20, 20), (80, 20)]

    # Generate target coordinates
    for i in range(num_targets):
        x = tasks_pos[i][0]
        y = tasks_pos[i][1]
        coords.append((x, y))

    # Node indices
    all_nodes = list(range(num_robots + num_targets))
    depots = list(range(num_robots))  # Depot node indices

    # Compute distance matrix
    dist = {}
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                dist[(i, j)] = 0
            elif i >= num_robots and j >= num_robots:       # target to target
                i_idx, j_idx = i - num_robots, j - num_robots
                dist[(i, j)] = dist_t2t[int(tasks_pos[i_idx][2])][int(tasks_pos[j_idx][2])]
            elif i < num_robots <= j:                       # robot to target
                i_idx, j_idx = i, j - num_robots
                dist[(i, j)] = dist_u2t[int(robs_info[i_idx].ta_pos_[2])][int(tasks_pos[j_idx][2])]
            elif j < num_robots <= i:                       # target to robot
                i_idx, j_idx = i - num_robots, j
                dist[(i, j)] = dist_t2u[int(robs_info[j_idx].ta_pos_[2])][int(tasks_pos[i_idx][2])]
            else:
                dist[(i, j)] = distance(coords[i], coords[j])

    # Convert distance matrix to a 2D list
    distance_matrix = []
    for i in all_nodes:
        row = []
        for j in all_nodes:
            row.append(dist[(i, j)])
        distance_matrix.append(row)

    # Fill the data model
    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = num_robots
    data['depots'] = depots  # Start node for each robot
    data['demands'] = [0] * num_robots + [1] * num_targets  # Depot demand is 0, target demand is 1
    data['vehicle_capacities'] = [15] * num_robots  # Capacity of each robot
    data['coords'] = coords  # Store coordinates

    return data


def get_solution(data, manager, routing, solution):
    """Return the node indices for each route."""
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != vehicle_id:
                route.append(node_index - data['num_vehicles'])
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    return routes


def plot_routes(data, routes):
    """Plot the routes."""
    coords = data['coords']
    depots = data['depots']
    num_robots = data['num_vehicles']

    # Plot all points
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(coords):
        if i in depots:
            plt.plot(x, y, 'ks', markersize=10, label=f'Depot {i}' if i == depots[0] else "")  # Depots as black squares
        else:
            plt.plot(x, y, 'bo')  # Targets as blue circles

    # Plot each route
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Different colors for different robots
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            start_node = route[j]
            end_node = route[j + 1]
            start_coord = coords[start_node]
            end_coord = coords[end_node]
            plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                     color=color, linestyle='-', linewidth=2, label=f'Vehicle {i}' if j == 0 else "")

    # Add legend and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Multi-Depot Multi-TSP Routes')
    plt.legend()
    plt.grid(True)
    plt.show()


def allocateTargets(robs_info, tasks_pos, scale, name=None):
    task_num = len(tasks_pos)
    robot_num = len(robs_info)
    if name is not None:  # Use the global planner to compute obstacle-avoiding paths and path costs
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)

    # Create the data model
    data = create_data_model(robs_info, tasks_pos, dist_u2t, dist_t2t, dist_t2u)

    # Create routing index manager
    ts = time.time()
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),  # Number of nodes
        data['num_vehicles'],  # Number of vehicles
        data['depots'],  # Start node for each vehicle
        data['depots']   # End node for each vehicle (return to its own depot)
    )

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback
    def distance_callback(from_index, to_index):
        """Return the distance between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Set arc costs
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        """Return the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Null capacity slack
        data['vehicle_capacities'],  # Vehicle capacities
        True,  # Start cumul to zero
        'Capacity'
    )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    # search_parameters.log_search = True  # Let the solver decide the termination time

    # search_parameters.time_limit.FromSeconds(120)  # Increase solving time
    search_parameters.solution_limit = 100

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    routes = get_solution(data, manager, routing, solution)
    te = time.time()

    all_path = [[] for _ in range(robot_num)]
    dist_costs = []
    assigned_num = 0
    rewards = 0.0
    path_list = [[] for _ in range(robot_num)]

    for i in range(len(routes)):
        path = []
        i_idx = int(robs_info[i].ta_pos_[2])
        tour = routes[i]
        for j in range(len(tour)):
            j_idx = int(tasks_pos[tour[j]][2])
            if j == 0:  # first
                path += path_u2t[i_idx][j_idx]
            else:
                j_idx_last = int(tasks_pos[tour[j - 1]][2])
                path.pop()
                path += path_t2t[j_idx_last][j_idx]
            if j == len(tour) - 1:  # last
                path.pop()
                path_t2e = path_u2t[i_idx][j_idx]
                path_t2e.reverse()
                path += path_t2e
        all_path[i] = path
        dist_costs.append(path_length(path))
        path_list[i] = tour
        assigned_num += len(tour)

    return path_list, rewards, te - ts, dist_costs, all_path


def allocateTargets_ortools(robs_info, tasks_pos, scale, name):
    robot_num = len(robs_info)

    if name is not None:  # Use the global planner to compute obstacle-avoiding paths and path costs
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)

    # Create the data model
    data = create_data_model(robs_info, tasks_pos, dist_u2t, dist_t2t, dist_t2u)

    # Create routing index manager
    ts = time.time()
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),  # Number of nodes
        data['num_vehicles'],  # Number of vehicles
        data['depots'],  # Start node for each vehicle
        data['depots']   # End node for each vehicle (return to its own depot)
    )

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback
    def distance_callback(from_index, to_index):
        """Return the distance between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Set arc costs
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        """Return the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Null capacity slack
        data['vehicle_capacities'],  # Vehicle capacities
        True,  # Start cumul to zero
        'Capacity'
    )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    # search_parameters.log_search = True  # Let the solver decide the termination time

    # search_parameters.time_limit.FromSeconds(30)  # Increase solving time
    search_parameters.solution_limit = 100

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    routes = get_solution(data, manager, routing, solution)
    te = time.time()

    all_path = [[] for _ in range(robot_num)]
    dist_costs = []
    assigned_num = 0
    path_list = [[] for _ in range(robot_num)]

    for i in range(len(routes)):
        path = []
        i_idx = int(robs_info[i].ta_pos_[2])
        tour = routes[i]
        for j in range(len(tour)):
            j_idx = int(tasks_pos[tour[j]][2])
            if j == 0:  # first
                path += path_u2t[i_idx][j_idx]
            else:
                j_idx_last = int(tasks_pos[tour[j - 1]][2])
                path.pop()
                path += path_t2t[j_idx_last][j_idx]
            if j == len(tour) - 1:  # last
                path.pop()
                path_t2e = path_u2t[i_idx][j_idx]
                path_t2e.reverse()
                path += path_t2e
        all_path[i] = path
        dist_costs.append(path_length(path))
        path_list[i] = tour
        assigned_num += len(tour)

    max_dist = max(dist_costs)
    if assigned_num < len(tasks_pos):
        print("unfinished ortools, assigned_num: ", assigned_num)
        max_dist = robs_info[0].max_endurance_ * 2.0
    print('ortools: ', te - ts, max_dist)
    return path_list, te - ts, max_dist, all_path


def main():
    """Main function."""
    envs = env.Env(envs_type[5])
    agents, task_area, tars_pos, scl = build_obj_pos_large(envs, ag_num=10, tar_num=100)
    name = '/large_u10t100.json'
    tar_ids, res, allocost, dists, all_paths = allocateTargets(agents, tars_pos, scl, name)

    print("max travel distance: ", max(dists), dists)
    print("tar_id_list: ", tar_ids)

    # Plot
    all_length = 0.0
    for k in range(len(all_paths)):
        if len(all_paths[k]) == 0:
            all_paths[k].append(agents[k].pos_global_frame)
        length = 0.0
        for n in range(len(all_paths[k]) - 1):
            length += l2norm(all_paths[k][n], all_paths[k][n + 1])
        all_length += length
    print(all_length)

    assigned_num = 0
    empty_num = 0
    for tars in tar_ids:
        assigned_num += len(tars)
        if len(tars) == 0:
            empty_num += 1

    print("Time used [sec]: ", allocost)
    # print("all_rewards: ", res)
    # print("sum all distance: ", sum(dists))
    # print("all path length: ", all_length)
    print("assigned_num", assigned_num)
    print("empty num: ", empty_num)

    robs_nest = agents[0].ending_area_
    robs, tasks = build_agents_and_tasks(agents, tars_pos, tar_ids, envs)

    plt_visulazation(all_paths, robs, tasks, envs.poly_obs, task_area, robs_nest)


if __name__ == '__main__':
    main()
