import os
import time
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mamp.envs import env
from mamp.tools.utils import l2norm, path_length
from draw.plt2d import plt_visulazation
from mamp.configs.config import envs_type
from mata.ta_config.general_config import build_agents_and_tasks, read_json
from mata.ta_config.general_config import build_obj_pos_large, build_cost_path_matrix


def plot_routes(positions, targets_pos, tour_results, r_id=-1, tar_id=-1):
    """
    Plot routes from vehicles to target locations (using arrows to represent paths).

    Parameters:
    positions (list of tuple): List of vehicle start positions.
    targets_pos (list of tuple): List of target positions.
    tours (list of list): List of routes for each vehicle.
    """
    plt.figure(figsize=(10, 8))

    # Set colors
    sns.set_palette("husl", n_colors=len(positions))
    colors = sns.color_palette("husl", n_colors=len(positions))

    # Plot target positions
    targets_x = [pos[0] for pos in targets_pos]
    targets_y = [pos[1] for pos in targets_pos]
    plt.scatter(targets_x, targets_y, c='gray', label='Targets', s=100, alpha=0.8, edgecolors='gray')
    for i, pos in enumerate(targets_pos):
        plt.text(pos[0], pos[1], str(i), fontsize=12, ha='right', color='black')
    if r_id >= 0:
        plt.plot(targets_pos[tar_id][0], targets_pos[tar_id][1], marker='o', color='red', markersize=15)

    # Plot each vehicle route
    dists = []
    robots_path = []
    for rob_id, tour in tour_results.items():
        pos_x, pos_y = positions[rob_id][0], positions[rob_id][1]
        plt.plot(pos_x, pos_y, marker='*', color=colors[rob_id], markersize=20)
        plt.text(pos_x - 2.0, pos_y, "Start " + str(rob_id), fontsize=15, color=colors[rob_id])
        if not tour:
            continue

        # Build path coordinates
        path_x = [pos_x]
        path_y = [pos_y]
        path = [(pos_x, pos_y)]
        dist = 0.0
        for j in range(len(tour)):
            path_x.append(targets_pos[tour[j]][0])
            path_y.append(targets_pos[tour[j]][1])
            path.append((targets_pos[tour[j]][0], targets_pos[tour[j]][1]))
            if j == 0:
                dist += l2norm(positions[rob_id], targets_pos[tour[j]])
            else:
                dist += l2norm(targets_pos[tour[j - 1]], targets_pos[tour[j]])
        dists.append(dist)
        robots_path.append(path)
        path_x.append(pos_x)
        path_y.append(pos_y)

        # Plot a solid path line (also used to generate legend entries)
        plt.plot(path_x, path_y, color=colors[rob_id], label=f'Robot {rob_id}', markersize=15,
                 linewidth=2, alpha=1)

        # Draw arrow segments along the path
        for i in range(len(path_x) - 1):
            start_x, start_y = path_x[i], path_y[i]
            end_x, end_y = path_x[i + 1], path_y[i + 1]
            plt.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[rob_id],
                    lw=2,
                    mutation_scale=30  # Controls the arrowhead size
                )
            )

    print("maximum distance: ", max(dists))

    # Add legend and labels
    plt.legend(loc='lower left', fontsize=12)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.title('Drone Routes to Targets', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_routes_by_ids(nodes, ids, ego_num):
    """
    Plot routes from vehicles to target locations.
    """
    ids = [idx - 1 for idx in ids]
    ids.pop(0)
    print(111, len(ids), ids)
    plt.figure(figsize=(10, 8))

    # Set colors
    sns.set_palette("husl", n_colors=ego_num)  # Use Seaborn color palette
    colors = sns.color_palette("husl", n_colors=ego_num)  # Generate harmonious colors

    # Plot target positions
    targets_x = [pos[0] for pos in nodes]
    targets_y = [pos[1] for pos in nodes]
    plt.scatter(targets_x, targets_y, c='gray', label='Targets', s=100, alpha=0.8, edgecolors='gray')
    for i, (x, y) in enumerate(nodes):
        if i < len(nodes) - ego_num:
            plt.text(x, y, str(i), fontsize=12, ha='right', color='black', zorder=10)

    for i in range(ego_num):
        plt.plot(nodes[i][0], nodes[i][1], marker='*', color=colors[i], label=f"robot {i}", markersize=20, zorder=3)

    # Plot the tour path
    path_x = []
    path_y = []
    for idx in ids:
        if 0 < idx <= len(nodes) - 1:
            path_x.append(nodes[idx][0])
            path_y.append(nodes[idx][1])
        else:
            path_x.append(path_x[-1])
            path_y.append(path_y[-1])
    plt.plot(path_x, path_y, marker='o', color='orange', label='tour', linewidth=2, markersize=6)

    # Add legend and labels
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.title('Drone Routes to Targets', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def getCostMatrix(robs_info, tasks_pos, dist_u2t, dist_t2t, dist_t2u):
    """
    Calculate the cost matrix for the given drone positions, velocities, grid IDs, and first/second IDs.

    Returns:
    np.array: The resulting cost matrix.
    """
    # Number of drones and nodes
    ego_num = len(robs_info)
    tar_num = len(tasks_pos)

    # Matrix dimensions (1 + 2*drone_num + grid_num)
    dimen = 1 + ego_num + tar_num + ego_num
    mat = np.zeros((dimen, dimen))

    # Virtual depot to drones
    for i in range(ego_num):
        mat[0, 1 + i] = 0
        mat[1 + i, 0] = 1000

    # Virtual depot to target nodes
    for i in range(tar_num):
        mat[0, 1 + ego_num + i] = 1000          # Virtual node does not go directly to target nodes
        mat[1 + ego_num + i, 0] = 1000          # Target nodes do not go directly to the virtual node

    # Costs between drones
    for i in range(ego_num):
        for j in range(ego_num):
            mat[1 + i, 1 + j] = 1000

    # Costs from egos to target nodes
    for i in range(ego_num):
        rob_id = int(robs_info[i].ta_pos_[2])
        for j in range(tar_num):
            tar_id = int(tasks_pos[j][2])
            cost = dist_u2t[rob_id][tar_id]
            # cost = l2norm(ego_positions[i], targets_pos[j])
            mat[1 + i, 1 + ego_num + j] = cost                  # From robot node to target node
            mat[1 + ego_num + j, 1 + i] = 1000                  # Target nodes do not go directly to robot nodes

    # Costs between target nodes
    for i in range(tar_num):
        tar_id_i = int(tasks_pos[i][2])
        for j in range(i + 1, tar_num):
            tar_id_j = int(tasks_pos[j][2])
            cost = dist_t2t[tar_id_i][tar_id_j]
            mat[1 + ego_num + i, 1 + ego_num + j] = cost
            mat[1 + ego_num + j, 1 + ego_num + i] = cost

    # From added nodes to virtual depot
    for i in range(ego_num):
        mat[1 + ego_num + tar_num + i, 0] = 0
        mat[0, 1 + ego_num + tar_num + i] = 1000  # Virtual node does not go directly to added nodes

    # Costs between added nodes and egos
    for i in range(ego_num):
        for j in range(ego_num):
            mat[1 + ego_num + tar_num + i, j + 1] = 1000     # Non-corresponding added nodes can go directly to robot nodes
            mat[j + 1, 1 + ego_num + tar_num + i] = 1000

    # Costs between added and target nodes
    for i in range(tar_num):
        tar_id = int(tasks_pos[i][2])
        for j in range(ego_num):
            rob_id = int(robs_info[j].ta_pos_[2])
            mat[1 + ego_num + i, ego_num + tar_num + j + 1] = dist_t2u[rob_id][tar_id]
            mat[ego_num + tar_num + j + 1, 1 + ego_num + i] = 1000

    # Costs between added nodes
    for i in range(ego_num):
        for j in range(ego_num):
            mat[1 + ego_num + tar_num + i, ego_num + tar_num + j + 1] = 1000
            mat[ego_num + tar_num + j + 1, 1 + ego_num + tar_num + i] = 1000

    # Diagonal elements (self-to-self costs)
    np.fill_diagonal(mat, 1000)

    return mat


def generate_cvrp_file(robs_info, task_num, mat, filename):
    robot_num = len(robs_info)
    dimension = mat.shape[0]
    capacity = robs_info[0].max_load_
    distance = robs_info[0].max_endurance_ * 1.5

    with open(filename, 'w') as f:
        f.write("NAME : adcvrp_test\n")
        f.write("TYPE : ADCVRP\n")
        f.write(f"DIMENSION : {dimension}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write(f"DISTANCE : {distance}\n")
        f.write(f"VEHICLES : {robot_num}\n")

        # Write the cost matrix
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(dimension):
            f.write(" ".join([str(int(1000 * mat[i, j])) for j in range(dimension)]) + "\n")

        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(robot_num):
            f.write(f"{i + 2} 0\n")
        for i in range(task_num):
            f.write(f"{i + 2 + robot_num} {1}\n")
        for i in range(robot_num):
            f.write(f"{robot_num + task_num + i + 2} 0\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("EOF\n")


def generate_lkh_parameter_file(filename, param_filename, output_filename):
    with open(param_filename, 'w') as f:
        f.write("SPECIAL\n")
        f.write(f"PROBLEM_FILE = {filename}\n")
        f.write(f"TOUR_FILE = {output_filename}\n")
        f.write("MAX_TRIALS = 1000\n")
        # f.write("MTSP_MIN_SIZE = 0\n")
        # f.write("POPULATION_SIZE = 10\n")
        f.write("RUNS = 1\n")
        f.write("SEED = 0\n")
        f.write("TRACE_LEVEL = 0\n")


def allocateTargets(robs_info, tasks_pos, scale, name=None):
    filename = "../../draw/resource/lkh_test.vrp"
    param_filename = "../../draw/resource/lkh_test.par"
    output_filename = "../../draw/resource/lkh_test.tour"

    task_num = len(tasks_pos)
    robot_num = len(robs_info)
    if name is not None:  # Use global planner to compute obstacle-avoiding paths and path costs
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)

    if os.path.exists(output_filename):
        try:
            os.remove(output_filename)
            print(f"Deleted existing tour file: {output_filename}")
        except Exception as e:
            print(f"Error deleting tour file: {e}")
            return

    # Create the cost matrix
    mat = getCostMatrix(robs_info, tasks_pos, dist_u2t, dist_t2t, dist_t2u)
    dimension = mat.shape[0]

    # Create the problem file
    generate_cvrp_file(robs_info, task_num, mat, filename)

    # Create the PAR file
    generate_lkh_parameter_file(filename, param_filename, output_filename)

    # Solve the problem using LKH via subprocess
    ts = time.time()
    while True:
        try:
            # Adjust the LKH path if necessary
            lkh_command = f"LKH {param_filename}"
            subprocess.run(lkh_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during LKH execution: {e}")
            return
        if os.path.exists(output_filename):
            break

    # Read results from the TOUR file
    with open(output_filename, 'r') as f:
        lines = f.readlines()

    ids = []
    in_tour_section = False         # Flag indicating whether we are inside TOUR_SECTION
    for line in lines:
        line = line.strip()
        if line == "TOUR_SECTION":
            in_tour_section = True
            continue                # Skip the TOUR_SECTION line
        if in_tour_section:
            idx = int(line)         # Convert node index to integer
            ids.append(idx - 1)     # Convert to 0-based index
            if line == "-1":        # Encountering -1 indicates end of the tour section
                break

    tour = []
    ego_id = 0
    tour_results = {}
    for idx in ids:
        if 0 <= idx < dimension:
            if idx == 0:
                continue
            tour.append(idx)
        elif idx < 0:
            tour_results[ego_id] = tour
        else:
            tour_results[ego_id] = tour
            tour = []
            ego_id += 1

    te = time.time()

    dist_costs = []
    for rob_id, tour in tour_results.items():
        tour_results[rob_id].insert(0, 0)
        tour_results[rob_id].append(0)
        dist_cost = 0
        for i in range(len(tour)):
            if i < len(tour) - 1:
                dist_cost += int(100 * mat[tour[i], tour[i + 1]])
        dist_costs.append(dist_cost)

    tour_results1 = {}
    for rob_id, tour in tour_results.items():
        tour_results[rob_id].pop(0)
        based0_id = tour_results[rob_id].pop(0) - 1
        tour_results[rob_id].pop()
        tour_results[rob_id].pop()
        tour_results1[based0_id] = [idx - robot_num - 1 for idx in tour_results[rob_id]]

    # Plot the routes
    # positions = [rob.ta_pos_ for rob in robs_info]
    # plot_routes(positions, tasks_pos, tour_results1)

    all_path = [[] for _ in range(robot_num)]
    dist_costs = []
    assigned_num = 0
    rewards = 0.0
    path_list = [[] for _ in range(robot_num)]

    for rob_id, tour in tour_results1.items():
        i_idx = int(robs_info[rob_id].ta_pos_[2])
        path = []
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

        all_path[rob_id] = path
        dist_costs.append(path_length(path))
        path_list[rob_id] = tour
        assigned_num += len(tour)

    return path_list, rewards, te - ts, dist_costs, all_path


def main():
    # Large blocks scenario
    envs = env.Env(envs_type[5])
    agents, task_area, tars_pos, scl = build_obj_pos_large(envs, ag_num=10, tar_num=100)
    name = '/large_u10t100.json'
    tar_ids, res, allocost, dists, all_paths = allocateTargets(agents, tars_pos, scl, name)

    print(dists)
    print("max and average travel distance: ", max(dists), sum(dists) / 1000)

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
    print("assigned_num", assigned_num)
    print("empty num: ", empty_num)

    robs_nest = agents[0].ending_area_
    robs, tasks = build_agents_and_tasks(agents, tars_pos, tar_ids, envs)

    plt_visulazation(all_paths, robs, tasks, envs.poly_obs, task_area, robs_nest)


if __name__ == "__main__":
    main()
