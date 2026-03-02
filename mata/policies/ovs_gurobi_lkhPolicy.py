import time
import subprocess
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from mamp.envs import env
from mamp.tools.utils import l2norm, path_length
from draw.plt2d import plt_visulazation
from mamp.configs.config import envs_type
from mata.ta_config.general_config import build_agents_and_tasks, read_json
from mata.ta_config.general_config import build_obj_pos_large, build_cost_path_matrix


# Compute the distance matrix
def calculate_distance_matrix(robs_info, tasks_pos, dist_u2t):
    distance_matrix = [[0] * len(tasks_pos) for _ in range(len(robs_info))]
    for i in range(len(robs_info)):
        i_idx = int(robs_info[i].ta_pos_[2])
        for j in range(len(tasks_pos)):
            j_idx = int(tasks_pos[j][2])
            distance_matrix[i][j] = dist_u2t[i_idx][j_idx]
    return distance_matrix


def solve_minimax_assignment(robs_info, tasks_pos, dist_u2t):
    num_robots = len(robs_info)
    num_targets = len(tasks_pos)
    distance_matrix = calculate_distance_matrix(robs_info, tasks_pos, dist_u2t)

    # Create model
    model = gp.Model('minimax_assignment')

    # Define variables
    x = model.addVars(num_robots, num_targets, vtype=GRB.BINARY, name='x')  # Assignment variables
    z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z')  # Maximum distance variable

    # Add constraints
    for j in range(num_targets):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_robots)) == 1)  # Each target is assigned to one robot

    # Add maximum distance constraints
    for i in range(num_robots):
        for j in range(num_targets):
            model.addConstr(distance_matrix[i][j] * x[i, j] <= z)  # z is the maximum among assigned distances

    # Define objective function: minimize maximum distance
    model.setObjective(z, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        print('Optimal solution:')
        assignments = []
        assignments1 = []
        for i in range(num_robots):
            assignment = []
            for j in range(num_targets):
                if x[i, j].x > 0.5:  # Binary variable check
                    assignment.append(j)
                    assignments1.append((i, j))
            assignments.append(assignment)
        print(f'Minimized maximum distance: {z.x}')
        return assignments, z.x
    else:
        print('No feasible solution.')
        return None, None


# Visualize assignment results
def plot_assignments(robot_pos, targets, assignments):
    plt.figure(figsize=(15, 12))

    # Plot robot positions
    for i, (x, y) in enumerate(robot_pos):
        plt.scatter(x, y, color='blue', label='Robots' if i == 0 else "", s=100)
        plt.text(x, y, f'R{i}', fontsize=12, ha='right')

    # Plot target positions
    for j, (x, y) in enumerate(targets):
        plt.scatter(x, y, color='red', label='Targets' if j == 0 else "", s=100)
        plt.text(x, y, f'T{j}', fontsize=12, ha='left')

    # Plot assignment lines
    for (i, j) in assignments:
        plt.plot([robot_pos[i][0], targets[j][0]],
                 [robot_pos[i][1], targets[j][1]],
                 color='green', linestyle='-', linewidth=1)

    plt.title('Robot-Target Assignment')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_tsp_file(rob, assignment, tasks_pos, dist_u2t, dist_t2t, filename):
    """
    Generate a TSP file based on an explicit distance matrix
    """
    num_targets = len(assignment)
    num_nodes = 1 + num_targets

    # Compute distance matrix (scaled by 1000 and converted to integer)
    distance_matrix = []
    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            if i > 0 and j > 0:
                i_idx = int(tasks_pos[assignment[i - 1]][2])
                j_idx = int(tasks_pos[assignment[j - 1]][2])
                dist = dist_t2t[i_idx][j_idx]
            elif i == 0 and j > 0:
                i_idx = int(rob.ta_pos_[2])
                j_idx = int(tasks_pos[assignment[j - 1]][2])
                dist = dist_u2t[i_idx][j_idx]
            elif j == 0 and i > 0:
                i_idx = int(tasks_pos[assignment[i - 1]][2])
                j_idx = int(rob.ta_pos_[2])
                dist = dist_u2t[j_idx][i_idx]
            else:
                dist = 0.0
            dist_int = int(round(dist * 1000))
            row.append(str(dist_int))
        distance_matrix.append(row)

    with open(filename, "w") as f:
        f.write("NAME : test1\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for row in distance_matrix:
            f.write(" ".join(row) + "\n")

        f.write("EOF\n")


def generate_lkh_parameter_file(tsp_filename, param_filename, output_filename):
    """
    Generate parameter file for LKH solver
    """
    param_data = f"""
    PROBLEM_FILE = {tsp_filename}
    OUTPUT_TOUR_FILE = {output_filename}
    RUNS = 1
    """
    with open(param_filename, "w") as f:
        f.write(param_data.strip())


def solve_with_lkh(param_filename):
    """
    Solve the TSP using LKH and return whether successful
    """
    command = f"LKH {param_filename}"
    result = subprocess.run(command, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error executing LKH: {result.stderr.decode()}")
        return None

    return True


def parse_solution_file(solution_file):
    """
    Parse LKH solution file and extract optimal tour
    """
    indices = []
    with open(solution_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.strip() == "TOUR_SECTION":
                break

        for line in lines:
            if line.strip() == "-1":
                break
            try:
                indices.append(int(line.strip()) - 2)
            except ValueError:
                continue

    indices.pop(0)
    return indices


def allocateTargets(robs_info, tasks_pos, scale, name=None):
    filename = "../../draw/resource/lkh_test.tsp"
    param_filename = "../../draw/resource/lkh_test.par"
    output_filename = "../../draw/resource/lkh_test.tour"

    task_num = len(tasks_pos)
    robot_num = len(robs_info)

    if name is not None:  # Use global planner to compute obstacle-avoiding paths and path costs
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = read_json(name)
    else:
        dist_u2t, path_u2t, dist_t2t, path_t2t, dist_t2u, turning = build_cost_path_matrix(robs_info, tasks_pos, scale)

    ts = time.time()
    assignments, max_distance = solve_minimax_assignment(robs_info, tasks_pos, dist_u2t)
    t_gurobi1 = time.time()
    t_gurobi = t_gurobi1 - ts
    print('gurobi time', t_gurobi)

    t_lkh1 = time.time()
    routes = []
    for i in range(len(assignments)):
        if len(assignments[i]) > 1:
            generate_tsp_file(robs_info[i], assignments[i], tasks_pos,
                              dist_u2t, dist_t2t, filename)
            generate_lkh_parameter_file(filename, param_filename, output_filename)
            if not solve_with_lkh(param_filename):
                print("Failed to solve TSP.")
                break
            indices = parse_solution_file(output_filename)
        elif len(assignments[i]) == 1:
            indices = [0]
        else:
            indices = []

        route = []
        for index in indices:
            route.append(assignments[i][index])
        routes.append(route)

    te = time.time()
    t_lkh = (te - t_lkh1) / len(assignments)
    print('lkh time for single robot', t_lkh)

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
            if j == 0:
                path += path_u2t[i_idx][j_idx]
            else:
                j_idx_last = int(tasks_pos[tour[j - 1]][2])
                path.pop()
                path += path_t2t[j_idx_last][j_idx]

            if j == len(tour) - 1:
                path.pop()
                path_t2e = path_u2t[i_idx][j_idx]
                path_t2e.reverse()
                path += path_t2e

        all_path[i] = path
        dist_costs.append(path_length(path))
        path_list[i] = tour
        assigned_num += len(tour)

    return path_list, rewards, t_gurobi + t_lkh, dist_costs, all_path


def main(envs, agents, task_area, tars_pos, scl):

    name = '/large_u10t100.json'
    tar_ids, res, allocost, dists, all_paths = allocateTargets(agents, tars_pos, scl, name)

    print("max travel distance: ", max(dists), dists)

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
    return max(dists), allocost


# Main function
if __name__ == '__main__':
    envs = env.Env(envs_type[5])
    agents, task_area, tars_pos, scl = build_obj_pos_large(envs, ag_num=10, tar_num=100)
    main(envs, agents, task_area, tars_pos, scl)
