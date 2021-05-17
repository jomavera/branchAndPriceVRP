import re
from BnP_v2 import branch_n_price
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time
from utils import GreedySolution
from models import MasterProblem

# ------------ Read problem instance ------------#
coordinates_list = []
demands = []
with open("instances/A-VRP/A-n32-k5.vrp") as f:
    lines = f.readlines()
    for line in lines:
        sol_C = re.findall(r"CAPACITY : \d*", line)
        if sol_C != []:
            list_str = sol_C[0].split(" ")
            capacity = int(list_str[2])
        sol_coord = re.findall(r"\d{1,2}\s\d{1,2}\s\d{1,2}", line)
        sol_demands = re.findall(r"\d{1,2}\s\d{1,2}\s\s", line)
        if sol_coord != []:
            coords = sol_coord[0].split(" ")
            coordinates_list.append([int(coords[1]), int(coords[2])])
        if sol_demands != []:
            demand_str = sol_demands[0].split(" ")
            demands.append(int(demand_str[1]))

coordinates = np.stack(coordinates_list, axis=0)
distances = euclidean_distances(coordinates)
n = coordinates.shape[0]
EPS = 0.001

start_time = time.time()

# ------------ Find initial feasible routes ------------#
greedy_problem = GreedySolution(n, capacity, distances, demands)
init_assingments, init_routes = greedy_problem.generate_init_solution()


# ----------- Get total distance of initial routes -----#
c = []
for route in init_routes:
    c_temp = 0
    for ix, val in enumerate(route):

        if ix < len(route) - 1:
            c_temp += distances[val, route[ix + 1]]
    c.append(c_temp)


# ----------- Create initial Master Problem -------#
MP = MasterProblem(c, init_assingments)
MP.build_model()

# ------------- Run Branch and Price ----------#
MP_sol = branch_n_price(n, c, init_assingments, demands, capacity, distances, MP)


end_time = time.time()
print("RUN TIME:{} seg.".format(end_time - start_time))
MP_sol.RelaxOptimize()
print(MP_sol.relax_modelo.getAttr("X"))
print(MP_sol.relax_modelo.ObjVal)
