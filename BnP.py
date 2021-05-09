import re
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics.pairwise import euclidean_distances
from models import GreedySolution, SubProblem, MasterProblem
import time
from itertools import combinations
from utils import get_min_dist
from bst import Tree


def copy_models(MP_to_copy):

    MP_1 = MasterProblem(c, init_assingments, MP_to_copy.modelo)
    MP_1.update_model()
    MP_2 = MasterProblem(c, init_assingments, MP_to_copy.modelo)
    MP_2.update_model()

    return MP_1, MP_2


def solve_price(obj_val, demands, capacity, distances, MasterProb):

    # MP_branch.modelo.Params.BestObjStop = obj_val
    bst_temp = Tree()
    MasterProb.RelaxOptimize()
    bst_temp.insert(MasterProb.relax_modelo.ObjVal, MasterProb)
    while True:
        solution = MasterProb.getSolution()
        duals = MasterProb.getDuals()
        print(MasterProb.relax_modelo.ObjVal)

        branched = False

        for i in solution:
            if i > 0.01 and np.abs(i - 1.0) > 0.01:
                print("#--#--#--# Not integer solution  ........Branching")
                MP_1, MP_2 = branch(
                    n,
                    demands,
                    capacity,
                    distances,
                    duals,
                    solution,
                    MasterProb,
                    bst_temp,
                )
                if MP_1 != None:
                    MP_1.RelaxOptimize()
                    bst_temp.insert(MP_1.relax_modelo.ObjVal, MP_1)
                if MP_2 != None:
                    MP_2.RelaxOptimize()
                    bst_temp.insert(MP_2.relax_modelo.ObjVal, MP_2)
                if MP_1 == None and MP_2 == None:
                    print("Not branched")
                else:
                    branched = True
                break

        if not branched:

            SP_branch = SubProblem(n, demands, capacity, distances, duals)
            SP_branch.build_model()

            SP_branch.optimize()

            print("--------SP Obj Val:{}".format(SP_branch.modelo.ObjVal))

            if SP_branch.modelo.ObjVal >= -0.0001:  # break if no more columns

                break

            newAssing = [SP_branch.y[i].x for i in SP_branch.y]  # new route
            newColumn = gp.Column(newAssing, MasterProb.modelo.getConstrs())
            obj = get_min_dist(newAssing, distances)  # Cost of new route
            MasterProb.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
            MasterProb.modelo.update()
            MasterProb.RelaxOptimize()
            bst_temp.insert(MasterProb.relax_modelo.ObjVal, MasterProb)

        else:
            node = bst_temp.find_min()
            MasterProb = node.model
            matriz = MasterProb.modelo.getA().toarray()[: len(MP.locations_index), :]

    return MP_branch


def branch(
    n, demands, C, distances, duals, solution_to_branch, MP_to_copy, binary_tree
):

    SP_1 = SubProblem(n, demands, capacity, distances, duals)
    SP_2 = SubProblem(n, demands, capacity, distances, duals)
    SP_1.build_model()
    SP_2.build_model()

    frac_ixs = []

    for ix, val in enumerate(solution_to_branch):
        if val > 0.0 and val < 1.0:
            frac_ixs.append(ix)

    A_mp = MP_to_copy.modelo.getA().toarray()[: len(MP_to_copy.locations_index), :]
    locations_index = list(MP_to_copy.locations_index)

    for comb in combinations(frac_ixs, 2):

        s1_and_s2 = [
            True
            if (
                A_mp[i, comb[0]] > 0
                and A_mp[i, comb[0]] < 1
                and A_mp[i, comb[1]] > 0
                and A_mp[i, comb[1]] < 1
            )
            else False
            for i in range(len(MP.locations_index))
        ]
        s1_not_s2 = [
            True
            if (A_mp[i, comb[0]] > 0 and A_mp[i, comb[0]] < 1 and A_mp[i, comb[1]] == 0)
            else False
            for i in range(len(MP.locations_index))
        ]

        for i in locations_index:
            locations_prime = [x for x in locations_index if x != i]
            for j in locations_prime:

                if s1_and_s2[i - 1] and s1_not_s2[j - 1]:
                    SP_1.modelo.addConstr(SP_1.y[i - 1] + SP_1.y[j - 1] >= 1)
                    SP_2.modelo.addConstr(SP_2.y[i - 1] + SP_2.y[j - 1] <= 1)

    MP_1, MP_2 = copy_models(MP_to_copy)

    SP_1.modelo.update()
    SP_1.optimize()
    if SP_1.modelo.Status == 2:
        newAssing = [SP_1.y[i].x for i in SP_1.y]  # new Assingment
        newColumn = gp.Column(newAssing, MP_1.modelo.getConstrs())
        obj = get_min_dist(newAssing, distances)  # Cost of new route
        MP_1.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
        MP_1.modelo.update()

    SP_2.modelo.update()
    SP_2.optimize()
    if SP_2.modelo.Status == 2:
        matriz = MP_2.modelo.getA().toarray()[: len(MP.locations_index), :]
        newAssing = [SP_2.y[i].x for i in SP_2.y]  # new Assingment
        newColumn = gp.Column(newAssing, MP_2.modelo.getConstrs())
        obj = get_min_dist(newAssing, distances)  # Cost of new route
        MP_2.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
        MP_2.modelo.update()

    if SP_1.modelo.Status == 2 and SP_2.modelo.Status == 2:
        return MP_1, MP_2
    elif SP_1.modelo.Status == 2:
        return MP_1, None
    elif SP_2.modelo.Status == 2:
        return None, MP_2
    else:
        return None, None


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
K = 5

start_time = time.time()
# ------ Encontrar assingments iniciales factibles ------#
greedy_problem = GreedySolution(n, capacity, distances, demands)
init_assingments, init_routes = greedy_problem.generate_init_solution()
# ------------------------------------------------------#

c = []
for route in init_routes:
    c_temp = 0
    for ix, val in enumerate(route):

        if ix < len(route) - 1:
            c_temp += distances[val, route[ix + 1]]
    c.append(c_temp)

MP = MasterProblem(c, init_assingments)
MP.build_model()


obj_val = 100000

iteraciones = 0

MP_sol = solve_price(obj_val, demands, capacity, distances, MP)

end_time = time.time()
print("RUN TIME:{} seg.".format(end_time - start_time))
# print(np.sum(MP..getA().toarray()[:n,:7],axis=1))
# print(np.sum(best_model.getA().toarray()[:n,:7],axis=0))
print(MP_sol.relax_modelo.getAttr("X"))
