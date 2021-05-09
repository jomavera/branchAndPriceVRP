import re
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics.pairwise import euclidean_distances
from models import GreedySolution, SubProblem, MasterProblem
import time
from itertools import combinations
from utils import get_min_dist

def copy_model(MP_to_copy):

    MP_1 = MasterProblem(c, K, initAssingments, MP_to_copy.modelo)
    MP_1.update_model()
    MP_2 = MasterProblem(c, K, initAssingments, MP_to_copy.modelo)
    MP_2.update_model()

    return MP_1, MP_2   

def solve_price(obj_val, demands, capacity, distances, MasterProb):
    
    # MP_branch.modelo.Params.BestObjStop = obj_val
    
    branches = [MasterProb]
    print("# - Branchs to Explore:{}".format(len(branches)))
    while True:


        branched = False
        
        MP_branch = branches.pop(-1)
        MP_branch.RelaxOptimize()
        
        if MP_branch.relax_modelo.Status == 2:

            solution = MP_branch.getSolution()
            duals   = MP_branch.getDuals()
            print(duals)
            for i in solution:
                if i > 0.0001 and np.abs(i-1.0) > 0.0001:
                    print("#### Solution:{}".format(solution)+"....Branching")
                    MP_1, MP_2 = branch(n, demands, capacity, distances, duals, solution, MP_branch)
                    if MP_1 != None:
                        branches.append(MP_1)
                    if MP_2 != None:
                        branches.append(MP_2)
                    branched = True
                    print("# - Branchs to Explore:{}".format(len(branches)))
                    break
            
            if not branched:

                SP_branch = SubProblem(n, demands, capacity, distances, duals)
                SP_branch.build_model()

                SP_branch.optimize()
                print("------Status Code:{}".format(SP_branch.modelo.Status) )
                solution = SP_branch.getSolution()
                print("--------Found MP interger solution:{}".format(solution))
                print("--------SP Obj Val:{}".format(SP_branch.modelo.ObjVal))
            
                if SP_branch.modelo.ObjVal >= -0.0001: # break if no more columns

                    break

                newAssing = [ SP_branch.y[i].x for i in SP_branch.y ]	# new Assingment
                newColumn = gp.Column(newAssing, MP_branch.modelo.getConstrs())
                obj = get_min_dist(newAssing, distances)
                MP_branch.modelo.addVar(vtype=GRB.BINARY, obj=obj , column=newColumn )
                MP_branch.modelo.update()
                branches.append(MP_branch)
            

    return MP_branch


def branch(n, demands, C, distances, duals, solution_to_branch, MP_to_copy):

    SP_1 = SubProblem(n, demands, capacity, distances, duals)
    SP_2 = SubProblem(n, demands, capacity, distances, duals)
    SP_1.build_model()
    SP_2.build_model()
    
    frac_ixs = []

    for ix, val in enumerate(solution_to_branch):
        if val > 0.0 and val < 1.0:
            frac_ixs.append(ix)

    A_mp  = MP_to_copy.modelo.getA().toarray()[:len(MP.locations_index),:]
    locations_index = list(MP.locations_index)

    for comb in combinations(frac_ixs,2):
        for i in locations_index:
            locations_prime = [x for x in locations_index if x != i]
            for j in locations_prime:
                if A_mp[i,comb[0]] + A_mp[i,comb[1]] == 2 and  A_mp[j,comb[0]] + A_mp[j,comb[1]] == 2:
                    SP_1.modelo.addConstr(SP_1.y[i] + SP_1.y[j] >= 1 )
                if A_mp[i,comb[0]] + A_mp[j,comb[1]] == 2 and A_mp[i,comb[1]] + A_mp[j,comb[0]] == 0:
                    SP_2.modelo.addConstr(SP_2.y[i] + SP_2.y[j] <= 1 )
    
    MP_1, MP_2 = copy_model(MP_to_copy)
    SP_1.modelo.update()
    SP_1.optimize()

    if SP_1.modelo.Status == 2:
        newAssing = [ SP_1.y[i].x for i in SP_1.y ] + [ 1 ]	# new Assingment
        newColumn = gp.Column(newAssing, MP_1.modelo.getConstrs())
        MP_1.modelo.addVar(vtype=GRB.BINARY, obj=np.matmul(delay_i.reshape(1,-1),np.array(newAssing[:-1]).reshape(-1,1)) + SP_1.l_k.X , column=newColumn )
        MP_1.modelo.update()

    SP_2.modelo.update()
    SP_2.optimize()

    if SP_2.modelo.Status == 2:
        newAssing = [ SP_2.y[i].x for i in SP_2.y ] + [ 1 ]	# new Assingment
        newColumn = gp.Column(newAssing, MP_2.modelo.getConstrs())
        MP_2.modelo.addVar(vtype=GRB.BINARY, obj=np.matmul(delay_i.reshape(1,-1),np.array(newAssing[:-1]).reshape(-1,1)) + SP_2.l_k.X , column=newColumn )
        MP_2.modelo.update()

    if SP_1.modelo.Status == 2 and SP_2.modelo.Status == 2:
        return MP_1, MP_2
    elif SP_1.modelo.Status == 2:
        return MP_1, None
    elif SP_2.modelo.Status == 2:
        return None, MP_2

coordinates_list = []
demands = []
with open('instances/A-VRP/A-n32-k5.vrp') as f:
    lines = f.readlines()
    for line in lines:
        sol_C = re.findall(r"CAPACITY : \d*",line)
        if sol_C != []:
            list_str = sol_C[0].split(' ')
            capacity = int(list_str[2])
        sol_coord = re.findall(r"\d{1,2}\s\d{1,2}\s\d{1,2}",line)
        sol_demands = re.findall(r"\d{1,2}\s\d{1,2}\s\s",line)
        if sol_coord != []:
            coords = sol_coord[0].split(' ')
            coordinates_list.append( [int(coords [1]), int(coords [2])] )
        if sol_demands != []:
            demand_str = sol_demands[0].split(' ')
            demands.append( int(demand_str[1]) )

coordinates = np.stack(coordinates_list, axis=0)
distances = euclidean_distances(coordinates)
n = coordinates.shape[0]
EPS = 0.001
K = 5

start_time = time.time()
#------ Encontrar assingments iniciales factibles ------#
greedy_problem = GreedySolution(n, capacity, distances, demands)
init_assingments, init_routes = greedy_problem.generate_init_solution()
#------------------------------------------------------#

c  = []
for route in init_routes:
    c_temp = 0
    for ix, val in enumerate(route):

        if ix < len(route)-1:
            c_temp += distances[val, route[ix+1]]
    c.append(c_temp)

MP = MasterProblem(c, K, init_assingments)
MP.build_model()


obj_val = 100000

iteraciones = 0

MP_sol = solve_price(obj_val, demands, capacity, distances, MP)

end_time = time.time()
print("RUN TIME:{} seg.".format(end_time- start_time))
# print(np.sum(MP..getA().toarray()[:n,:7],axis=1))
# print(np.sum(best_model.getA().toarray()[:n,:7],axis=0))
print(MP_sol.relax_modelo.getAttr('X'))