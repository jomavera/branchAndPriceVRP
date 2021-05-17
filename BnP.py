import numpy as np
import gurobipy as gp
from gurobipy import GRB
from models import SubProblem, MasterProblem
from itertools import combinations
from utils import get_min_dist, copy_model, copy_models


def branch_n_price(n, c, init_assignments, demands, capacity, distances, MasterProb):

    queue = [ MasterProb ]

    best_obj = 1e3
    nodes_explored = 0
    best_model = None
    
    while len(queue) > 0:
        MP_branch = queue.pop()
        nodes_explored += 1
        MP_branch.RelaxOptimize()
        solution = MP_branch.getSolution()
        duals = MP_branch.getDuals()
        obj_val = MP_branch.relax_modelo.ObjVal
        branch_cost = MP_branch.getCosts()
        branch_routes = MP_branch.modelo.getA().toarray()
        sol_is_int = all([float(round(s,4)).is_integer() for s in solution ])
        # sol_is_int = all([False if i > 0.3 and np.abs(i - 1.0) > 0.3 else True for i in solution ])
        if obj_val < best_obj:
            print(f"Best Obj: {obj_val}")
            print(f"Nodes explored: {nodes_explored}")
            best_obj = obj_val
            if sol_is_int:
                print(f"best sol: {solution}")
                best_model = copy_model(branch_cost, branch_routes, MP_branch)

        # --- # --- # Column generation # --- # --- #
        new_MP = column_generation(n, demands, capacity, distances, duals, MP_branch)

        if new_MP != None:
            new_MP.RelaxOptimize()
            branch_cost = new_MP.getCosts()
            branch_routes = new_MP.modelo.getA().toarray()
            queue.insert(0, copy_model(branch_cost, branch_routes, new_MP))
        
        else:
            # --- # If stopped col generation then branch if solution is not integer # --- #
            if not sol_is_int:
                mp1_obj = None
                mp2_obj = None
                print("#--#--#--# Not integer solution  ........Branching")
                MP_1, MP_2 = branch(
                    branch_cost, branch_routes, n, demands, capacity, distances, duals, solution, MP_branch
                )
                if MP_1 != None:
                    MP_1.RelaxOptimize()
                    # mp1_obj = MP_1.relax_modelo.ObjVal
                    mp1_cost = MP_1.getCosts()
                    mp1_routes = MP_1.modelo.getA().toarray()
                    queue.insert(0, copy_model(mp1_cost, mp1_routes, MP_1) )
                if MP_2 != None:
                    MP_2.RelaxOptimize()
                    # mp2_obj = MP_2.relax_modelo.ObjVal
                    mp2_cost = MP_2.getCosts()
                    mp2_routes = MP_2.modelo.getA().toarray()
                    queue.insert(0, copy_model(mp2_cost, mp2_routes, MP_2))

                if MP_1 == None and MP_2 == None:
                    print("Not branched")

    return best_model


def branch(branch_cost, branch_routes, n, demands, capacity, distances, duals, solution_to_branch, MP_to_copy):

    SP_1 = SubProblem(n, demands, capacity, distances, duals)
    SP_2 = SubProblem(n, demands, capacity, distances, duals)
    SP_1.build_model()
    SP_2.build_model()

    frac_ixs = []

    for ix, val in enumerate(solution_to_branch):
        if val > 0.0 and val < 1.0:
            frac_ixs.append(ix)

    A_mp = MP_to_copy.modelo.getA().toarray()

    locations_index = list(MP_to_copy.locations_index)

    for comb in combinations(frac_ixs, 2):

        s1_and_s2 = [
            True
            if (
                A_mp[i-1, comb[0]] == 1
                and A_mp[i-1, comb[1]] == 1
            )
            else False
            for i in range(len(MP_to_copy.locations_index))
        ]
        s1_not_s2 = [
            True
            if (A_mp[i-1, comb[0]] == 1 and A_mp[i-1, comb[1]] == 0)
            else False
            for i in range(len(MP_to_copy.locations_index))
        ]

        add_constr = False
        for i in locations_index:
            locations_prime = [x for x in locations_index if x != i]
            for j in locations_prime:

                # if (s1_and_s2[i - 1] and s1_not_s2[j - 1]) or (
                #     s1_and_s2[j - 1] and s1_not_s2[i - 1]
                # ):
                #     SP_1.modelo.addConstr(SP_1.y[i - 1] + SP_1.y[j - 1] == 2)
                #     SP_2.modelo.addConstr(SP_2.y[i - 1] + SP_2.y[j - 1] <= 1)
                #     add_constr = True
                #     break
                if (s1_and_s2[i - 1] and s1_not_s2[j - 1]):
                    SP_1.modelo.addConstr(SP_1.y[i - 1] == 1.0)
                    SP_1.modelo.addConstr(SP_1.y[j - 1] == 1.0)
                    SP_2.modelo.addConstr(SP_2.y[i - 1] == 1.0)
                    SP_2.modelo.addConstr(SP_2.y[j - 1] == 0.0)
                    add_constr = True
                    break
            
            if add_constr:
                break
        if add_constr:
            break

    MP_1, MP_2 = copy_models(branch_cost, branch_routes, MP_to_copy)

    return_MP_1 = False
    return_MP_2 = False

    SP_1.modelo.update()
    SP_1.optimize()
    if SP_1.modelo.Status == 2:
        
        newAssing = [SP_1.y[i].x for i in SP_1.y]  # new Assingment
        obj = get_min_dist(newAssing, distances)  # Cost of new route

        if obj + SP_1.modelo.ObjVal < 0.0:
            newColumn = gp.Column(newAssing, MP_1.modelo.getConstrs())
            MP_1.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
            MP_1.modelo.update()
            return_MP_1 = True

    SP_2.modelo.update()
    SP_2.optimize()
    if SP_2.modelo.Status == 2:

        newAssing = [SP_2.y[i].x for i in SP_2.y]  # new Assingment
        obj = get_min_dist(newAssing, distances)  # Cost of new route

        if obj + SP_2.modelo.ObjVal < 0.0:

            newColumn = gp.Column(newAssing, MP_2.modelo.getConstrs())
            MP_2.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
            MP_2.modelo.update()
            return_MP_2 = True

    if return_MP_1 and return_MP_2:
        return MP_1, MP_2
    elif return_MP_1:
        return MP_1, None
    elif return_MP_2:
        return None, MP_2
    else:
        return None, None

def column_generation(n, demands, capacity, distances, duals, MP_branch):
    
    SP_branch = SubProblem(n, demands, capacity, distances, duals)
    SP_branch.build_model()

    SP_branch.optimize()

    new_MP = None

    newAssing = [SP_branch.y[i].x for i in SP_branch.y]  # new route
    obj = get_min_dist(newAssing, distances)  # Cost of new route

    if obj + SP_branch.modelo.ObjVal < 0.0: 
        newColumn = gp.Column(newAssing, MP_branch.modelo.getConstrs())
        MP_branch.modelo.addVar(vtype=GRB.BINARY, obj=obj, column=newColumn)
        MP_branch.modelo.update()
        MP_branch.RelaxOptimize()
        best_cost = MP_branch.getCosts()
        routes = MP_branch.modelo.getA().toarray()

        new_MP = copy_model(best_cost, routes, MP_branch)
    
    return new_MP
        
