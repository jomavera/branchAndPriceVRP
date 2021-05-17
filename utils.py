from copy import copy
import numpy as np
from models import SubProblem, MasterProblem

def get_min_dist(route, distances):
    index_loc = 0
    route_temp = copy(route)
    total_dist = 0
    while sum(route_temp) != 0:
        distances_temp = distances[index_loc, :]
        min_dist = 1e3
        min_index = None
        for ix, val in enumerate(route_temp):
            dist_node = distances_temp[ix]
            if dist_node < min_dist and (ix != index_loc or ix != 0) and val == 1:
                min_dist = dist_node
                min_index = ix
                
        if min_index != None:
            route_temp[min_index] = 0
            index_loc = min_index
            total_dist += min_dist
        if sum(route_temp) == 0:
            total_dist += distances[min_index, 0]
    
    return total_dist

def copy_models(coeff, assingments, MP_to_copy):

    MP_1 = MasterProblem(coeff, assingments, MP_to_copy.modelo)
    MP_1.update_model()
    MP_2 = MasterProblem(coeff, assingments, MP_to_copy.modelo)
    MP_2.update_model()

    return MP_1, MP_2

def copy_model(c, init_assingments, MP_to_copy):
    MP_copy = MasterProblem(c, init_assingments, MP_to_copy.modelo)
    MP_copy.update_model()

    return MP_copy

class GreedySolution:
    """
    Model to find initial assingments/routes
    """

    def __init__(self, n, capacity, distances, demands):

        self.n = n
        self.C = capacity
        self.distances = distances  # dim=[locations, locations]
        self.D = demands  # dim=[locations]

        self.locations_index = np.arange(n, dtype=int)

    def generate_init_solution(self):
        routes = []
        route_orders = []
        D_temp = copy(self.D)

        while np.sum(D_temp) != 0:
            index_loc = 0  # Start at depot
            route = [0 for _ in range(self.n)]  # Empty assingtments in route
            route_order = [0]
            C_temp = copy(self.C)  # Full capacity
            load = 0
            while not np.greater_equal(load, self.C):

                distances_loc = self.distances[index_loc, :]
                min_dist = 1e3
                min_index = None
                for index_node, dist_node in enumerate(distances_loc):
                    if (
                        dist_node < min_dist
                        and D_temp[index_node] != 0
                        and D_temp[index_node] <= C_temp
                        and (index_node != index_loc or index_node != 0)
                    ):

                        min_dist = dist_node
                        min_index = index_node

                if min_index != None:
                    route[min_index] = 1
                    route_order.append(min_index)
                    index_loc = min_index
                    C_temp -= D_temp[min_index]
                    load += D_temp[min_index]
                    D_temp[min_index] = 0

                if (
                    not all(
                        np.greater(
                            C_temp,
                            D_temp,
                        )
                    )
                    or np.sum(D_temp) == 0
                ):
                    routes.append(route)
                    route_order.append(0)
                    route_orders.append(route_order)
                    break

        return np.stack(routes, axis=1), route_orders