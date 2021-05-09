from copy import copy

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