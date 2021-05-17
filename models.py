import numpy as np
import copy
import gurobipy as gp
from gurobipy import GRB


class SubProblem:
    """
    Model to find  new assignment (column for Master Problem) according to duals
    """

    def __init__(self, n, demands, capacity, distances, duals, modelo=None):

        self.d = demands[1:]
        self.C = capacity
        self.distances = distances
        self.duals = duals
        self.locations_index = np.arange(n-1, dtype=int)
        self.distances = distances

        if modelo == None:
            self.modelo = gp.Model("subProblem")
        else:
            self.modelo = modelo.copy()

    def update_model(self):

        self.y = {}
        for var in self.modelo.getVars():
            if var.VarName in "if_locations_is_served":
                ix = var.index
                varName = "locations[{}]".format(ix)
                self.y[ix] = self.modelo.getVarByName(varName)

        self.modelo.update()

    def build_model(self):

        self.y = self.modelo.addVars(
            self.locations_index, vtype=GRB.BINARY, name="if_locations_is_served"
        )  # dim=[locations]

        self.modelo.addConstr(
            gp.quicksum([self.d[i] * self.y[i] for i in self.locations_index])
            <= self.C,
            "capacity_constraint",
        )

        self.modelo.setObjective(
            gp.quicksum([-self.duals[i] * self.y[i] for i in self.locations_index]),
            sense=GRB.MINIMIZE,
        )

        self.modelo.update()
        self.modelo.Params.OutputFlag = 0

    def optimize(self):
        self.modelo.optimize()
        # self.modelo.Params.MIPGap = 0.5

    def getSolution(self):

        return self.modelo.getAttr("X")


class MasterProblem:
    """
    Model to find duals (set covering problem)
    """

    def __init__(self, c, routes, modelo=None):

        self.c = c  # Cost for each route dim = [routes]
        self.routes = routes  # dim = [locations, routes]
        self.routes_index = np.arange(routes.shape[1])
        self.locations_index = np.arange(1, routes.shape[0])

        if modelo == None:
            self.modelo = gp.Model("masterProblem")
        else:
            self.modelo = modelo.copy()

    def update_model(self):

        self.v = {}
        for var in self.modelo.getVars():
            ix = var.index
            varName = var.VarName
            self.v[ix] = self.modelo.getVarByName(varName)

        self.modelo.update()

    def build_model(self):

        self.v = self.modelo.addVars(self.routes_index, vtype=GRB.BINARY, name="routes")
        self.modelo.addConstrs(
            (
                gp.quicksum([self.routes[i, j] * self.v[j] for j in self.routes_index])
                >= 1
                for i in self.locations_index
            ),
            name="locations_must_be_served",
        )

        self.modelo.setObjective(
            gp.quicksum([np.sum(self.c[j]) * self.v[j] for j in self.routes_index]),
            sense=GRB.MINIMIZE,
        )  # Minimize total distance of all routes

        self.modelo.Params.OutputFlag = 0
        self.modelo.update()

    def RelaxOptimize(self):

        self.relax_modelo = self.modelo.relax()
        self.relax_modelo.optimize()


    def getDuals(self):

        pi = []
        for constr in self.relax_modelo.getConstrs():
            if (
                "locations_must_be_served" in constr.ConstrName
                or "num_vehicles" in constr.ConstrName
            ):
                pi.append(constr.Pi)

        return pi

    def getSolution(self):

        return self.relax_modelo.getAttr("X")

    def getCosts(self):

        return self.relax_modelo.getAttr("Obj")