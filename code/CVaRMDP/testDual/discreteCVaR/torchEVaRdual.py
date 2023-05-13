import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch

import distDF as ddf
import riskMeasure as rm

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getEVaRs(list_of_distributions, Lambda):
    return [ddf.EVaR(d, Lambda) for d in list_of_distributions]


def discreteEVaR(Xi, P, V, Lam):
    z = Xi[0] * torch.log(Xi[0] / P[0]) + Xi[1] * torch.log(Xi[1] / P[1])
    z[torch.isnan(z)] = float("inf")

    def par_solve(l):
        if l <= 0:
            return min([v.min() for v, p in zip(V, P) if p > 0])

        budget = -torch.log(l) - z
        valid = 0 <= budget

        if not torch.any(valid):
            return sum([v[torch.argmin(torch.abs(Lam - p))] * p for p, v in zip(P, V)])

        validIndex = torch.where(valid)[0]

        def min_c(index):
            remainBudget = budget[index] + Xi[0][index] * torch.log(Lam)
            c0valid = 0 <= remainBudget
            c0 = torch.where(c0valid)[0]
            c1 = torch.searchsorted(
                Lam, torch.exp(-remainBudget[c0valid] / (Xi[1][index])), side="left"
            )
            return ((Xi[0][index] * V[0][c0]) + (Xi[1][index] * V[1][c1])).min()

        return (torch.tensor([min_c(i) for i in validIndex])).min()

    return torch.tensor([par_solve(l) for l in Lam])


# sol = []
# for l in Lam:
#     print(l)
#     sol.append(par_solve(l))
Lambda = torch.linspace(0, 1, 1001)
xi = torch.linspace(0, 1, 1001)

s0a0 = ddf.distribution(X=np.array([-300, 300]), p=np.array([0.25, 0.75]))
# Simple case 1
s0a0s0 = torch.full_like(Lambda, s0a0.X[0])
s0a0s1 = torch.full_like(Lambda, s0a0.X[1])
# Get s1 EVaR Values for each action and get the max of the EVaR values
EVaRs0a0 = ddf.EVaR(s0a0, Lambda.numpy())

t1 = time.time()
s0a0dEVaR = discreteEVaR([xi, 1 - xi], s0a0.p.values, [s0a0s0, s0a0s1], Lambda)
t2 = time.time()
print(t2 - t1)
# Compare difference
(torch.abs(s0a0dEVaR - torch.from_numpy(EVaRs0a0))).max()
