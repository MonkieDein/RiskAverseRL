from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

import distDF as ddf
import riskMeasure as rm


def getEVaRs(list_of_distributions, Lambda):
    return [ddf.EVaR(d, Lambda) for d in list_of_distributions]


def discreteEVaR(Xi, P, V, Lam):
    z = np.sum([xi * np.log(xi / p) for xi, p in zip(Xi, P)], axis=0)
    np.nan_to_num(z, copy=False, nan=np.inf)

    def par_solve(l):
        if l <= 0:
            return min([v.min() for v, p in zip(V, P) if p > 0])

        budget = -np.log(l) - z
        valid = 0 <= budget

        if not np.any(valid):
            return sum([v[np.argmin(np.abs(Lam - p))] * p for p, v in zip(P, V)])

        validIndex = np.where(valid)[0]

        def min_c(index):
            remainBudget = budget[index] + Xi[0][index] * np.log(Lam)
            c0valid = 0 <= remainBudget
            c0 = np.where(c0valid)[0]
            c1 = np.searchsorted(
                Lam, np.exp(-remainBudget[c0valid] / (Xi[1][index])), side="left"
            )
            return ((Xi[0][index] * V[0][c0]) + (Xi[1][index] * V[1][c1])).min()

        return (np.vectorize(min_c, otypes=[np.double])(validIndex)).min()

    return np.array([par_solve(l) for l in Lam])


s0a0 = ddf.distribution(X=np.array([-300, 300]), p=np.array([0.25, 0.75]))
s0a1 = ddf.distribution(X=np.array([0]), p=np.array([1.0]))
s0a2 = ddf.distribution(X=np.array([-50, 250]), p=np.array([0.5, 0.5]))
s1 = ddf.distribution(X=np.array([100]), p=np.array([1.0]))
Lambda = np.linspace(0, 1, 10001)
# Simple case 1
s0a0s0 = np.full_like(Lambda, s0a0.X[0])
s0a0s1 = np.full_like(Lambda, s0a0.X[1])
s0a1s = np.full_like(Lambda, s0a1.X[0])
s0a2s0 = np.full_like(Lambda, s0a2.X[0])
s0a2s1 = np.full_like(Lambda, s0a2.X[1])
s1as = np.full_like(Lambda, s1.X[0])

# Get s1 EVaR Values for each action and get the max of the EVaR values
EVaRs0a = getEVaRs([s0a0, s0a1, s0a2], Lambda)

xi = np.linspace(0, 1, 10001)
s0a0dEVaR = discreteEVaR([xi, 1 - xi], s0a0.p.values, [s0a0s0, s0a0s1], Lambda)
s0a2dEVaR = discreteEVaR([xi, 1 - xi], s0a2.p.values, [s0a2s0, s0a2s1], Lambda)
# Compare difference
(np.abs(s0a0dEVaR - EVaRs0a[0])).max()
(np.abs(s0a2dEVaR - EVaRs0a[2])).max()

# Case 2: Harder case
jointa0 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a0, s1]])
jointa1 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a1, s1]])
jointa2 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a2, s1]])
jointEVaRa = getEVaRs([jointa0, jointa1, jointa2], Lambda)

jointa0dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[0], s1as], Lambda)
jointa1dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[1], s1as], Lambda)
jointa2dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[2], s1as], Lambda)
(np.abs(jointa0dEVaR - jointEVaRa[0])).max()
(np.abs(jointa1dEVaR - jointEVaRa[1])).max()
(np.abs(jointa2dEVaR - jointEVaRa[2])).max()
