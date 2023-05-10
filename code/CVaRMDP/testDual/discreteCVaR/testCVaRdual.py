from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

import distDF as ddf
import riskMeasure as rm


def getCVaRs(list_of_distributions, Lambda):
    return [ddf.CVaR(d, Lambda) for d in list_of_distributions]


def discreteCVaR(Xi, P, V, Lam):
    z = np.max([xi / p for xi, p in zip(Xi, P)], axis=0)

    def par_solve(l):
        if l != 0:
            valid = z <= (1 / l)
        else:
            return min([v.min() for v, p in zip(V, P) if p > 0])
        if not np.any(valid):
            return sum([v[np.argmin(np.abs(Lam - p))] * p for p, v in zip(P, V)])

        return (
            np.sum(
                [
                    xi[valid]
                    * v[np.searchsorted(Lam, xi[valid] / p * l, side="right") - 1]
                    for p, xi, v in zip(P, Xi, V)
                ],
                axis=0,
            )
        ).min()

    return np.vectorize(par_solve, otypes=[np.double])(Lam)


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
CVaRs0a = getCVaRs([s0a0, s0a1, s0a2], Lambda)

xi = np.linspace(0, 1, 1000001)
s0a0dCVaR = discreteCVaR([xi, 1 - xi], s0a0.p.values, [s0a0s0, s0a0s1], Lambda)
s0a2dCVaR = discreteCVaR([xi, 1 - xi], s0a2.p.values, [s0a2s0, s0a2s1], Lambda)
# Compare difference
(np.abs(s0a0dCVaR - CVaRs0a[0])).max()
(np.abs(s0a2dCVaR - CVaRs0a[2])).max()

# Case 2: Harder case
jointa0 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a0, s1]])
jointa1 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a1, s1]])
jointa2 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a2, s1]])
jointa0CVaR = ddf.CVaR(jointa0, Lambda)
jointa1CVaR = ddf.CVaR(jointa1, Lambda)
jointa2CVaR = ddf.CVaR(jointa2, Lambda)

jointa0dCVaR = discreteCVaR([xi, 1 - xi], [0.5, 0.5], [CVaRs0a[0], s1as], Lambda)
jointa1dCVaR = discreteCVaR([xi, 1 - xi], [0.5, 0.5], [CVaRs0a[1], s1as], Lambda)
jointa2dCVaR = discreteCVaR([xi, 1 - xi], [0.5, 0.5], [CVaRs0a[2], s1as], Lambda)
(np.abs(jointa0CVaR - jointa0dCVaR)).max()
(np.abs(jointa1CVaR - jointa1dCVaR)).max()
(np.abs(jointa2CVaR - jointa2dCVaR)).max()
