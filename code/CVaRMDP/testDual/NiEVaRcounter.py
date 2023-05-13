from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import distDF as ddf
import riskMeasure as rm


def getEVaRs(list_of_distributions, Lambda):
    return [ddf.EVaR(d, Lambda) for d in list_of_distributions]


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


def discreteEVaR(Xi, P, V, Lam):
    z = np.sum([xi * np.log(xi / p) for xi, p in zip(Xi, P)], axis=0)
    np.nan_to_num(z, copy=False, nan=np.inf)
    LamNan = Lam.copy()
    LamNan[LamNan == 0] = np.nan

    def par_solve(l):
        if l <= 0:
            return min([v.min() for v, p in zip(V, P) if p > 0])

        budget = -np.log(l) - z
        valid = 0 <= budget

        if not np.any(valid):
            return sum([v[np.argmin(np.abs(Lam - p))] * p for p, v in zip(P, V)])

        validIndex = np.where(valid)[0]

        def min_c(index):
            remainBudget = budget[index] + Xi[0][index] * np.log(LamNan)
            c0valid = 0 <= remainBudget
            c0 = np.where(c0valid)[0]
            c1 = np.searchsorted(
                Lam, np.exp(-remainBudget[c0valid] / (Xi[1][index])), side="left"
            )
            return ((Xi[0][index] * V[0][c0]) + (Xi[1][index] * V[1][c1])).min()

        return (np.vectorize(min_c, otypes=[np.double])(validIndex)).min()

    return np.array(Parallel(n_jobs=7)(delayed(par_solve)(l) for l in Lam))
    # return np.array([par_solve(l) for l in Lam])


def discreteNi(Xi, P, V, Lam):
    z1 = np.sum([xi * np.log(xi / p) for xi, p in zip(Xi, P)], axis=0)
    np.nan_to_num(z1, copy=False, nan=np.inf)

    z2 = np.max([xi / p for xi, p in zip(Xi, P)], axis=0)

    def par_solve(l):
        if l > 0:
            valid = (z2 <= (1 / l)) & (z1 <= (-np.log(l)))
        else:
            valid = np.full_like(Lam, True, dtype=bool)
        if not np.any(valid):
            return sum([v[np.argmin(np.abs(Lam - p))] * p for p, v in zip(P, V)])
        return (np.sum([xi[valid] * v[valid] for xi, v in zip(Xi, V)], axis=0)).min()

    return np.vectorize(par_solve, otypes=[np.double])(Lam)


s1a1 = ddf.distribution(X=np.array([-300, 300]), p=np.array([0.25, 0.75]))
s1a2 = ddf.distribution(X=np.array([0]), p=np.array([1.0]))
s1a3 = ddf.distribution(X=np.array([-50, 250]), p=np.array([0.5, 0.5]))
s2 = ddf.distribution(X=np.array([100]), p=np.array([1.0]))
Lambda = np.linspace(0, 1, 10001)
# Lambda = np.insert((0.999 ** np.arange(10000))[::-1], 0, 0)
s1a1s1 = np.full_like(Lambda, s1a1.X[0])
s1a1s2 = np.full_like(Lambda, s1a1.X[1])
s1a3s1 = np.full_like(Lambda, s1a3.X[0])
s1a3s2 = np.full_like(Lambda, s1a3.X[1])

s1a1dCVaR = discreteCVaR([Lambda, 1 - Lambda], s1a1.p.values, [s1a1s1, s1a1s2], Lambda)
s1a1dEVaR = discreteEVaR([Lambda, 1 - Lambda], s1a1.p.values, [s1a1s1, s1a1s2], Lambda)
s1a1dNi = discreteNi([Lambda, 1 - Lambda], s1a1.p.values, [s1a1s1, s1a1s2], Lambda)

# Get s1 EVaR Values for each action and get the max of the EVaR values
CVaRs1a = getCVaRs([s1a1, s1a2, s1a3], Lambda)
EVaRs1a = getEVaRs([s1a1, s1a2, s1a3], Lambda)

(np.abs(s1a1dEVaR - EVaRs1a[0])).max()
(np.abs(s1a1dCVaR - CVaRs1a[0])).max()
(np.abs(s1a1dCVaR - s1a1dNi)).max()

s1maxEVaR = np.maximum.reduce(EVaRs1a)
s1maxCVaR = np.maximum.reduce(CVaRs1a)
s1maxNi = ddf.CVaR2D(s1maxCVaR, Lambda)

# EVaR s1 plot
plt.plot(Lambda, s1maxEVaR, "k--", label="max_a", linewidth=2.5)
plt.plot(Lambda, EVaRs1a[0], "r-", label="a1")
plt.plot(Lambda, EVaRs1a[1], "b-", label="a2")
plt.plot(Lambda, EVaRs1a[2], "g-", label="a3")
plt.title("s1 EVaR value")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="lower right")
plt.show()
# Ni's EVaR s1 plot
plt.plot(Lambda, s1maxCVaR, "k--", label="max_a", linewidth=2.5)
plt.plot(Lambda, CVaRs1a[0], "r-", label="a1")
plt.plot(Lambda, CVaRs1a[1], "b-", label="a2")
plt.plot(Lambda, CVaRs1a[2], "g-", label="a3")
plt.title("s1 Ni's EVaR value")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="lower right")
plt.show()
# Joining s1 and s2 distribution
jointExp = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1maxNi, s2]])
jointa1 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a1, s2]])
jointa2 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a2, s2]])
jointa3 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a3, s2]])
jointExpectedNi = ddf.CVaR(jointExp, Lambda)
jointa1EVaR = ddf.EVaR(jointa1, Lambda)
jointa2EVaR = ddf.EVaR(jointa2, Lambda)
jointa3EVaR = ddf.EVaR(jointa3, Lambda)
jointDeployNi = np.maximum(jointa1EVaR, jointa2EVaR)
jointoptEVaR = np.maximum.reduce([jointa1EVaR, jointa2EVaR, jointa3EVaR])
# join EVaR plot
plt.plot(Lambda, jointExpectedNi, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, jointa1EVaR, "r-", label="a1")
plt.plot(Lambda, jointa2EVaR, "b-", label="a2")
plt.plot(Lambda, jointa3EVaR, "g-", label="a3")
plt.title("join EVaR value")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()
# join EVaR plot performance plot
plt.plot(Lambda, jointExpectedNi, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, jointoptEVaR, "g-", label="True Opt")
plt.plot(Lambda, jointDeployNi, "r-.", label="a2")
plt.title("join EVaR performance plot")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()
