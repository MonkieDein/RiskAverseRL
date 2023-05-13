import pickle
import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from joblib import Parallel, delayed

import distDF as ddf
import riskMeasure as rm


def getEVaRs(list_of_distributions, Lambda):
    return [ddf.EVaR(d, Lambda) for d in list_of_distributions]


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def save_object(filename, obj):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


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
            vals = (Xi[0][index] * V[0][c0]) + (Xi[1][index] * V[1][c1])
            amin = np.argmin(vals)
            return (vals[amin], c0[amin], c1[amin], Xi[0][index])

        valmin = [min_c(i) for i in validIndex]
        sol = np.argmin([v[0] for v in valmin])
        return valmin[sol]
        # return (np.vectorize(min_c, otypes=[np.double])(validIndex)).min()

    return Parallel(n_jobs=7)(delayed(par_solve)(l) for l in Lam)
    # return np.array([par_solve(l) for l in Lam])


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
s0maxEVaR = np.maximum.reduce(EVaRs0a)

plt.plot(Lambda, s0maxEVaR, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, EVaRs0a[0], "r-", label="a1")
plt.plot(Lambda, EVaRs0a[1], "b-", label="a2")
plt.plot(Lambda, EVaRs0a[2], "g-", label="a3")
plt.title("join EVaR value")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()
xi = np.linspace(0, 1, 1001)
# t1 = time.time()
# s0a0dEVaR = discreteEVaR([xi, 1 - xi], s0a0.p.values, [s0a0s0, s0a0s1], Lambda)
# t2 = time.time()
# print(t2 - t1)

# s0a2dEVaR = discreteEVaR([xi, 1 - xi], s0a2.p.values, [s0a2s0, s0a2s1], Lambda)
# # Compare difference
# (np.abs(s0a0dEVaR - EVaRs0a[0])).max()
# (np.abs(s0a2dEVaR - EVaRs0a[2])).max()

# Case 2: Harder case
jointa0 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a0, s1]])
jointa1 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a1, s1]])
jointa2 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s0a2, s1]])
jointEVaRa = getEVaRs([jointa0, jointa1, jointa2], Lambda)
# jointamax = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [s0maxEVaR, s1as], Lambda)
# save_object("data.pickle", jointamax)
test = load_object("data.pickle")


plt.plot(Lambda, jointamax, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, jointEVaRa[0], "r-", label="a1")
plt.plot(Lambda, jointEVaRa[1], "b-", label="a2")
plt.plot(Lambda, jointEVaRa[2], "g-", label="a3")
plt.title("join EVaR value")
plt.ylabel("EVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()
# jointa0dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[0], s1as], Lambda)
# jointa1dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[1], s1as], Lambda)
# jointa2dEVaR = discreteEVaR([xi, 1 - xi], [0.5, 0.5], [EVaRs0a[2], s1as], Lambda)
# (np.abs(jointa0dEVaR - jointEVaRa[0])).max()
# (np.abs(jointa1dEVaR - jointEVaRa[1])).max()
# (np.abs(jointa2dEVaR - jointEVaRa[2])).max()

# Running this with 1001 linspace works okay but 10001 would let it shoot to hours
# long wait
