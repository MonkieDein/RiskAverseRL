# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:00:17 2023

@author: USER
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from distDF import *

X = np.array([1, 80, 1, 70, 2, 2, 50, 60])
p = np.array([1, 9, 2, 9, 5, 2, 45, 27])/100
d = distribution(X, p)

Lambda = np.linspace(0, 1, 101)
cvar, var = AVaR(d, Lambda), VaR(d, Lambda)

RiskTransition = namedtuple("RiskTransition", ["states", "riskindexes"])


def VaR_iter(d, Lam, mdp_Psa, Average=True, optTran=True):
    M, N = len(Lam), len(d.index)
    S = range(len(mdp_Psa))
    j = 0  # j = 0-M iterate over lambda

    # initialize answer array
    output = np.zeros(M)
    condRisks = np.zeros_like(mdp_Psa)
    riskindexes = np.searchsorted(Lam, condRisks)
    risktrans = []

    for i in range(N):
        while (j < M) and (Lam[j] <= d.cdf[i]):
            output[j] = d.X[i] if (i == 0 or not Average) else (
                d.XTP[i-1] + d.X[i]*(Lam[j] - d.cdf[i-1]))/Lam[j]
            if not optTran:
                riskindexes[d.s[i]] = np.searchsorted(
                    Lam, (condRisks[d.s[i]] + (Lam[j] - (d.cdf[i-1] if i > 0 else 0))/mdp_Psa[d.s[i]]))
            risktrans[j] = RiskTransition(S, riskindexes.copy())
            j += 1
        condRisks[d.s[i]] = d.cdf_cond[i]
        riskindexes[d.s[i]] = min(
            M-1, np.searchsorted(Lam, condRisks[d.s[i]]) + optTran)

    if (j < M):
        output[j:] = d.X[N-1] if Average else d.XTP[N-1]
        risktrans += [RiskTransition(S, riskindexes.copy())] * (M-j)

    return output, risktrans


def VaR2D(Lam, var, Average=False, decimal=10):
    p = np.diff(Lam, prepend=0)
    if Average:
        X = np.round(np.diff(Lam * var, prepend=0) / (p), decimal)
    else:
        X = var.values
    d = distribution(X, p)
    return d

# def quantileRegQvalues()