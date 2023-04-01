import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

import distDF as ddf
import riskMeasure as rm

Q_sa = namedtuple('Qsa', ['val', 'risk_trans'])  # risk_trans[L,S]
V_s = namedtuple('Vs', ['val', 'action', 'risk_trans'])
obj = namedtuple('obj', ['rho', 'Lam', 'l'])


class TabMDP:

    def __init__(self, S, A: list[int], R: np.ndarray, P: np.ndarray, discount: float):
        self.S = S
        self.lSl = len(S)
        self.A = A
        self.lAl = len(A)
        self.R = R
        self.P = P
        self.gamma = discount

    def value_iteration(self, obj):
        nested = set(['ERM', 'nERM', 'nVaR', 'nCVaR', 'nEVaR', 'E', 'mean'])
        quantile = set(["CVaR", "VaR"])
        if obj.rho in nested:
            return self.finiteVi(obj, param=None)
        elif obj.rho in quantile:
            return self.quantileVi(obj)
        else:
            assert 0, obj.rho + "-MDP not supported. Use :" + \
                str(nested | quantile)
            return self.finiteVi(obj, param=None)

    def qval(self, obj, s, t, v, param):
        if obj.rho == 'ERM':
            return [rm.ERM(self.R[s, a, :] + self.gamma * v[t+1, :],
                           param * self.gamma ** t, self.P[s, a, :]) for a in self.A]
        if obj.rho == 'nERM':
            return [rm.ERM(self.R[s, a, :] + self.gamma * v[t+1, :], param, self.P[s, a, :]) for a in self.A]
        if obj.rho == 'nVaR':
            return [rm.VaR(self.R[s, a, :] + self.gamma * v[t+1, :], param, self.P[s, a, :]) for a in self.A]
        if obj.rho == 'nCVaR':
            return [rm.CVaR(self.R[s, a, :] + self.gamma * v[t+1, :], param, self.P[s, a, :]) for a in self.A]
        if obj.rho == 'nEVaR':
            return [rm.EVaR(self.R[s, a, :] + self.gamma * v[t+1, :], param, self.P[s, a, :]) for a in self.A]
        return [rm.E(self.R[s, a, :] + self.gamma * v[t+1, :], self.P[s, a, :]) for a in self.A]

    def finiteVi(self, obj, param=None):
        v = np.empty((obj.T+1, self.lSl))
        pi = np.empty((obj.T, self.lSl))
        v[obj.T, :] = 0

        for t in reversed(range(obj.T)):
            # initialize vectors
            for s in self.S:
                Qs = self.qval(obj, s, t, v, param)
                v[t, s] = max(Qs)
                pi[t, s] = np.argmax(Qs)
        return V_s(v, pi, None)

    def getRiskTrans(self, obj, d_conds, var):
        N = obj.l-1
        sol = []
        for d in d_conds:
            i = np.maximum(np.searchsorted(d.X.values, var)-1, 0)
            # sol.append(np.minimum(np.searchsorted(obj.Lam,d.cdf.values[i], side='right'),N))
            sol.append(
                np.minimum(np.searchsorted(obj.Lam, d.cdf.values[i])+1, N))
        return np.stack(sol, axis=1)

    def quantileQval(self, obj, s: int, a: int, V: np.ndarray):
        distV = [ddf.CVaR2D(V[sn, :], obj.Lam) for sn in self.S]
        # calculate the conditional probability of the next state
        d_conds = [ddf.condD((self.R[s, a, sn] + (self.gamma * distV[sn].X.values)),
                             distV[sn].p.values, self.P[s, a, sn]) for sn in self.S]
        # calculate the statistics (VaR/CVaR) for each risk level lambda of the joint distribution (d)
        d = ddf.joinD(d_conds)
        var = ddf.VaR(d, obj.Lam)

        if obj.rho == 'VaR':
            return Q_sa(var, self.getRiskTrans(obj, d_conds, var))
        else:
            cvar = ddf.CVaR(d, obj.Lam)
            return Q_sa(cvar, self.getRiskTrans(obj, d_conds, var))

    def quantileVi(self, obj):
        v = np.empty((obj.T+1, self.lSl, obj.l))
        pi = np.empty((obj.T, self.lSl, obj.l))
        xi = np.empty((obj.T, self.lSl, obj.l, self.lSl))
        v[obj.T, :, :] = 0
        for t in reversed(range(obj.T)):
            # initialize vectors
            for s in self.S:
                Qs = [self.quantileQval(obj, s, a, v[t+1, :, :])
                      for a in self.A]
                v[t, s, :] = np.maximum.reduce([q.val for q in Qs])
                pi[t, s, :] = np.argmax([q.val for q in Qs], axis=0)
                for l, a in enumerate(pi[t, s, :]):
                    xi[t, s, l, :] = Qs[a].risk_trans[l, :]

        return V_s(v, pi, xi)


# def VaR_iter(d, Lam, mdp_Psa, Average=True, optTran=True):
#     M, N = len(Lam), len(d.index)
#     S = range(len(mdp_Psa))
#     j = 0  # j = 0-M iterate over lambda

#     # initialize answer array
#     output = np.zeros(M)
#     condRisks = np.zeros_like(mdp_Psa)
#     riskindexes = np.searchsorted(Lam, condRisks)
#     risktrans = []

#     for i in range(N):
#         while (j < M) and (Lam[j] <= d.cdf[i]):
#             output[j] = d.X[i] if (i == 0 or not Average) else (
#                 d.XTP[i-1] + d.X[i]*(Lam[j] - d.cdf[i-1]))/Lam[j]
#             if not optTran:
#                 riskindexes[d.s[i]] = np.searchsorted(
#                     Lam, (condRisks[d.s[i]] + (Lam[j] - (d.cdf[i-1] if i > 0 else 0))/mdp_Psa[d.s[i]]))
#             risktrans[j] = riskindexes.copy()
#             j += 1
#         condRisks[d.s[i]] = d.cdf_cond[i]
#         riskindexes[d.s[i]] = min(
#             M-1, np.searchsorted(Lam, condRisks[d.s[i]]) + optTran)

#     if (j < M):
#         output[j:] = d.X[N-1] if Average else d.XTP[N-1]
#         risktrans +=  riskindexes.copy() * (M-j)

#     return output, risktrans
