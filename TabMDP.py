import warnings
from collections import namedtuple
from math import ceil

import numpy as np
import pandas as pd

import distDF as ddf
import riskMeasure as rm

Q_sa = namedtuple('Qsa', ['val', 'risk_trans'])  # risk_trans[L,S]
V_s = namedtuple('Vs', ['val', 'action', 'risk_trans'])


class obj:
    def __init__(self, rho, Lam, T: int = -1, delta: float = 0.0):
        self.rho = rho  # represent the risk measure used
        self.T = T  # T = -1 refers to infinite horizon
        self.Lam = Lam  # risk levels, (consider numpy or list)
        self.delta = delta  # EVaR and ERM delta epsilon parameter
        self.l = len(Lam)  # number of risk levels


class rhoMDP:

    def __init__(self, S, A: list[int], R: np.ndarray, P: np.ndarray, discount: float, s0: np.ndarray):
        self.S = S
        self.s0 = s0
        self.lSl = len(S)
        self.A = A
        self.lAl = len(A)
        self.R = R
        self.P = P
        self.gamma = discount

    def value_iteration(self, obj, vTerm=0, piTerm=0):
        nested = {'ERM', 'nERM', 'nVaR', 'nCVaR', 'nEVaR', 'E', 'mean'}
        quantile = {"CVaR", "VaR"}
        maxErm = {"EVaR"}
        assert obj.rho not in (nested | quantile | maxErm), obj.rho + \
            "-MDP not supported. Use :"+str(nested | quantile | maxErm)
        if obj.rho in nested:
            if obj.T == -1:
                return [self.infiniteVi(obj, param=param) for param in obj.Lam]
            else:
                return [self.finiteVi(obj, param=param, vTerm=vTerm, piTerm=piTerm) for param in obj.Lam]
        elif obj.rho in quantile:
            return self.quantileVi(obj)
        else:  # obj.rho in maxErm: # self.evarVi( obj )
            return self.evarVi(obj)

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

    def finiteVi(self, obj, param=None, vTerm=0, piTerm=0):
        v = np.empty((obj.T+1, self.lSl))
        pi = np.empty((obj.T+1, self.lSl))
        v[obj.T, :] = vTerm
        pi[obj.T, :] = piTerm
        for t in reversed(range(obj.T)):
            for s in self.S:
                Qs = self.qval(obj, s, t, v, param)
                v[t, s] = max(Qs)
                pi[t, s] = np.argmax(Qs)
        return V_s(v, pi, None)

    def infiniteVi(self, obj, param=None, eps=1e-14, maxiter=1000):
        if obj.rho == "ERM":
            warnings.warn("ERM degrade to nested-ERM in infinite horizon")
        v = np.ones((self.lSl, 2))
        pi = np.empty((self.lSl))
        t = 0  # index of t=1 means next, t=0 means current
        v[t, :] = 0
        i = 0
        while (np.abs(v[t+1, :] - v[t, :])).max() > eps:
            v[t+1, :] = v[t, :]
            for s in self.S:
                Qs = self.qval(obj, s, t, v, param)
                v[t, s] = max(Qs)
                pi[s] = np.argmax(Qs)
            i += 1
            if i > maxiter:
                warnings.warn("Maximum number of iterations reached")
                break
        return V_s(v[0, :], pi, None)

    def genAlpha(self, obj, delta):
        lam = min([l for l in obj.Lam if l > 0])
        DeltaR = np.ptp(self.R)
        Alpha = []
        alp = 8*delta/(DeltaR**2)
        while alp < (-np.log(lam)/delta):
            Alpha.append(alp)
            alp = alp * np.log(lam)/(alp*delta+np.log(lam))
        Alpha.append(alp)
        return Alpha

    def esTerm(self, delta, minA):
        DeltaR = np.ptp(self.R)
        return ceil(np.log(8*delta/((minA)*DeltaR**2))/(2*np.log(delta)))

    def evarVi(self, obj):
        assert obj.delta > 0, "please set delta > 0"
        if obj.T == -1:  # If infinite horizon
            # Solve risk neutral Nominal-MDP
            objE = obj("E", -1, [1.0], 1, 0.0)
            vTerm = self.infiniteVi(objE)
            # Get necessary Alpha parameters
            Alpha = self.genAlpha(obj, obj.delta/2)
            objERM = obj("ERM", self.esTerm(obj.delta/2, min(Alpha)),
                         Alpha, len(Alpha), 0.0)

        else:  # If finite horizon
            assert obj.T > 0, "please set objective horizon > 0"
            vTerm = V_s(0.0, None, None)
            Alpha = self.genAlpha(obj, obj.delta)
            objERM = obj("ERM", obj.T, Alpha, len(Alpha), 0.0)
        ERM_ret = self.value_iteration(objERM, vTerm=vTerm.val)
        ERMs = np.array([rm.ERM(v.val[0, :], Alpha[i], self.s0)
                        for i, v in enumerate(ERM_ret)])
        npAlp = np.array(Alpha)
        return [ERM_ret[np.argmax(ERMs + np.log(l)/npAlp)] for l in obj.Lam]

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
