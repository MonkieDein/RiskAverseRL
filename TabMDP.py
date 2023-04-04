import warnings
from collections import namedtuple
from math import ceil

import numpy as np
import pandas as pd

import distDF as ddf
import riskMeasure as rm

Q_sa = namedtuple('Qsa', ['val', 'risk_trans'])  # risk_trans[L,S]
V_s = namedtuple('Vs', ['val', 'pi', 'risk_trans'])


class objective:
    def __init__(self, rho: str = "E", pars: list[float] = [1.0], T: int = -1, delta: float = 0.0):
        self.rho = rho  # represent the risk measure used
        self.T = T  # T = -1 refers to infinite horizon
        self.pars = pars  # risk levels, (consider numpy or list)
        self.delta = delta  # EVaR and ERM delta epsilon parameter
        self.l = len(pars)  # number of risk levels


class rhoMDP:
    rType = {'nest': {'nERM', 'nVaR', 'nCVaR', 'nEVaR', "min", 'E', 'mean'},
             'quant': {"CVaR", "VaR"},
             'ent': {'ERM', "EVaR"}}

    def __init__(self, S, A: list[int], R: np.ndarray, P: np.ndarray, discount: float, s0: np.ndarray):
        self.S = S  # Distinct finite states
        self.lSl = len(S)  # number of states
        self.s0 = s0  # Initial state distribution
        self.A = A  # Action space
        self.lAl = len(A)  # number of actions
        self.R = R  # Reward matrix R[s,a,s']
        self.P = P  # Probability matrix P[s,a,s']
        self.gamma = discount  # discount factor

    def VI(self, obj, vTerm=0, piTerm=0):
        all_risks = set().union(*self.rType.values())
        assert obj.rho not in (all_risks), obj.rho + \
            "-MDP not supported. Use :"+str(all_risks)
        if obj.rho in self.rType['nest']:
            if obj.T == -1:  # infinite horizon nested
                return [self.infiniteVi(obj, param=param) for param in obj.pars]
            else:  # finite horizon nested
                return [self.finiteVi(obj, param=param, vTerm=vTerm, piTerm=piTerm) for param in obj.pars]
        elif obj.rho in self.rType['quant']:  # finite horizon nested
            assert obj.T > 0, "T must be positive"
            return self.quantileVi(obj)
        else:  # elif obj.rho in entropic:
            if obj.rho == "ERM":
                return self.ermVi(obj)
            else:  # EVaR case
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
        if obj.rho == "min":
            return [rm.min(self.R[s, a, :] + self.gamma * v[t+1, :], self.P[s, a, :]) for a in self.A]
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
        return V_s(v, pi, param)

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
        return V_s(v[0, :], pi, param)

    def customAlpha(self, obj):
        pars = min([l for l in obj.pars if l > 0])
        DeltaR = np.ptp(self.R)
        Alpha = []
        alp = 8*obj.delta/(DeltaR**2)
        while alp < (-np.log(pars)/obj.delta):
            Alpha.append(alp)
            alp = alp * np.log(pars)/(alp*obj.delta+np.log(pars))
        Alpha.append(alp)
        return Alpha

    def esTerm(self, obj):
        assert obj.delta > 0, "please set delta > 0, for ERM infinite horizon"
        minA = min([l for l in obj.pars if l > 0])
        DeltaR = np.ptp(self.R)
        return ceil(np.log(8*obj.delta/((minA)*DeltaR**2))/(2*np.log(obj.delta)))

    def ermVi(self, obj):
        if obj.T == -1:  # infinite horizon
            # Solve risk neutral Nominal-MDP
            term = self.infiniteVi(objective(rho="E"))
            obj.T = 0 if all(
                l == 0 for l in obj.pars) else self.esTerm(obj)
        else:  # finite horizon
            assert obj.T > 0, "please set objective horizon > 0"
            term = V_s(0.0, 0, None)
        return [self.finiteVi(obj, param=alp, vTerm=term.val, piTerm=term.pi) for alp in obj.pars]

    def evarVi(self, obj):
        # If lambda == 1 then is standard MDP
        if all(x == 1 for x in obj.pars):
            return [self.VI(objective(rho="E", T=obj.T)) for x in obj.pars]

        assert obj.delta > 0, "please set delta > 0 for EVaR-MDP"

        if obj.T == -1:  # If infinite horizon
            obj.delta /= 2  # split half for discrete Alpha and approx T'
        else:  # If finite horizon
            assert obj.T > 0, "please set objective horizon > 0"

        Alpha = self.customAlpha(obj)  # generate RASR alphas
        objERM = objective(rho="ERM", T=obj.T, pars=Alpha, delta=obj.delta)
        ERM_ret = self.ermVi(objERM)
        ERMs = np.array([rm.ERM(v.val[0, :], Alpha[i], self.s0)
                         for i, v in enumerate(ERM_ret)])
        npAlp = np.array(Alpha)
        return [ERM_ret[np.argmax(ERMs + np.log(l)/npAlp)] for l in obj.pars]

    def getRiskTrans(self, obj, d_conds, var):
        N = obj.l-1
        sol = []
        for d in d_conds:
            i = np.maximum(np.searchsorted(d.X.values, var)-1, 0)
            # sol.append(np.minimum(np.searchsorted(obj.pars,d.cdf.values[i], side='right'),N))
            sol.append(
                np.minimum(np.searchsorted(obj.pars, d.cdf.values[i])+1, N))
        return np.stack(sol, axis=1)

    def quantileQval(self, obj, s: int, a: int, V: np.ndarray):
        distV = [ddf.CVaR2D(V[sn, :], obj.pars) for sn in self.S]
        # calculate the conditional probability of the next state
        d_conds = [ddf.condD((self.R[s, a, sn] + (self.gamma * distV[sn].X.values)),
                             distV[sn].p.values, self.P[s, a, sn]) for sn in self.S]  # type: ignore
        # calculate the statistics (VaR/CVaR) for each risk level lambda of the joint distribution (d)
        d = ddf.joinD(d_conds)
        var = ddf.VaR(d, obj.pars)

        if obj.rho == 'VaR':
            return Q_sa(var, self.getRiskTrans(obj, d_conds, var))
        else:
            cvar = ddf.CVaR(d, obj.pars)
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


# def VaR_iter(d, pars, mdp_Psa, Average=True, optTran=True):
#     M, N = len(pars), len(d.index)
#     S = range(len(mdp_Psa))
#     j = 0  # j = 0-M iterate over lambda

#     # initialize answer array
#     output = np.zeros(M)
#     condRisks = np.zeros_like(mdp_Psa)
#     riskindexes = np.searchsorted(pars, condRisks)
#     risktrans = []

#     for i in range(N):
#         while (j < M) and (pars[j] <= d.cdf[i]):
#             output[j] = d.X[i] if (i == 0 or not Average) else (
#                 d.XTP[i-1] + d.X[i]*(pars[j] - d.cdf[i-1]))/pars[j]
#             if not optTran:
#                 riskindexes[d.s[i]] = np.searchsorted(
#                     pars, (condRisks[d.s[i]] + (pars[j] - (d.cdf[i-1] if i > 0 else 0))/mdp_Psa[d.s[i]]))
#             risktrans[j] = riskindexes.copy()
#             j += 1
#         condRisks[d.s[i]] = d.cdf_cond[i]
#         riskindexes[d.s[i]] = min(
#             M-1, np.searchsorted(pars, condRisks[d.s[i]]) + optTran)

#     if (j < M):
#         output[j:] = d.X[N-1] if Average else d.XTP[N-1]
#         risktrans +=  riskindexes.copy() * (M-j)

#     return output, risktrans
