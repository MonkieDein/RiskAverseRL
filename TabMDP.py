from collections import namedtuple

import numpy as np
import pandas as pd

import distDF as ddf

Q_sa = namedtuple('Qsa', ['val', 'transition'])
V_s = namedtuple('Vs', ['val', 'action','transition'])


def getRiskTrans(d_conds,var,Lam):
    N = len(Lam)-1
    sol = []
    for d in d_conds:
        i = np.maximum(np.searchsorted(d.X, var)-1,0)
        # sol.append(np.minimum(np.searchsorted(Lam,d.cdf.values[i], side='right'),N))
        sol.append(np.minimum(np.searchsorted(Lam,d.cdf.values[i])+1,N))
    return np.stack(sol, axis=0)


def VaR_iter(d_conds, Lam):
    var = ddf.VaR(ddf.joinD(d_conds),Lam)
    return Q_sa(var , getRiskTrans(d_conds,var,Lam))

def qvalue_VaR(MDP, obj, s, a, V:list[V_s]):
    # MDP.S,A,r,p,gamma
    # obj.lam
    # v[S][L]
    Vs_next = [ddf.VaR2D(vs.val,obj.lam) for vs in V] 
    # calculate the conditional probability of the next state
    d_conds = [ddf.condD( (MDP.r[s,a,sn] + (MDP.gamma * Vs_next[sn].X)), Vs_next[sn].p , MDP.p[s,a,sn] ) for sn in MDP.S]
    # calculate the value at risk for each risk level lambda
    var = ddf.VaR(ddf.joinD(d_conds),obj.lam)
    # retuun Q value function
    return Q_sa(var , getRiskTrans(d_conds,var,obj.lam))

def qvalue_CVaR(MDP, obj, s, a, V:list[V_s]):
    # MDP.S,A,r,p,gamma
    # obj.lam
    # v[S][L]
    distV = [ddf.CVaR2D(vs.val,obj.lam) for vs in V] 
    # calculate the conditional probability of the next state
    d_conds = [ddf.condD( (MDP.r[s,a,sn] + (MDP.gamma * distV[sn].X)), distV[sn].p , MDP.p[s,a,sn] ) for sn in MDP.S]
    # calculate the statistics (VaR/CVaR) for each risk level lambda of the joint distribution (d)
    d = ddf.joinD(d_conds)
    var = ddf.VaR(d,obj.lam)
    avar = ddf.CVaR(d,obj.lam)
    return Q_sa(avar , getRiskTrans(d_conds,var,obj.lam))

def v_VaR(MDP, obj, s, V:list[V_s]):
    Qs = [qvalue_VaR(MDP, obj, s, a, V) for a in MDP.A]
    pi_s = np.argmax([q.val for q in Qs],axis=0)
    v_s = np.maximum.reduce([q.val for q in Qs],axis=0)
    

