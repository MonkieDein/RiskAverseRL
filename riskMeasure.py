
# from collections import namedtuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp as LSE

import distDF as ddf

'''
E(X,prob): takes in random variables X and optional probability prob
and returns the expected value of the distribution.
'''


def E(X: np.ndarray, prob: np.ndarray = np.empty(0)):
    if prob.size > 0:
        return (X * prob).sum()
    return X.mean()


'''
check_size(X,prob): takes in random variables X and optional probability prob
1. if no probability is given, uniform distribution is assumed.
2. if X and prob are of different sizes, error is thrown.
3. remove values with probability 0.
'''


def check_size(X: np.ndarray, prob: np.ndarray):
    if prob.size == 0:
        prob = np.full(X.shape, 1/X.size)
    else:
        assert X.size == prob.size, "X and prob must have the same size"
        X = X[prob > 0]
        prob = prob[prob > 0]
    return X, prob


'''
ERM(X,Alpha,prob) = LSE(-alpha*X,b=prob)/-alpha
LSE(a,b)  = np.log(np.sum(b*np.exp(a)))
LSE trick = np.log(np.sum(b*np.exp(a-C)))+C # where #C = a.max()
'''


def ERM(X: np.ndarray, Alpha: np.ndarray, prob: np.ndarray = np.empty(0)):
    X, prob = check_size(X, prob)

    def opt(alpha):
        return (LSE(-alpha*X, b=prob) / -alpha) if alpha > 0 else E(X, prob)

    return np.vectorize(opt)(Alpha)


'''
EVaR(X,Lam,prob) = sup_a( ERM(X,a,prob) + log(lam)/a )
                 = sup_a( (- LSE(-a*X,b=prob) + log(lam))/a )
                 = sup_t( (- LSE(-X/t,b=prob) + log(lam))*t )
                 = - inf_t( (LSE(-X/t,b=prob) - log(lam))*t )
'''


def EVaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0)):
    Lam = ddf.npfy(Lam)

    mu = E(X, prob)
    X, prob = check_size(X, prob)
    minima = X.min()

    def opt(lam):
        return max(min(-minimize_scalar(
            lambda t: (LSE(-X/t, b=prob) - np.log(lam))*t,
            bounds=(np.nextafter(0, 1), 1e10)).fun,
            mu), minima) if lam > 0 else minima
    # sol = namedtuple("sol", ["value", "optimizer"])
    # return sol(max(min(-res.fun, E(X, prob)), min(X[prob > 0])), 1/res.x)
    return np.vectorize(opt)(Lam)


'''
VaR(X,Lam,prob,mode) 
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
1. Create a distribution (d).
2. Call ddf.VaR function with distribution (d). 
'''


def VaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0), mode=1):
    X, prob = check_size(X, prob)
    d = ddf.distribution(X, prob)
    return ddf.VaR(d, Lam, mode=mode)


'''
AVaR(X,Lam,prob,mode) 
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
1. Create a distribution (d).
2. Call ddf.AVaR function with distribution (d). 
'''


def AVaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0), mode=1):
    X, prob = check_size(X, prob)
    d = ddf.distribution(X, prob)
    return ddf.AVaR(d, Lam, mode=mode)


def CVaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0), mode=1):
    return AVaR(X, Lam, prob, mode=mode)
