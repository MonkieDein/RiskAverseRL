import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp as LSE

import base as bs
import distDF as ddf

"""
E(X,prob): takes in random variables X and optional probability prob
and returns the expected value of the distribution.
"""


def E(X: np.ndarray, prob: np.ndarray = np.empty(0)):
    if prob.size > 0:
        return (X * prob).sum()
    return X.mean()


"""
min(X,prob): takes in random variables X and optional probability prob
and returns the minimum of X that is possible to occur.
"""


def min(X: np.ndarray, prob: np.ndarray = np.empty(0)):
    if prob.size > 0:
        return (X[prob > 0]).min()
    return X.min()


"""
ERM(X,Alpha,prob) = LSE(-alpha*X,b=prob)/-alpha
LSE(a,b)  = np.log(np.sum(b*np.exp(a)))
LSE trick = np.log(np.sum(b*np.exp(a-C)))+C # where #C = a.max()
"""


def ERM(X: np.ndarray, Alpha: np.ndarray, prob: np.ndarray = np.empty(0)):
    X, prob = bs.check_size(X, prob)

    def opt(alpha):
        return (LSE(-alpha * X, b=prob) / -alpha) if alpha > 0 else E(X, prob)

    return np.vectorize(opt, otypes=[np.double])(Alpha)


"""
EVaRaprx(X,Lam,prob) = sup_a( ERM(X,a,prob) + log(lam)/a )
"""


def EVaRaprx(
    X: np.ndarray,
    Lam: np.ndarray,
    prob: np.ndarray = np.empty(0),
    Beta: np.ndarray = np.empty(0),
):
    if len(Beta) == 0:
        Beta = 0.98 ** np.arange(-1000, 1000)
    Lam = bs.npfy(Lam)
    mu = E(X, prob)
    minima = X.min()
    lhs = ERM(X, Beta, prob)

    def get_opt(l):
        return min(max(minima, (lhs + np.log(l) / Beta).max()), mu)

    return np.vectorize(get_opt, otypes=[np.double])(Lam)


"""
EVaR(X,Lam,prob) = sup_a( ERM(X,a,prob) + log(lam)/a )
                 = sup_a( (- LSE(-a*X,b=prob) + log(lam))/a )
                 = sup_t( (- LSE(-X/t,b=prob) + log(lam))*t )
                 = - inf_t( (LSE(-X/t,b=prob) - log(lam))*t )
"""


def EVaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0)):
    if len(X) == 1:
        return np.full_like(Lam, X[0])

    Lam = bs.npfy(Lam)

    mu = E(X, prob)
    X, prob = bs.check_size(X, prob)
    minima = X.min()

    def opt(lam):
        return (
            max(
                min(
                    -minimize_scalar(
                        lambda t: (LSE(-X / t, b=prob) - np.log(lam)) * t,
                        bounds=(np.nextafter(0, 1), 1e10),
                    ).fun,
                    mu,
                ),
                minima,
            )
            if lam > 0
            else minima
        )

    # sol = namedtuple("sol", ["value", "optimizer"])
    # return sol(max(min(-res.fun, E(X, prob)), min(X[prob > 0])), 1/res.x)
    return np.vectorize(opt, otypes=[np.double])(Lam)


"""
VaR(X,Lam,prob,mode) 
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
1. Create a distribution (d).
2. Call ddf.VaR function with distribution (d). 
"""


def VaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0), mode=1):
    X, prob = bs.check_size(X, prob)
    d = ddf.distribution(X, prob)
    return ddf.VaR(d, Lam, mode=mode)


"""
AVaR(X,Lam,prob,mode) 
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
1. Create a distribution (d).
2. Call ddf.AVaR function with distribution (d). 
"""


def CVaR(X: np.ndarray, Lam: np.ndarray, prob: np.ndarray = np.empty(0), mode=1):
    X, prob = bs.check_size(X, prob)
    d = ddf.distribution(X, prob)
    return ddf.CVaR(d, Lam, mode=mode)
