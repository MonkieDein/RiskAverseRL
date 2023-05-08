import numpy as np
import pandas as pd

import base as bs
import riskMeasure as rm

"""
preProcess(d) -> pd.DataFrame
Takes in a distribution dataframe (d):
1. remove probabilities p <= 0.
2. sorts the distribution dataframe by its value X
3. precompute cdf and XTP
returns the sorted distribution dataframe.
"""


def preProcess(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.p > 0]
    d.sort_values("X", inplace=True)

    # Normalization of pdf
    if abs(d.p.sum() - 1.0) > 1e-13:
        print("Distribution does not sum to one")
    d.p /= d.p.sum()
    probSum = d.p.sum()
    if probSum < 1:
        d.p = d.p / probSum

    # precompute repeatedly used values
    d["cdf"] = np.cumsum(d.p)
    d["XTP"] = np.cumsum((d.p * d.X))

    return d


"""
distribution(X,p) -> pd.DataFrame
Takes in an array of X values its respective probability p:
1. Creates a distribution dataframe (d) from X and p.
2. group and combine probabilities p with identical X values.
returns the preprocess (sorted and precompute cdf and XTP) distribution dataframe (d).
"""


def distribution(X: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"X": X, "p": p})
    d = d.groupby("X").agg({"p": sum}).reset_index()
    return preProcess(d)


"""
VaR2D(X,cdf) -> pd.DataFrame
Takes in an array of sorted X values its respective cdf:
1. Calcaulate the pmf from cdf :np.diff(cdf, prepend=0).
returns the distribution dataframe (d).
"""


def VaR2D(X: np.ndarray, cdf: np.ndarray) -> pd.DataFrame:
    assert bs.is_sorted(X), "values of var must be sorted"
    d = distribution(np.diff(cdf, prepend=0), X)
    return d


"""
CVaR2D(X,cdf) -> pd.DataFrame
Takes in an array of sorted CVaR values its respective cdf:
1. Calcaulate the pmf from cdf :np.diff(cdf, prepend=0).
1. Calcaulate the X from cvar and cdf : np.diff(cdf * cvar, prepend=0)/p.
returns the distribution dataframe (d).
"""


def CVaR2D(cvar: np.ndarray, cdf: np.ndarray, decimal: int = 10) -> pd.DataFrame:
    assert bs.is_sorted(cvar), "values of cvar must be sorted"
    p = np.diff(cdf, prepend=0)
    X = np.round(np.diff(cdf * cvar, prepend=0) / p, decimal)
    d = distribution(X, p)
    return d


"""
condD(X,p_cond,pr,**kwargs) -> pd.DataFrame
Takes in an array of conditional values (X), its respective conditional probability (p_cond),
marginal probability (pr), and optional arbitrary input (**kwargs):
1. Creates a conditional distribution dataframe (d) with X and p_cond.
2. Rename the columns of the dataframe to match the conditional distribution dataframe.
3. Calculate the joint probability of X.
4. For each arbitary inputs, match its key with its corresponding value.
returns the conditional distribution dataFrame.
"""


def condD(X: np.ndarray, p_cond: np.ndarray, pr: float, **kwargs) -> pd.DataFrame:
    d = distribution(X, p_cond)
    d.rename(
        columns={"p": "p_cond", "cdf": "cdf_cond", "XTP": "XTP_cond"}, inplace=True
    )
    d["p"] = d.p_cond * pr
    for key, value in kwargs.items():
        d[key] = value
    return d


"""
joinD(d_conds) -> pd.DataFrame
Takes in a list of conditional distribution (d_conds):
1. Combines the conditional distribution dataframe (d_conds) into the joint distribution dataframe (d).
returns the preprocess (sorted and precompute cdf and XTP) distribution dataframe (d).
"""


def joinD(d_conds: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat(d_conds)
    return preProcess(d)


"""
searchVaR(d,lam) -> float: Search over cdf.
Takes in a distribution (d) and risk level (lam).
Return the (lam)-th quantile of the distribution (d).
"""


def searchVaR(d: pd.DataFrame, lam: np.ndarray) -> np.ndarray:
    i = np.minimum(np.searchsorted(d.cdf, lam), len(d.cdf.values) - 1)
    return d.X.values[i]


"""
iterVaRs(d,Lam) -> float:  Iterate over sorted element of the distribution.
Takes in a distribution (d) and an array of risk level (Lam).
Return the (Lam)-th quantile of the distribution (d).
"""


def iterVaRs(d: pd.DataFrame, Lam: np.ndarray) -> np.ndarray:
    M, N = len(Lam), len(d.index)

    # initialize answer array
    output = np.empty(M)

    j = np.searchsorted(Lam, 0, side="right")
    output[:j] = d.X[0]

    for i in range(N):
        while (j < M) and (Lam[j] <= d.cdf[i]):
            output[j] = d.X[i]
            j += 1

    output[j:] = d.X[N - 1]

    return output


"""
VaR(d,Lam,mode=1) -> np.ndarray: General function that takes mode as input.
Takes in a distribution (d) and an array of risk level (Lam).
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
Return the (Lam)-th quantile of the distribution (d).
"""


def VaR(d: pd.DataFrame, Lam: np.ndarray, mode: int = 1) -> np.ndarray:
    Lam = bs.npfy(Lam)
    if mode == 0:  # iterative methods good for large Lam array
        assert bs.is_sorted(Lam), "Lambda array must be sorted"
        return iterVaRs(d, Lam)
    # search methods good for small Lam array
    return searchVaR(d, Lam)


"""
searchAVaR(d,lam) -> float: Search over cdf.
Takes in a distribution (d) and risk level (lam).
Return the (lam)-CVaR of the distribution (d).
"""


def searchCVaR(d: pd.DataFrame, lam: np.ndarray) -> np.ndarray:
    i = np.minimum(np.searchsorted(d.cdf, lam), len(d.cdf.values) - 1)
    return np.divide(
        (d.XTP.values[i] + d.X.values[i] * (lam - d.cdf.values[i])),
        lam,
        out=np.full_like(lam, d.X.values[0]),
        where=lam != 0,
    )


"""
iterAVaRs(d,Lam) -> np.ndarray: Iterate over sorted element of the distribution.
Takes in a distribution (d) and an array of risk level (Lam).
Return an array of (Lam)-CVaR of the distribution (d).
"""


def iterCVaRs(d: pd.DataFrame, Lam: np.ndarray) -> np.ndarray:
    M, N = len(Lam), len(d.index)

    # initialize answer array
    output = np.empty(M)

    j = np.searchsorted(Lam, 0, side="right")
    output[:j] = d.X[0]

    for i in range(N):
        while (j < M) and (Lam[j] <= d.cdf[i]):
            output[j] = (d.XTP[i] + d.X[i] * (Lam[j] - d.cdf[i])) / Lam[j]
            j += 1

    output[j:] = d.XTP[N - 1]

    return output


"""
AVaRs(d,Lam,mode=1) -> np.ndarray: General function that takes mode as input.
Takes in a distribution (d) and an array of risk level (Lam).
mode = 0 # iterative methods good for large Lam array
mode != 0 # search methods good for small Lam array
Return an array of (Lam)-CVaR of the distribution (d).
"""


def CVaR(d: pd.DataFrame, Lam: np.ndarray, mode: int = 1) -> np.ndarray:
    Lam = bs.npfy(Lam)

    # iterative methods good for large Lam array
    if mode == 0:
        assert bs.is_sorted(Lam), "Lambda array must be sorted"
        return iterCVaRs(d, Lam)
    # search methods good for small Lam array
    return searchCVaR(d, Lam)


"""
E(X,prob): takes in random variables X and optional probability prob
and returns the expected value of the distribution.
"""


def E(d: pd.DataFrame):
    return rm.E(d.X.values, d.p.values)  # type: ignore


"""
min(X,prob): takes in random variables X and optional probability prob
and returns the minimum of X that is possible to occur.
"""


def min(d: pd.DataFrame):
    return rm.min(d.X.values, d.p.values)  # type: ignore


"""
ERM(X,Alpha,prob) = LSE(-alpha*X,b=prob)/-alpha
LSE(a,b)  = np.log(np.sum(b*np.exp(a)))
LSE trick = np.log(np.sum(b*np.exp(a-C)))+C # where #C = a.max()
"""


def ERM(d: pd.DataFrame, Alpha: np.ndarray):
    return rm.ERM(d.X.values, Alpha, d.p.values)  # type: ignore


"""
EVaR(X,Lam,prob) = sup_a( ERM(X,a,prob) + log(lam)/a )
                 = sup_a( (- LSE(-a*X,b=prob) + log(lam))/a )
                 = sup_t( (- LSE(-X/t,b=prob) + log(lam))*t )
                 = - inf_t( (LSE(-X/t,b=prob) - log(lam))*t )
"""


def EVaR(d: pd.DataFrame, Lam: np.ndarray):
    return rm.EVaR(d.X.values, Lam, d.p.values)  # type: ignore


"""
EVaRaprx(X,Lam,prob) = sup_a( ERM(X,a,prob) + log(lam)/a )
"""


def EVaRaprx(d: pd.DataFrame, Lam: np.ndarray, Beta: np.ndarray = np.empty(0)):
    return rm.EVaRaprx(d.X.values, Lam, d.p.values, Beta)  # type: ignore
