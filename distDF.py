
import numpy as np
import pandas as pd

'''
preProcess(d) -> pd.DataFrame
Takes in a distribution dataframe (d):
1. remove probabilities p <= 0.
2. sorts the distribution dataframe by its value X
3. precompute cdf and XTP
returns the sorted distribution dataframe.
'''


def preProcess(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.p > 0]
    d.sort_values('X', inplace=True)

    # Normalization of pdf
    if abs(d.p.sum() - 1.0) > 1e-13:
        print("Distribution does not sum to one")
    d.p /= d.p.sum()
    probSum = d.p.sum()
    if probSum < 1:
        d.p = d.p/probSum

    # precompute repeatedly used values
    d['cdf'] = np.cumsum(d.p)
    d['XTP'] = np.cumsum((d.p * d.X))

    return d


'''
distribution(X,p) -> pd.DataFrame
Takes in an array of X values its respective probability p:
1. Creates a distribution dataframe (d) from X and p.
2. group and combine probabilities p with identical X values.
returns the preprocess (sorted and precompute cdf and XTP) distribution dataframe (d).
'''


def distribution(X: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({'X': X, 'p': p})
    d = d.groupby('X').agg({'p': sum}).reset_index()
    return preProcess(d)


'''
condD(X,p_cond,pr,**kwargs) -> pd.DataFrame
Takes in an array of conditional values (X), its respective conditional probability (p_cond),
marginal probability (pr), and optional arbitrary input (**kwargs):
1. Creates a conditional distribution dataframe (d) with X and p_cond.
2. Rename the columns of the dataframe to match the conditional distribution dataframe.
3. Calculate the joint probability of X.
4. For each arbitary inputs, match its key with its corresponding value.
returns the conditional distribution dataFrame.
'''


def condD(X: np.ndarray, p_cond: np.ndarray, pr: float, **kwargs):
    d = distribution(X, p_cond)
    d.rename(columns={'p': 'p_cond', 'cdf': 'cdf_cond', 'XTP': 'XTP_cond'})
    d['p'] = d.p_cond * pr
    for key, value in kwargs.items():
        d[key] = value
    return d


'''
joinD(d_conds) -> pd.DataFrame
Takes in a list of conditional distribution (d_conds):
1. Combines the conditional distribution dataframe (d_conds) into the joint distribution dataframe (d).
returns the preprocess (sorted and precompute cdf and XTP) distribution dataframe (d).
'''


def joinD(d_conds: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat(d_conds)
    return preProcess(d)
