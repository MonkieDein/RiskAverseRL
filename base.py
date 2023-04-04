import numpy as np
import pandas as pd

'''
This document contains the basic functions to be used in multi module of the project.
'''


'''
npfy(var1) -> pass in a variable X and return a numpy array of variable X.
'''


def npfy(var1):
    if type(var1) is not np.ndarray:
        if type(var1) is not list:
            return np.array([var1])
        else:
            return np.array(var1)
    return var1


'''
is_sorted(a) -> Boolean
Takes in an array (a): identify if the array is sorted. 
'''
def is_sorted(a: np.ndarray): return np.all(np.diff(a) >= 0)


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
        X = X[prob > 0]  # type: ignore
        prob = prob[prob > 0]
    return X, prob
