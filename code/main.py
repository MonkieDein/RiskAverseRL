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
cvar, var = CVaR(d, Lambda), VaR(d, Lambda)
