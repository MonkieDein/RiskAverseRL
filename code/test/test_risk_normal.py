import matplotlib.pyplot as plt
import numpy as np

import riskMeasure as rm

X = np.random.normal(loc=0, scale=1, size=10000)
Lambda = np.linspace(0, 1, 101)

var = rm.VaR(X, Lambda, mode=1)
cvar = rm.CVaR(X, Lambda, mode=1)
evar = rm.EVaR(X, Lambda)

plt.plot(Lambda, var, 'b-.', label="var")
plt.plot(Lambda, cvar, 'g--', label="cvar")
plt.plot(Lambda, evar, 'r-', label="evar")
plt.title("Risk of Rewards for Normal Distribution")
plt.ylabel('risk of reward')
plt.xlabel('significant level')
plt.legend(loc="upper left")
plt.show()
