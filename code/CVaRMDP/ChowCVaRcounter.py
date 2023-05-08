import matplotlib.pyplot as plt
import numpy as np

import distDF as ddf
import riskMeasure as rm

s1a1 = ddf.distribution(X=np.array([-300, 300]), p=np.array([0.25, 0.75]))
s1a2 = ddf.distribution(X=np.array([0]), p=np.array([1.0]))
s1a3 = ddf.distribution(X=np.array([-50, 250]), p=np.array([0.5, 0.5]))
s2 = ddf.distribution(X=np.array([100]), p=np.array([1.0]))
Lambda = np.linspace(0, 1, 1001)

# Get s1 CVaR Values for each action and get the max of the CVaR values
s1a1CVaR = ddf.CVaR(s1a1, Lambda)
s1a2CVaR = ddf.CVaR(s1a2, Lambda)
s1a3CVaR = ddf.CVaR(s1a3, Lambda)
s1maxCVaR = np.maximum.reduce([s1a1CVaR, s1a2CVaR, s1a3CVaR])
s1max = ddf.CVaR2D(s1maxCVaR, Lambda)
# CVaR s1 plot
plt.plot(Lambda, s1maxCVaR, "k--", label="max_a", linewidth=2.5)
plt.plot(Lambda, s1a1CVaR, "r-", label="a1")
plt.plot(Lambda, s1a2CVaR, "b-", label="a2")
plt.plot(Lambda, s1a3CVaR, "g-", label="a3")
plt.title("s1 CVaR value")
plt.ylabel("CVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()

# Joining s1 and s2 distribution
jointExp = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1max, s2]])
jointa1 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a1, s2]])
jointa2 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a2, s2]])
jointa3 = ddf.joinD([ddf.condD(df.X, df.p, 0.5) for df in [s1a3, s2]])
jointExpectedCVaR = ddf.CVaR(jointExp, Lambda)
jointa1CVaR = ddf.CVaR(jointa1, Lambda)
jointa2CVaR = ddf.CVaR(jointa2, Lambda)
jointa3CVaR = ddf.CVaR(jointa3, Lambda)
jointoptCVaR = np.maximum.reduce([jointa1CVaR, jointa2CVaR, jointa3CVaR])
jointDeploy = np.maximum(jointa1CVaR, jointa2CVaR)
# join CVaR plot
plt.plot(Lambda, jointExpectedCVaR, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, jointa1CVaR, "r-", label="a1")
plt.plot(Lambda, jointa2CVaR, "b-", label="a2")
plt.plot(Lambda, jointa3CVaR, "g-", label="a3")
plt.title("join CVaR value")
plt.ylabel("CVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()

# join CVaR plot performance plot
plt.plot(Lambda, jointExpectedCVaR, "k--", label="Expect", linewidth=2.5)
plt.plot(Lambda, jointoptCVaR, "g-", label="True Opt")
plt.plot(Lambda, jointDeploy, "r-.", label="a2")
plt.title("join CVaR performance plot")
plt.ylabel("CVaR Value")
plt.xlabel("α")
plt.legend(loc="upper left")
plt.show()

# Manually derive the theta function for each action
Z = np.linspace(0, 1, 10001)
Za1 = np.maximum(100 - 400 * Z, -50 + 200 * Z)
Za2 = 100 - 100 * Z
Za3 = np.maximum(100 - 150 * Z, -50 + 150 * Z)
Zexp = np.maximum.reduce([Za1, Za2, Za3])
# The maxmin vs minmax optimal values
expI = np.argmin(Zexp)
expVal = Zexp[expI]
trueI = np.argmin(Za3)
trueVal = Za3[trueI]
# Plot the theta functions
plt.plot(Z, Zexp, "k--", label="Expect", linewidth=2.5)
plt.plot(Z, Za1, "r-", label="a1")
plt.plot(Z, Za2, "b-", label="a2")
plt.plot(Z, Za3, "g-", label="a3")
plt.plot([Z[expI]], [expVal], marker="o", color="purple")
plt.plot([Z[trueI]], [trueVal], marker="o", color="darkgreen")
plt.title("θπ(ζ₁) plot for CVaR 0.5")
plt.ylabel("θπ(ζ₁)")
plt.xlabel("ζ₁")
plt.legend(loc="upper left")
plt.show()
