import numpy as np


def VaR(d, Lam, Average=True):
    M, N = len(Lam), len(d.index)

    # initialize answer array
    output = np.zeros(M)

    j = 0

    for i in range(N):
        while (j < M) and (Lam[j] <= d.cdf[i]):
            output[j] = d.X[i] if (i == 0 or not Average) else (
                d.XTP[i] + d.X[i]*(Lam[j] - d.cdf[i]))/Lam[j]
            j += 1

    if (j < M):
        output[j:] = d.X[N-1] if Average else d.XTP[N-1]

    return output
