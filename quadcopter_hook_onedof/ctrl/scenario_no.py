import numpy as np
import matplotlib.pyplot as plt
import scipy.special

k = 1
b = 1e-6


def eps(N, k=1):
    if k == N:
        return 1
    else:
        # temp = (N * scipy.special.binom(N, k))
        temp = (N * N)
        return 1 - (b / temp)**(1/(N-k))


# res = np.zeros(500)
# for i in range(500):
#     res[i] = eps(500, i)
# plt.figure()
# plt.plot(res)
# plt.show()
print(eps(35000, 1)*100)
