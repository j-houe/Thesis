import numpy as np
import matplotlib.pyplot as plt
from generators import Bachelier

##########################
# Plots for introduction #
##########################
S0 = 1
sigma = 0.2
K = 1
n = 8
steps = 200
maturity = 1
step_size = maturity/steps
text_size = 20
# Point-wise
np.random.seed(seed=1)
normals = np.random.normal(scale=sigma*np.sqrt(step_size), size=([n, steps+1]))
normals[:, 0] = S0
paths = normals.cumsum(axis=1)
x = np.linspace(0, 1, steps+1)
for i in range(n):
    plt.plot(x, paths[i, :], alpha=0.6)
plt.scatter(0, S0, color='grey', edgecolors='grey', linewidths=1, s=50)
plt.scatter(np.repeat(maturity, n), paths[:, -1], color='lightgrey', edgecolors='grey', linewidths=1, s=50)
plt.xticks([0, maturity], ['0', r'$T$'], size=text_size)
plt.yticks([], [])
plt.xlabel(r'$t$', size=text_size)
plt.ylabel(r'$S_t$', size=text_size)
plt.tight_layout()
plt.show()
# Shape
np.random.seed(439)
normals = np.random.normal(scale=sigma*np.sqrt(step_size), size=[n, steps+1])
normals[:, 0] = np.random.normal(scale=sigma*np.sqrt(maturity), size=[n])
paths = normals.cumsum(axis=1)
x = np.linspace(0, 1, steps+1)
for i in range(n):
    plt.plot(x, paths[i, :], alpha=0.6)
plt.scatter(np.repeat(0, n), paths[:, 1], color='lightgrey', edgecolors='grey', linewidths=1, s=50)
plt.scatter(np.repeat(maturity, n), paths[:, -1], color='lightgrey', edgecolors='grey', linewidths=1, s=50)
plt.xticks([0, maturity], ['0', r'$T$'], size=text_size)
plt.yticks([], [])
plt.xlabel(r'$t$', size=text_size)
plt.ylabel(r'$S_t$', size=text_size)
plt.tight_layout()
plt.show()
# Shape estimation
generator = Bachelier(basketSigma=sigma, K=1, S0=S0, T2=1+maturity)
x, y, xbar = generator.sim_paths(1024, antithetic=False, seed=1)
true_x = np.linspace(0.35, 2.0, 100)
true_y = generator.true_price(true_x)
plt.scatter(x, y, color='lightgrey', edgecolors='grey', linewidths=1, s=50, alpha=0.5)
plt.plot(true_x, true_y, lw=4)
plt.xlim(0.35, 2.0)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel(r'$S_0$', size=text_size)
plt.ylabel(r'$(S_T-K)^+$', size=text_size)
plt.annotate(r'$\pi$', xy=(0.80, 0.75), fontsize=text_size*1.5, color='C0',
             xycoords='figure fraction', horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.subplots_adjust(bottom=0.175)
plt.show()
