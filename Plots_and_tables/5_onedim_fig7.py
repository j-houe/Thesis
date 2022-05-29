# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
# Import implementations in folder
from approximators import nn_approximator
from generators import SABR
from payoffs import Call, Digital
from plotters import stacked_comparison


# Hyper parameters
sigmaAdj = 1.5
epochs = 50
n_batches = 8
n_hidden_layers = 4
n_units = 10
antithetic = True
schedule_lr = True
learning_rate = 3.0e-2 # If schedule_Lr=True, this is the maximum lr obtained in the cycle
lam = 1
nPaths = 2**12


# Misc.
S0 = 1
K = 1.1
sigma0 = 0.2
T = 1
plot_lower = 0.35
plot_upper = 2.0
plot_num_true = 100
seed = 2
plot_seed = 1
alpha = 1
beta = 0.5
rho = 0.75
payoff = Digital()


# Generate Training Set
batch_size = int(nPaths / n_batches)
generator = SABR(payoff,
                 S0=S0,
                 K=K,
                 sigma0=sigma0,
                 sigmaAdj=sigmaAdj,
                 T2=1+T,
                 alpha=alpha,
                 beta=beta,
                 rho=rho)
x_raw, y_raw, xbar_raw = generator.sim_paths(nPaths, nSteps=100, seed=seed, antithetic=antithetic)

# Fit models
approximator = nn_approximator(x_raw, y_raw, xbar_raw)
approximator.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=True, silent=False)
approximator.train(epochs=epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   lam=lam,
                   schedule_lr=schedule_lr,
                   verbose=False)
approximator_vanilla = nn_approximator(x_raw, y_raw, xbar_raw)
approximator_vanilla.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=False, silent=False)
approximator_vanilla.train(epochs=epochs,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           lam=lam,
                           schedule_lr=schedule_lr,
                           verbose=False)

# Plot
fig, axs = plt.subplots(1, 2, dpi=150, figsize=(6, 3))

# def BS_digital_price(S0, sigma=0.2, T=1, K=1):
#     vol = sigma*np.sqrt(T)
#     drift = 0.5*sigma**2
#     d = (np.log(S0/K) - drift*T)/vol
#     return norm.cdf(d)
#
# def BS_digital_delta(S0, sigma=0.2, T=1, K=1):
#     vol = sigma*np.sqrt(T)
#     drift = 0.5*sigma**2
#     d = (np.log(S0/K) - drift*T)/vol
#     return norm.pdf(d)/(S0*vol)
#
# x = np.linspace(plot_lower, plot_upper, 100)
# axs[0].plot(x, BS_digital_price(x, sigma=sigma0, T=T, K=K), color='green', lw=5, alpha=0.7)
# axs[1].plot(x, BS_digital_delta(x, sigma=sigma0, T=T, K=K), color='green', lw=5, alpha=0.7)

price_handles = stacked_comparison(axs[0], approximator_vanilla, approximator, generator,
                                   labels=['Standard ANN', 'Twin Net'], type='Price', lower=plot_lower, upper=plot_upper)
delta_handles = stacked_comparison(axs[1], approximator_vanilla, approximator, generator,
                                   labels=['Standard ANN', 'Twin Net'], type='Delta', lower=plot_lower, upper=plot_upper)
fig.tight_layout()
fig.subplots_adjust(bottom=0.275, top=0.8)
fig.legend(handles=delta_handles, ncol=4, loc='lower center', bbox_to_anchor=[0.5, 0], fontsize=10)
fig.suptitle('SABR {} Option - Standard ANN vs. Twin Net'.format(payoff.type))
plt.show()
