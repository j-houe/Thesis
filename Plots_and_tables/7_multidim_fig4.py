# Import packages
import matplotlib.pyplot as plt
# Import implementations in folder
from approximators import nn_approximator
from generators import Bachelier
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
nPaths = 2**10
# Misc.
seed = 2
plot_seed = 1
nStocks = 100
asset_idx = 0
vols = None # if None then simulated uniformly before normalization
weights = None # if None then simulated uniformly normalization

# Initialization and training
batch_size = int(nPaths / n_batches)
# Generate data
generator = Bachelier(sigmaAdj=sigmaAdj, nStocks=nStocks)
x_raw, y_raw, xbar_raw = generator.sim_paths(nPaths, seed=seed, antithetic=antithetic, vols=vols, w=weights)
# Fit models
approximator = nn_approximator(x_raw, y_raw, xbar_raw)
approximator.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=True, PCA=True, silent=False)
approximator.train(epochs=epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   lam=lam,
                   schedule_lr=schedule_lr,
                   verbose=False)
approximator_vanilla = nn_approximator(x_raw, y_raw, xbar_raw)
approximator_vanilla.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=True, PCA=False, silent=False)
approximator_vanilla.train(epochs=epochs,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           lam=lam,
                           schedule_lr=schedule_lr,
                           verbose=False)
# Plot
fig, axs = plt.subplots(1, 2, dpi=150, figsize=(6, 3.5))
price_handle = stacked_comparison(axs[0], approximator_vanilla, approximator, generator,
                             type='Price', labels=['Excl PCA', 'Incl PCA'], num=2048, seed=seed, plot_sims=False, scatter=True)
delta_handle = stacked_comparison(axs[1], approximator_vanilla, approximator, generator,
                             type='Delta', labels=['Excl PCA', 'Incl PCA'], num=2048, seed=seed, plot_sims=False, scatter=True)
fig.tight_layout()
fig.subplots_adjust(bottom=0.275, top=0.8)
fig.legend(handles=delta_handle, ncol=4, loc='lower center', bbox_to_anchor=[0.5, 0], fontsize=10)
fig.suptitle('Bachelier Dimension {} - Twin Net with Differential PCA'.format(generator.nStocks))
plt.show()
print(approximator.PCA_time)
