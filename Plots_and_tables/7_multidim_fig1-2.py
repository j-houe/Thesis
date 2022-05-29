# Import packages
import matplotlib.pyplot as plt
# Import implementations in folder
from approximators import nn_approximator
from generators import Bachelier
from plotters import plot_comparison


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
nPaths_list = [2**8, 2**10, 2**13]

# Misc.
seed = 2
plot_seed = 1
nStocks = 20
asset_idx = 0
vols = None # if None then simulated uniformly before normalization
weights = None # if None then simulated uniformly before normalization

# Initialization and training
w, h = (7, 7)
price_fig, price_axs = plt.subplots(len(nPaths_list), 2, sharey=True, dpi=150, figsize=(w, h))
diff_fig, diff_axs = plt.subplots(len(nPaths_list), 2, sharey=True, dpi=150, figsize=(w, h))
price_axs[0, 0].set_title('Standard ANN')
price_axs[0, 1].set_title('Twin Net')
diff_axs[0, 0].set_title('Standard ANN')
diff_axs[0, 1].set_title('Twin Net')
price_fig.supylabel('Option Value')
diff_fig.supylabel('Option Delta')
for i, nPaths in enumerate(nPaths_list):
    batch_size = int(nPaths / n_batches)
    # Generate data
    generator = Bachelier(sigmaAdj=sigmaAdj, nStocks=nStocks)
    x_raw, y_raw, xbar_raw = generator.sim_paths(nPaths, seed=seed, antithetic=antithetic, vols=vols, w=weights)
    # Fit models
    approximator = nn_approximator(x_raw, y_raw, xbar_raw)
    approximator.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=True)
    approximator.train(epochs=epochs,
                       batch_size=batch_size,
                       learning_rate=learning_rate,
                       lam=lam,
                       schedule_lr=schedule_lr,
                       verbose=False)
    approximator_vanilla = nn_approximator(x_raw, y_raw, xbar_raw)
    approximator_vanilla.prepare(n_units=n_units, n_hidden_layers=n_hidden_layers, seed=seed, differential=False)
    approximator_vanilla.train(epochs=epochs,
                               batch_size=batch_size,
                               learning_rate=learning_rate,
                               lam=lam,
                               schedule_lr=schedule_lr,
                               verbose=False)
    # Plot
    price_handles, diff_handles = plot_comparison(price_axs[i, :],
                                                  diff_axs[i, :],
                                                  approximator_vanilla,
                                                  approximator,
                                                  generator,
                                                  lower=0.35,
                                                  upper=2.0,
                                                  seed=plot_seed,
                                                  asset_idx=asset_idx,
                                                  num=2**11,
                                                  plot_sims=False)

price_fig.tight_layout()
price_fig.suptitle('Bachelier dimension {} - Price'.format(nStocks), size=15)
price_fig.supxlabel('Basket Spot', x=0.5, y=0.07)
price_fig.subplots_adjust(bottom=0.125, top=0.9)
price_fig.legend(handles=price_handles, ncol=3, loc='lower center', bbox_to_anchor=[0.5, 0])

diff_fig.tight_layout()
diff_fig.suptitle('Bachelier dimension {} - Delta'.format(nStocks), size=15)
diff_fig.supxlabel('Basket Spot', x=0.5, y=0.07)
diff_fig.subplots_adjust(bottom=0.125, top=0.9)
diff_fig.legend(handles=diff_handles, ncol=3, loc='lower center', bbox_to_anchor=[0.5, 0])


plt.figure(price_fig.number)
plt.show()

plt.figure(diff_fig.number)
plt.show()
