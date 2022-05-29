# Import packages
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
# Import implementations in folder
from approximators import nn_approximator
from generators import BlackScholes
from plotters import plot_comparison


# Hyper parameters
sigmaAdj = 1.5
epochs = 100
n_batches = 8
n_hidden_layers = 4
n_units = 20
antithetic = True
schedule_lr = True
learning_rate = 3.0e-2 # If schedule_Lr=True, this is the maximum lr obtained in the cycle
lam = 1
nPaths_list = [256, 1024]
# Misc.
seed = 4

# Initialization and training
w, h = figaspect(0.65)
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
    BS_generator = BlackScholes(sigmaAdj=sigmaAdj)
    x_raw, y_raw, xbar_raw = BS_generator.sim_paths(nPaths, seed, antithetic=antithetic)
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
                                                  BS_generator,
                                                  lower=0.35,
                                                  upper=2.0)

price_fig.tight_layout()
price_fig.supxlabel('Spot', x=0.5, y=0.1)
price_fig.subplots_adjust(bottom=0.2)
price_fig.legend(handles=price_handles, ncol=3, loc='lower center', bbox_to_anchor=[0.5, 0])

diff_fig.tight_layout()
diff_fig.supxlabel('Spot', x=0.5, y=0.1)
diff_fig.subplots_adjust(bottom=0.2)
diff_fig.legend(handles=diff_handles, ncol=3, loc='lower center', bbox_to_anchor=[0.5, 0])

plt.figure(price_fig.number)
plt.show()

plt.figure(diff_fig.number)
plt.show()
