# import packages
import matplotlib.pyplot as plt
# import local implementations
from gridsearch import GridSearch
from generators import Bachelier

# HPs
onecycle = [True]
epochs = [50]
n_stocks = [1, 10, 20, 50, 100]
n_paths = [1024]
n_batches = [8]
n_hidden_layers = [4]
n_units = [10]
sigma_adj = [1.5]

# Gridsearch
n_seeds = 20
BA_generator = Bachelier()
# Twin
BA_gridsearch = GridSearch(n_seeds, BA_generator, differential=True, PCA=False)
BA_gridsearch.prepare(onecycle, epochs, n_stocks, n_paths, n_batches, n_hidden_layers, n_units, sigma_adj)
BA_gridsearch.begin(n_pred_points=20)
HP_table = BA_gridsearch.HP_table
# Standard
BA_gridsearch_std = GridSearch(n_seeds, BA_generator, differential=False, PCA=False)
BA_gridsearch_std.prepare(onecycle, epochs, n_stocks, n_paths, n_batches, n_hidden_layers, n_units, sigma_adj)
BA_gridsearch_std.begin(n_pred_points=20)
HP_table_std = BA_gridsearch_std.HP_table

# Plot
fontsize = 12
fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6, 4), dpi=150)
axs.plot('n_stocks', 'price_mean', data=HP_table, label='Twin Net',
            color='dodgerblue', marker='o', markeredgecolor='white', markersize=8, lw=1.5)
axs.plot('n_stocks', 'price_mean', data=HP_table_std, label='Standard ANN',
            color='salmon', marker='o', markeredgecolor='white', markersize=8, lw=1.5)
axs.set_xlabel(r'Number of Assets $(p)$', fontsize=fontsize)
axs.set_ylabel(r'Mean Price MSE($\times 1000$)', fontsize=fontsize)
axs.legend(fontsize=fontsize*0.8)
fig.suptitle('Increasing Dimensionality', fontsize=fontsize*1.25)
fig.tight_layout()
plt.show()
