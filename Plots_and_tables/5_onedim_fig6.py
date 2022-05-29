# import packages
import matplotlib.pyplot as plt
import seaborn as sns
# import local implementations
from gridsearch import GridSearch
from generators import BlackScholes

# HPs
onecycle = [True]
epochs = [50]
n_stocks = [1]
n_paths = [1024]
n_batches = [8]
n_hidden_layers = [2, 4, 8]
n_units = [5, 10, 20, 40]
sigma_adj = [1.5]

# Gridsearch
n_seeds = 20
BS_generator = BlackScholes()
BS_gridsearch = GridSearch(n_seeds, BS_generator, differential=True)
BS_gridsearch.prepare(onecycle, epochs, n_stocks, n_paths, n_batches, n_hidden_layers, n_units, sigma_adj)
BS_gridsearch.begin()
HP_table = BS_gridsearch.HP_table

# Plot
fontsize = 11
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(7, 4), dpi=150)
price_axs = sns.lineplot(data=HP_table, x='n_units', y='price_mean',
                         hue='n_hidden_layers', palette='Set2', ax=axs[0], legend=False, lw=2.5, marker='o')
delta_axs = sns.lineplot(data=HP_table, x='n_units', y='delta_mean',
                         hue='n_hidden_layers', palette='Set2', ax=axs[1], legend=False, lw=2.5, marker='o')
delta_axs.set_xlabel(r'Width ($n_l$)', fontsize=fontsize)
delta_axs.set_ylabel('')
delta_axs.set_title('Delta')
price_axs.set_xlabel(r'Width ($n_l$)', fontsize=fontsize)
price_axs.set_ylabel(r'Mean MSE($\times 1000$)', fontsize=fontsize)
price_axs.set_title('Price')
fig.suptitle('Analysis of Depth and Width', fontsize=fontsize*1.25)
fig.legend(labels=n_hidden_layers, ncol=len(n_hidden_layers), loc='lower center', fontsize=fontsize,
           title=r'Depth ($L$)')
fig.tight_layout()
fig.subplots_adjust(bottom=0.3)
plt.show()
