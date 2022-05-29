# import packages
import matplotlib.pyplot as plt
import matplotlib
# import local implementations
from gridsearch import GridSearch
from generators import BlackScholes

# HPs
onecycle = [False, True]
epochs = [25, 50, 75, 100]
n_stocks = [1]
n_paths = [1024]
n_batches = [8]
n_hidden_layers = [4]
n_units = [20]
sigma_adj = [1.5]

# Gridsearch
n_seeds = 20
BS_generator = BlackScholes()
BS_gridsearch = GridSearch(n_seeds, BS_generator, differential=True)
BS_gridsearch.prepare(onecycle, epochs, n_stocks, n_paths, n_batches, n_hidden_layers, n_units, sigma_adj)
BS_gridsearch.begin()
HP_table = BS_gridsearch.HP_table

# Plot
fig, ax1 = plt.subplots(1, 1, figsize=(7, 4), dpi=150)
ax2 = ax1.twinx()
ax1.plot('epochs', 'price_mean', data=HP_table[HP_table['onecycle'] == True],
         color='lightsalmon', marker='o', markeredgecolor='white', lw=2, ls='--')
ax1.plot('epochs', 'price_mean', data=HP_table[HP_table['onecycle'] == False],
         color='lightsalmon', marker='o', markeredgecolor='white', lw=2, ls='-')
ax2.plot('epochs', 'delta_mean', data=HP_table[HP_table['onecycle'] == True],
         color='dodgerblue', marker='o', markeredgecolor='white', lw=2, ls='--')
ax2.plot('epochs', 'delta_mean', data=HP_table[HP_table['onecycle'] == False],
         color='dodgerblue', marker='o', markeredgecolor='white', lw=2, ls='-')
ax1.set_xlabel('Epochs', size=13)
ax1.set_ylabel(r'Mean Price MSE$(\times 1000)$', size=13)
ax1.yaxis.label.set_color('salmon')
ax1.tick_params(axis='y', colors='salmon')
ax2.set_ylabel(r'Mean Delta MSE$(\times 1000)$', size=13)
ax2.yaxis.label.set_color('dodgerblue')
ax2.tick_params(axis='y', colors='dodgerblue')
ax2.set_xticks([25, 50, 75, 100])
legend_elements = [matplotlib.lines.Line2D([0], [0], color='lightsalmon', label='Price'),
                   matplotlib.lines.Line2D([0], [0], color='dodgerblue', label='Delta'),
                   matplotlib.lines.Line2D([0], [0], color='black', ls='--', label='1cycle'),
                   matplotlib.lines.Line2D([0], [0], color='black', ls='-', label='Fixed LR')]
ax1.legend(handles=legend_elements, loc='upper right')
plt.title('Fixed Learning Rate vs. 1cycle Convergence', size=15)
plt.tight_layout()
plt.show()
