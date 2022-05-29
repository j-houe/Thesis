# import packages
import numpy as np
import pandas as pd
# import local implementations
from generators import BlackScholes, Generator
from approximators import nn_approximator
from plotters import mse_test

# Hyper parameters
epochs = 100
n_batches = 8
antithetic = True
nPaths = 1024
# Misc.
batch_size = int(nPaths / n_batches)


fixed_lr = pd.DataFrame({'alpha': [0.0005, 0.001, 0.005, 0.01]})
fixed_lr['one_cycle'] = False
onecycle_lr = pd.DataFrame({'alpha': [0.003, 0.006, 0.03, 0.06]})
onecycle_lr['one_cycle'] = True
lr_table = pd.concat([fixed_lr, onecycle_lr], ignore_index=True)
seeds = np.arange(1, 21)
for i, row in lr_table.iterrows():
    print('Training with alpha={} and onecycle={}'.format(row['alpha'], row['one_cycle']))
    price_mses = []
    delta_mses = []
    for seed in seeds:
        BS_generator = BlackScholes()
        x, y, xbar = BS_generator.sim_paths(nPaths, seed)
        twin_approximator = nn_approximator(x, y, xbar)
        twin_approximator.prepare(n_hidden_layers=4, n_units=20, differential=True, seed=seed)
        twin_approximator.train(epochs, batch_size, row['alpha'], schedule_lr=row['one_cycle'], silent=True)
        price_mse, delta_mse = mse_test(twin_approximator, BS_generator)
        price_mses.append(price_mse)
        delta_mses.append(delta_mse)
    price_array = np.array(price_mses)
    delta_array = np.array(delta_mses)
    lr_table.at[i, 'price_mean'] = price_array.mean()
    lr_table.at[i, 'price_sd'] = price_array.std()
    lr_table.at[i, 'delta_mean'] = delta_array.mean()
    lr_table.at[i, 'delta_sd'] = delta_array.std()
formats = [None]*2 + [lambda x: '%.3f' % x]*4
print(lr_table.to_latex(index=False, formatters=formats))
