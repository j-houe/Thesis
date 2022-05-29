# import packages
import numpy as np
import pandas as pd
import itertools
# import local implementations
from generators import Generator
from approximators import nn_approximator
from plotters import mse_test


class GridSearch:
    def __init__(self, n_seeds, generator: Generator, differential=True, PCA=False):
        self.differential = differential
        self.PCA = PCA
        self.generator = generator
        self.n_seeds = n_seeds

    def prepare(self, onecycle, epochs, n_stocks, n_paths, n_batches, n_hidden_layers, n_units, sigma_adj):
        self.HP_grid = list(itertools.product(onecycle,
                                              epochs,
                                              n_stocks,
                                              n_paths,
                                              n_batches,
                                              n_hidden_layers,
                                              n_units,
                                              sigma_adj))
        self.HP_table = pd.DataFrame(data=self.HP_grid,
                                    columns=['onecycle',
                                    'epochs',
                                    'n_stocks',
                                    'n_paths',
                                    'n_batches',
                                    'n_hidden_layers',
                                    'n_units',
                                    'sigma_adj'])
        self.n_permutations = len(self.HP_table.index)

    def begin(self, n_pred_points=100):
        seeds = np.arange(1, self.n_seeds + 1)
        for i, row in self.HP_table.iterrows():
            print('Permutation {} of {}: '
                  'onecycle={}, '
                  'Epochs={}, '
                  'n_stocks={}, '
                  'n_paths={}, '
                  'n_batches={}, '
                  'n_hidden_layers={}, '
                  'n_units={}, '
                  'sigma_adj={}'.format(i + 1,
                                      self.n_permutations,
                                      row['onecycle'],
                                      row['epochs'],
                                      row['n_stocks'],
                                      row['n_paths'],
                                      row['n_batches'],
                                      row['n_hidden_layers'],
                                      row['n_units'],
                                      row['sigma_adj']))
            batch_size = int(row['n_paths'] / row['n_batches'])
            if row['onecycle']:
                learning_rate = 0.03
            else:
                learning_rate = 0.005
            price_mses = []
            delta_mses = []
            training_times = []
            for seed in seeds:
                self.generator.update_sigmaAdj(row['sigma_adj'])
                self.generator.update_nStocks(row['n_stocks'])
                x, y, xbar = self.generator.sim_paths(row['n_paths'], seed=seed)
                approximator = nn_approximator(x, y, xbar)
                approximator.prepare(n_hidden_layers=int(row['n_hidden_layers']),
                                     n_units=int(row['n_units']),
                                     differential=self.differential,
                                     PCA=self.PCA,
                                     seed=seed)
                approximator.train(int(row['epochs']),
                                   batch_size,
                                   learning_rate,
                                   schedule_lr=row['onecycle'],
                                   silent=True)
                price_mse, delta_mse = mse_test(approximator, self.generator, num=n_pred_points)
                price_mses.append(price_mse)
                delta_mses.append(delta_mse)
                training_times.append(approximator.training_time)
            price_array = np.array(price_mses)
            delta_array = np.array(delta_mses)
            training_times_array = np.array(training_times)
            self.HP_table.at[i, 'training_time'] = training_times_array.mean()
            self.HP_table.at[i, 'price_mean'] = price_array.mean()
            self.HP_table.at[i, 'price_sd'] = price_array.std()
            self.HP_table.at[i, 'delta_mean'] = delta_array.mean()
            self.HP_table.at[i, 'delta_sd'] = delta_array.std()
        print('Grid search done!')

    def print_tex_table(self):
        formats = [None] * len(self.HP_grid[0]) + [lambda x: '%.3f' % x] * 5
        print(self.HP_table.to_latex(index=False, formatters=formats))
