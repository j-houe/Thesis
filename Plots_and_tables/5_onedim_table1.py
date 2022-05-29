# Import packages
import itertools
import pandas as pd
import numpy as np
# Import implementations in folder
from approximators import nn_approximator, poly_approximator
from generators import BlackScholes
from plotters import mse_test

# Hyper parameters
sigmaAdj = 1.5
epochs = 100
n_batches = 8
antithetic = True
schedule_lr = True
learning_rate = 3.0e-2 # If schedule_Lr=True, this is the maximum lr obtained in the cycle
lam = 1
nPaths = 1024
poly_degree = 7
# Misc.
seed = 4
batch_size = int(nPaths / n_batches)

# BS parameters
T = [0.5, 1.0, 2.0]
K = [0.9, 1.1]
sigma = [0.2, 0.4]
BS_params = list(itertools.product(T, K, sigma))
BS_table = pd.DataFrame(data=BS_params, columns=['T', 'K', 'sigma'])
BS_table['Poly Price'] = np.nan
BS_table['ANN Price'] = np.nan
BS_table['Twin Price'] = np.nan
BS_table['Poly Delta'] = np.nan
BS_table['ANN Delta'] = np.nan
BS_table['Twin Delta'] = np.nan

# Train and fill out table
for i, row in BS_table.iterrows():
    # Generate data
    BS_generator = BlackScholes(sigmaAdj=sigmaAdj,
                                sigma=row['sigma'],
                                K=row['K'],
                                T2=1+row['T'])
    print('Training with sigma={}, K={} and T={}'.format(row['sigma'], row['K'], row['T']))
    x_raw, y_raw, xbar_raw = BS_generator.sim_paths(nPaths, seed, antithetic=antithetic)
    # Fit models
    twin = nn_approximator(x_raw, y_raw, xbar_raw)
    twin.prepare(seed=seed, differential=True)
    twin.train(epochs=epochs,
               batch_size=batch_size,
               learning_rate=learning_rate,
               lam=lam,
               schedule_lr=schedule_lr,
               silent=True)
    ann = nn_approximator(x_raw, y_raw, xbar_raw)
    ann.prepare(seed=seed, differential=False)
    ann.train( epochs=epochs,
               batch_size=batch_size,
               learning_rate=learning_rate,
               lam=lam,
               schedule_lr=schedule_lr,
               silent=True)
    poly = poly_approximator(x_raw, y_raw, xbar_raw)
    poly.prepare(poly_degree, differential=True)
    poly.train(lam=lam, silent=True)
    # Inference
    spots, pred_input, y_true, delta_true = BS_generator.test_set(lower=0.35, upper=2, num=100, seed=seed)
    row['Twin Price'], row['Twin Delta'] = mse_test(twin, BS_generator)
    row['ANN Price'], row['ANN Delta'] = mse_test(ann, BS_generator)
    row['Poly Price'], row['Poly Delta'] = mse_test(poly, BS_generator)
formats = [None]*3 + [lambda x: '%.3f' % x]*6
print(BS_table.to_latex(index=False, formatters=formats))
