# Import packages
import matplotlib.pyplot as plt
# Import implementations in folder
from approximators import nn_approximator, poly_approximator
from generators import BlackScholes
from plotters import stacked_comparison


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
seed = 248
differential = True

# Generate data
BS_generator = BlackScholes(sigmaAdj=sigmaAdj, T2=13/12)
x_raw, y_raw, xbar_raw = BS_generator.sim_paths(nPaths, seed, antithetic=antithetic)
# Fit models
ann = nn_approximator(x_raw, y_raw, xbar_raw)
ann.prepare(seed=seed, differential=differential)
ann.train(epochs=epochs,
          batch_size=int(nPaths/n_batches),
          learning_rate=learning_rate,
          lam=lam,
          schedule_lr=schedule_lr,
          verbose=False)
poly = poly_approximator(x_raw, y_raw, xbar_raw)
poly.prepare(poly_degree, differential=False)
poly.train(lam=lam)
# Plot
fig, axs = plt.subplots(1, 2, dpi=150, figsize=(6, 3.5))
price_handles = stacked_comparison(axs[0], poly, ann, BS_generator,
                                   labels=['Polynomial', 'Twin Net'], type='Price', lower=0.35, upper=2)
delta_handles = stacked_comparison(axs[1], poly, ann, BS_generator,
                                   labels=['Polynomial', 'Twin Net'], type='Delta', lower=0.35, upper=2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.275, top=0.8)
fig.legend(handles=delta_handles, ncol=4, loc='lower center', bbox_to_anchor=[0.5, 0], fontsize=10)
fig.suptitle('Polynomial Regression vs. Twin Net')
plt.show()
