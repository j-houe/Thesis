# Import packages
import numpy as np
import matplotlib.pyplot as plt
# Import local implementations
from diffPCA import DiffPCA
from generators import Bachelier

# Generate Bachelier paths
BA_generator = Bachelier(nStocks=2)
X, _, _ = BA_generator.sim_paths(40, seed=2, corr=np.array([[1, 0.95], [0.95, 1]]), vols=np.array([1, 1]))
y = BA_generator.S2[:, 1] - BA_generator.S2[:, 0]
Z = np.repeat([[-1, 1]], len(y), axis=0)

# Initial normalization
X_mean = X.mean(axis=0)
X_sd = X.std(axis=0) + 1.0e-8
X = (X - X_mean) / X_sd
y_mean = y.mean()
y_sd = y.std()
y = (y - y_mean) / y_sd
Z = Z / y_sd

# PCA step and plot
fig, axs = plt.subplots(2, 2, sharey=True, dpi=150)
for i, differential in enumerate([False, True]):
    diffPCA_obj = DiffPCA(1.0e-8, differential=differential)
    X_tilde, Z_tilde = diffPCA_obj.fit_transform(X, Z)
    X_recon = diffPCA_obj.inverse_transform(X_tilde)
    Z_recon = diffPCA_obj.inverse_transform(Z_tilde)
    # Plot PCA asset 1 against asset 2 alongside PCA embedding
    axs[0, i].scatter(X[:, 0], X[:, 1], color='dodgerblue', linewidth=0.5, edgecolor='darkblue')
    axs[0, i].scatter(X_recon[:, 0], X_recon[:, 1], color='salmon', linewidth=0.5, edgecolor='darkred', s=30)
    xs = np.linspace(min(X[:, 0]), max(X[:, 1]), 100)
    ys = xs*diffPCA_obj.V[1]/diffPCA_obj.V[0]
    axs[0, i].axline((0, 0), tuple(diffPCA_obj.V.flatten()), ls='--', color='darkred', alpha=0.6)
    axs[0, i].set_aspect('equal', adjustable='datalim')
    # Plot principal component 1 against payoff
    axs[1, i].scatter(X_tilde, y, color='salmon', linewidth=0.5, edgecolor='darkred')
    asp = np.diff(axs[1, i].get_xlim())[0] / np.diff(axs[1, i].get_ylim())[0]
    axs[1, i].set_aspect(asp, adjustable='datalim')
axs[0, 0].set_title('PCA')
axs[0, 1].set_title('Differential PCA')
axs[0, 0].set_xlabel(r'Asset 1 at $T_1$')
axs[0, 1].set_xlabel(r'Asset 1 at $T_1$')
axs[0, 0].set_ylabel(r'Asset 2 at $T_1$')
axs[1, 0].set_xlabel('Component 1')
axs[1, 1].set_xlabel('Component 1')
axs[1, 0].set_ylabel(r'Payoff at $T_2$')
fig.tight_layout()
plt.show()

