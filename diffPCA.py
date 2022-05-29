# import libraries
import numpy as np


class DiffPCA:
    def __init__(self, n_components, differential=True):
        self.n_components = n_components
        self.differential = differential

    def fit(self, Z):
        # Perform eigendecomposition
        self.Z_cov = np.matmul(Z.T, Z) / Z.shape[0]
        eigen_vals, eigen_vecs = np.linalg.eigh(self.Z_cov)
        # Reverse order to sort descending eigenvalues
        self.eigen_vals = np.flip(eigen_vals)
        self.eigen_vecs = np.flip(eigen_vecs, axis=1)
        # Determine number of principal components if n_components<1 defines a tolerance
        explained_var = np.cumsum(self.eigen_vals)
        total_var = eigen_vals.sum()
        explained_variance_ratio = explained_var / total_var
        if self.n_components < 1:
            self.n_components = np.searchsorted(explained_variance_ratio, self.n_components, side='left') + 1
        # Select principal components
        self.d = self.eigen_vals[:self.n_components]
        self.V = self.eigen_vecs[:, :self.n_components]

    def transform(self, X):
        return np.matmul(X, self.V)

    def fit_transform(self, X, Z):
        if self.differential:
            self.fit(Z)
        else:
            self.fit(X)
        X_tilde = self.transform(X)
        Z_tilde = self.transform(Z)
        return X_tilde, Z_tilde

    def inverse_transform(self, X_tilde):
        return np.matmul(X_tilde, self.V.T)
