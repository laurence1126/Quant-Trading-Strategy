import pandas as pd
import numpy as np


class PCA:
    def __init__(self, factor_data: pd.DataFrame, n_components: int = None, explained_variance_ratio: float = None):
        self.factor_data = factor_data
        self.cov_matrix = self.factor_data.cov()
        self.eigenvalues, self.eigenvectors = self.fit_pca()
        if n_components:
            self.eigenvalues = self.eigenvalues[:n_components]
            self.eigenvectors = self.eigenvectors[:, :n_components]
        elif explained_variance_ratio:
            total_variance = sum(self.eigenvalues)
            single_explained_variance_ratio = self.eigenvalues / total_variance
            cum_explained_variance_ratio = np.cumsum(single_explained_variance_ratio)
            n_components = np.argmax(cum_explained_variance_ratio >= explained_variance_ratio) + 1
            self.eigenvalues = self.eigenvalues[:n_components]
            self.eigenvectors = self.eigenvectors[:, :n_components]

    def fit_pca(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors
