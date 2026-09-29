import numpy as np
from scipy.stats import multivariate_normal as mvn, norm


class MultivariateSkewGaussian:
    def __init__(self, mean,shape, cov=None):
        self.shape = np.asarray(shape, dtype=float)
        self.dim = len(self.shape)
        self.mean = mean
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov, dtype=float)
        self._mvn = mvn(mean=self.mean, cov=self.cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)

        # Check if the input is a single 1D point (dim,)
        is_single_point = (x.ndim == 1 and x.shape[0] == self.dim)

        # Ensure 2D shape (N, dim) for consistent vectorized matrix operations
        if is_single_point:
            x_mat = x.reshape(1, -1)
        elif x.ndim == 2 and x.shape[1] == self.dim:
            x_mat = x
        else:
            raise ValueError(f"Last dimension of x must match shape dimension ({self.dim})")

        # Compute log components
        pdf_log = self._mvn.logpdf(x_mat)
        cdf_log = norm.logcdf(x_mat @ self.shape)
        res = np.log(2.0) + pdf_log + cdf_log

        # Return a scalar float if input was a single point, else return 1D/nd array
        return float(res[0]) if is_single_point else np.squeeze(res)

    def rvs_slow(self, size=1):
        std_mvn = mvn(np.zeros(self.dim),
                      np.eye(self.dim))
        x = np.empty((size, self.dim))

        # Apply rejection sampling.
        n_samples = 0
        while n_samples < size:
            z = std_mvn.rvs(size=1)
            u = np.random.uniform(0, 2 * std_mvn.pdf(z))
            if not u > self.pdf(z):
                x[n_samples] = z
                n_samples += 1

        # Rescale based on correlation matrix.
        chol = np.linalg.cholesky(self.cov)
        x = (chol @ x.T).T

        return x

    def rvs_fast(self, size=1):
        aCa = self.shape @ self.cov @ self.shape
        delta = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block([[np.ones(1), delta],
                             [delta[:, None], self.cov]])
        x = mvn(np.zeros(self.dim + 1), cov_star).rvs(size)
        x0, x1 = x[:, 0], x[:, 1:]
        inds = x0 <= 0
        x1[inds] = -1 * x1[inds]
        return x1