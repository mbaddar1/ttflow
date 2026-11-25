"""
Multiwell target from the Tensor Train Diffusion paper (Gruhlke et al., ICML 2026, arXiv:2607.06841), Sec. 4.1, Eq. (18):

    rho(x) = exp( - sum_{i=1}^{m} (x_i^2 - delta)^2  -  1/2 sum_{i=m+1}^{d} x_i^2 )

* the first m coordinates are symmetric double wells with minima at +-sqrt(delta)
* the remaining d - m coordinates are standard Gaussian
* the density is a product of 1-D factors, so it has 2^m equally weighted modes

Paper settings (delta = 2): d = 10 with 8 modes (m = 3), d = 50 with 32 modes (m = 5),
and d = 10, m = 10 (1024 modes) for the baseline comparison in the appendix.

This differs from the FAB Many-Well in `many_well.py` (asymmetric wells, unequal mode weights).
See docs/01-rectified-flow-multiwell-exploratory.md, Sec. 4.2.
"""
import math

import numpy as np
import torch
from scipy.integrate import quad


class MultiwellTTD:
    def __init__(self, dim: int = 10, m: int = 3, delta: float = 2.0, normalised: bool = True,
                 dtype: torch.dtype = torch.float64):
        assert 1 <= m <= dim, f"need 1 <= m <= dim, got m = {m}, dim = {dim}"
        assert delta > 0, f"delta must be > 0, got {delta}"
        self.dim = dim
        self.m = m
        self.delta = float(delta)
        self.normalised = normalised
        self.dtype = dtype
        self.n_modes = 2 ** m
        self.mode_centre = math.sqrt(self.delta)
        # Exact 1-D normaliser of exp(-(x^2 - delta)^2) by quadrature
        self.log_Z_1d_well = math.log(quad(lambda x: math.exp(-(x * x - self.delta) ** 2), -np.inf, np.inf)[0])
        # Rejection-sampling proposal for one well coordinate:
        # 0.5 N(-c, s^2) + 0.5 N(c, s^2), s from the local curvature (Var = 1 / (8 delta)), widened for safety
        self._prop_scale = max(1.5 / math.sqrt(8.0 * self.delta), 0.5)
        self._rejection_log_M = self._compute_rejection_log_bound()

    # ---------------- densities ----------------
    def _log_unnorm_1d_well(self, x: torch.Tensor) -> torch.Tensor:
        return -(x ** 2 - self.delta) ** 2

    def _log_proposal_1d(self, x: torch.Tensor) -> torch.Tensor:
        s, c = self._prop_scale, self.mode_centre
        log_norm = -0.5 * math.log(2 * math.pi) - math.log(s)
        lp_minus = log_norm - 0.5 * ((x + c) / s) ** 2
        lp_plus = log_norm - 0.5 * ((x - c) / s) ** 2
        return torch.logaddexp(lp_minus, lp_plus) - math.log(2.0)

    def _compute_rejection_log_bound(self) -> float:
        # target/proposal -> 0 in the tails (quartic vs quadratic), so the max is attained on a bounded grid
        half_width = self.mode_centre + 10.0 * self._prop_scale + 3.0
        grid = torch.linspace(-half_width, half_width, 400_001, dtype=torch.float64)
        log_ratio = self._log_unnorm_1d_well(grid) - self._log_proposal_1d(grid)
        return float(log_ratio.max()) + math.log(1.05)  # 5% safety margin over the grid maximum

    @property
    def log_Z(self) -> float:
        return self.m * self.log_Z_1d_well + 0.5 * (self.dim - self.m) * math.log(2 * math.pi)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.shape[1] == self.dim, f"x must be (N, {self.dim}), got {tuple(x.shape)}"
        log_p = self._log_unnorm_1d_well(x[:, :self.m]).sum(dim=1) - 0.5 * (x[:, self.m:] ** 2).sum(dim=1)
        return log_p - self.log_Z if self.normalised else log_p

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """grad_x log rho(x), closed form."""
        g = -x.clone()
        xm = x[:, :self.m]
        g[:, :self.m] = -4.0 * xm * (xm ** 2 - self.delta)
        return g

    # ---------------- sampling ----------------
    def _sample_1d_wells(self, n: int, generator: torch.Generator = None) -> torch.Tensor:
        """Exact samples of exp(-(x^2 - delta)^2) / Z by rejection sampling."""
        out = []
        n_needed = n
        while n_needed > 0:
            n_prop = int(1.5 * n_needed * math.exp(self._rejection_log_M - self.log_Z_1d_well)) + 16
            signs = torch.randint(0, 2, (n_prop,), generator=generator, dtype=torch.float64) * 2 - 1
            z = signs * self.mode_centre + self._prop_scale * torch.randn(n_prop, generator=generator,
                                                                          dtype=torch.float64)
            log_u = torch.log(torch.rand(n_prop, generator=generator, dtype=torch.float64))
            accept = log_u < self._log_unnorm_1d_well(z) - self._log_proposal_1d(z) - self._rejection_log_M
            acc = z[accept][:n_needed]
            out.append(acc)
            n_needed -= acc.shape[0]
        return torch.cat(out)

    def sample(self, shape, generator: torch.Generator = None) -> torch.Tensor:
        n = int(shape[0]) if isinstance(shape, (list, tuple, torch.Size)) else int(shape)
        wells = self._sample_1d_wells(n * self.m, generator=generator).reshape(n, self.m)
        gauss = torch.randn(n, self.dim - self.m, generator=generator, dtype=torch.float64)
        return torch.cat([wells, gauss], dim=1).to(self.dtype)

    # ---------------- diagnostics ----------------
    def mode_index(self, x: torch.Tensor) -> torch.Tensor:
        """Mode id in [0, 2^m) from the sign pattern of the m well coordinates (bit i = 1 if x_i > 0)."""
        bits = (x[:, :self.m] > 0).long()
        return (bits * (2 ** torch.arange(self.m, device=x.device))).sum(dim=1)

    def mode_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of samples in each of the 2^m modes. Exact target: 2^-m everywhere."""
        counts = torch.bincount(self.mode_index(x), minlength=self.n_modes).double()
        return counts / counts.sum()

    def mode_occupancy_tv(self, x: torch.Tensor) -> float:
        """Total-variation distance between empirical mode occupancy and the uniform 2^-m weights."""
        occ = self.mode_occupancy(x)
        return 0.5 * float((occ - 1.0 / self.n_modes).abs().sum())
