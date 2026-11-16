"""Population posterior controls for the isolated FM mechanism experiment.

All scales and queries use the canonical additive channel ``Y = X + lambda Z``.
An affine FM schedule must undo its alpha scaling before calling these oracles.
The data law is ``X = center + basis @ U`` with orthonormal basis columns and
independent Gaussian or uniform latent coordinates. There is no normal noise.

Uniform boxes provide boundary-dependent posterior means and responses while
retaining exact known tangent and normal spaces. This is a mathematical control
related to the benchmark's affine Uniform construction, not a reproduction of
its particular data artifact. Population RMS normalization below is explicit;
it must not be confused with a separately fitted empirical training transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _legendre_rule() -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(64)


def _tail_uniform_moments(
    query: Tensor, sigma: Tensor, low: Tensor, high: Tensor
) -> tuple[Tensor, Tensor]:
    """Stable bounded quadrature when truncated-normal ratios cancel.

    Integrate distance from the constrained Gaussian mode, avoiding subtraction
    of two enormous raw moments. Density beyond exp(-40) of that mode is
    omitted. The 64-point Gauss--Legendre rule is evaluated with torch on the
    caller's device and remains differentiable almost everywhere.
    """
    mode = torch.clamp(query, min=low, max=high)
    signed_mode = (mode - query) / sigma
    radius = torch.sqrt(signed_mode.square() + 80.0)
    # Solve t^2 + 2*abs(mode)*t = 80 without catastrophic cancellation.
    tail_width = 80.0 / (radius + signed_mode.abs())
    left = torch.where(signed_mode < 0, -tail_width, -radius - signed_mode)
    right = torch.where(signed_mode > 0, tail_width, radius - signed_mode)
    left = torch.maximum(left, (low - mode) / sigma)
    right = torch.minimum(right, (high - mode) / sigma)
    nodes_np, weights_np = _legendre_rule()
    nodes = torch.as_tensor(nodes_np, dtype=query.dtype, device=query.device)
    weights = torch.as_tensor(weights_np, dtype=query.dtype, device=query.device)
    half_width = (right - left) / 2.0
    location = (right + left) / 2.0
    offset = location[..., None] + half_width[..., None] * nodes
    exponent = -0.5 * offset * (offset + 2.0 * signed_mode[..., None])
    mass = weights * torch.exp(exponent)
    probabilities = mass / mass.sum(dim=-1, keepdim=True)
    displacement = (probabilities * offset).sum(dim=-1)
    variance = (
        probabilities * (offset - displacement[..., None]).square()
    ).sum(dim=-1)
    return mode + sigma * displacement, sigma.square() * variance


def uniform_posterior_moments(
    query: Tensor,
    sigma: Tensor | float,
    low: Tensor | float,
    high: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Mean/variance of N(query, sigma^2) restricted to [low, high].

    Uses closed-form truncated-Gaussian moments and log-CDF differences in
    ordinary cases. For far tails or narrow intervals, bounded quadrature
    avoids cancellation in variance. Computation and results are float64 even
    for float32 queries. Inputs broadcast as torch tensors; sigma is positive.
    """
    if not torch.is_floating_point(query):
        raise TypeError("query must be floating point")
    query = query.to(torch.float64)
    sigma = torch.as_tensor(sigma, dtype=query.dtype, device=query.device)
    low = torch.as_tensor(low, dtype=query.dtype, device=query.device)
    high = torch.as_tensor(high, dtype=query.dtype, device=query.device)
    query, sigma, low, high = torch.broadcast_tensors(query, sigma, low, high)
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query must be finite")
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("sigma must be finite and positive")
    if not bool((torch.isfinite(low) & torch.isfinite(high) & (high > low)).all()):
        raise ValueError("finite bounds must satisfy low < high")
    a = (low - query) / sigma
    b = (high - query) / sigma
    # Survival probabilities in the positive tail, CDFs otherwise.
    upper = torch.where(a >= 0, -a, b)
    lower = torch.where(a >= 0, -b, a)
    log_upper = torch.special.log_ndtr(upper)
    log_lower = torch.special.log_ndtr(lower)
    log_z = log_upper + torch.log(-torch.expm1(log_lower - log_upper))
    log_norm = -0.5 * np.log(2.0 * np.pi)
    density_a = torch.exp(-0.5 * a.square() + log_norm - log_z)
    density_b = torch.exp(-0.5 * b.square() + log_norm - log_z)
    shift = density_a - density_b
    mean = query + sigma * shift
    variance = sigma.square() * (
        1.0 + a * density_a - b * density_b - shift.square()
    )
    unstable = (
        (a > 6.0)
        | (b < -6.0)
        | ((b - a) < 0.05)
        | ~torch.isfinite(mean)
        | ~torch.isfinite(variance)
        | (variance < 0)
    )
    if bool(unstable.any()):
        stable_mean, stable_variance = _tail_uniform_moments(
            query[unstable], sigma[unstable], low[unstable], high[unstable]
        )
        # Clones preserve autograd and avoid modifying an input view.
        mean = mean.clone()
        variance = variance.clone()
        mean[unstable] = stable_mean
        variance[unstable] = stable_variance
    return mean, variance


@dataclass(frozen=True)
class PosteriorMoments:
    mean: Tensor
    tangent_mean: Tensor
    tangent_variance: Tensor
    tangent_gain: Tensor
    response: Tensor
    correction: Tensor
    full: Tensor

    def jacobian(self, basis: Tensor) -> Tensor:
        """Materialize Q diag(gain) Q^T only when explicitly requested."""
        basis = basis.to(dtype=self.mean.dtype, device=self.mean.device)
        return torch.einsum("ai,...i,bi->...ab", basis, self.tangent_gain, basis)


@dataclass(frozen=True)
class AffineFixture:
    kind: Literal["gaussian", "uniform"]
    basis: Tensor  # [D,d], orthonormal columns in canonical data coordinates.
    center: Tensor  # [D], actual population center, including fitted translations.
    latent_scale: Tensor  # [d], Gaussian std or uniform half-width.

    def __post_init__(self) -> None:
        if self.kind not in ("gaussian", "uniform"):
            raise ValueError("kind must be gaussian or uniform")
        if self.basis.ndim != 2 or self.basis.shape[0] < self.basis.shape[1]:
            raise ValueError("basis must have shape [D,d] with D >= d")
        if self.center.shape != (self.basis.shape[0],):
            raise ValueError("center must have shape [D]")
        if self.latent_scale.shape != (self.basis.shape[1],):
            raise ValueError("latent_scale must have shape [d]")
        valid_scales = torch.isfinite(self.latent_scale) & (self.latent_scale > 0)
        if not bool(valid_scales.all()):
            raise ValueError("latent scales must be finite and positive")
        if not bool(torch.isfinite(self.center).all()):
            raise ValueError("center must be finite")
        gram = self.basis.double().T @ self.basis.double()
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        if not torch.allclose(gram, identity, atol=2e-6, rtol=2e-6):
            raise ValueError("basis columns must be orthonormal")

    @property
    def ambient_dim(self) -> int:
        return self.basis.shape[0]

    @property
    def intrinsic_dim(self) -> int:
        return self.basis.shape[1]

    def sample(
        self,
        count: int,
        seed: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Deterministic CPU RNG, then device transfer; no global RNG mutation."""
        if count < 1:
            raise ValueError("count must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (count, self.intrinsic_dim)
        if self.kind == "gaussian":
            latent = torch.randn(shape, generator=generator, dtype=torch.float64)
        else:
            latent = (
                2.0 * torch.rand(shape, generator=generator, dtype=torch.float64) - 1.0
            )
        latent = latent * self.latent_scale.detach().cpu().double()
        sample = (
            self.center.detach().cpu().double()
            + latent @ self.basis.detach().cpu().double().T
        )
        return sample.to(device=device, dtype=dtype)

    def transformed(self, subtract: Tensor, divide: float) -> AffineFixture:
        """Express the same law after empirical (x-subtract)/divide preprocessing."""
        if not np.isfinite(divide) or divide <= 0:
            raise ValueError("divide must be finite and positive")
        subtract = torch.as_tensor(
            subtract, dtype=self.center.dtype, device=self.center.device
        )
        if subtract.shape != self.center.shape:
            raise ValueError("subtract must have shape [D]")
        return AffineFixture(
            self.kind,
            self.basis,
            (self.center - subtract) / divide,
            self.latent_scale / divide,
        )

    def posterior(self, query: Tensor, noise_ratio: Tensor | float) -> PosteriorMoments:
        """Evaluate canonical posterior moments at clean or noisy queries."""
        if query.shape[-1] != self.ambient_dim:
            raise ValueError("query last dimension must equal ambient dimension")
        query = query.to(torch.float64)
        basis = self.basis.to(dtype=query.dtype, device=query.device)
        center = self.center.to(dtype=query.dtype, device=query.device)
        scale = self.latent_scale.to(dtype=query.dtype, device=query.device)
        sigma = torch.as_tensor(noise_ratio, dtype=query.dtype, device=query.device)
        if sigma.ndim == query.ndim - 1:
            sigma = sigma[..., None]
        sigma = torch.broadcast_to(sigma, query.shape[:-1] + (1,))
        if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
            raise ValueError("noise_ratio must be finite and positive")
        latent_query = (query - center) @ basis
        if self.kind == "gaussian":
            gain = scale.square() / (scale.square() + sigma.square())
            gain = torch.broadcast_to(gain, latent_query.shape)
            latent_mean = gain * latent_query
            variance = sigma.square() * gain
        else:
            latent_mean, variance = uniform_posterior_moments(
                latent_query, sigma, -scale, scale
            )
            gain = variance / sigma.square()
        mean = center + latent_mean @ basis.T
        response = gain.sum(dim=-1)
        correction = (mean - query).square().sum(dim=-1) / sigma[..., 0].square()
        return PosteriorMoments(
            mean,
            latent_mean,
            variance,
            gain,
            response,
            correction,
            response + correction,
        )


def make_affine_fixture(
    kind: Literal["gaussian", "uniform"] = "gaussian",
    ambient_dim: int = 128,
    intrinsic_dim: int = 3,
    seed: int = 20260905,
    latent_scale: Tensor | float = 1.0,
    population_rms: float | None = 1.0,
    tangent_basis: Tensor | None = None,
    center: Tensor | None = None,
) -> AffineFixture:
    """Create a noiseless affine control with optional centered population RMS.

    ``population_rms`` means sqrt(E||X-center||^2 / D). For isotropic latent
    scales it yields Gaussian std sqrt(D/d) or uniform half-width sqrt(3D/d)
    when set to one. Set it to None to preserve supplied absolute latent scales.
    Pass the trainer's saved tangent_basis to use exactly the same embedding.
    """
    if not 1 <= intrinsic_dim <= ambient_dim:
        raise ValueError("require 1 <= intrinsic_dim <= ambient_dim")
    if tangent_basis is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(
            ambient_dim, intrinsic_dim, generator=generator, dtype=torch.float64
        )
        tangent_basis, triangular = torch.linalg.qr(matrix, mode="reduced")
        tangent_basis = tangent_basis * torch.sign(torch.diag(triangular))[None, :]
    else:
        tangent_basis = tangent_basis.detach().cpu().double()
        if tangent_basis.shape != (ambient_dim, intrinsic_dim):
            raise ValueError("tangent_basis must match declared dimensions")
    center = (
        torch.zeros(ambient_dim, dtype=torch.float64)
        if center is None
        else torch.as_tensor(center).detach().cpu().double()
    )
    scales = torch.broadcast_to(
        torch.as_tensor(latent_scale, dtype=torch.float64), (intrinsic_dim,)
    ).clone()
    if population_rms is not None:
        if not np.isfinite(population_rms) or population_rms <= 0:
            raise ValueError("population_rms must be finite and positive")
        variance_factor = 1.0 / 3.0 if kind == "uniform" else 1.0
        current_rms = torch.sqrt(variance_factor * scales.square().sum() / ambient_dim)
        scales = scales * population_rms / current_rms
    return AffineFixture(kind, tangent_basis, center, scales)
