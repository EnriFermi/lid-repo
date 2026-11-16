"""Stable population posterior for a uniform circle under Gaussian corruption.

The public input is the canonical channel r = X + lambda Z in R^2. An affine
FM adapter must undo its alpha scaling first. The radial normal and tangent
frame depend on the query; on a clean circle the nonzero inward displacement
and small radial posterior gain are genuine curvature effects, not errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


# I1(k)/I0(k) = 1 + sum_n a_n k^(-n). Coefficients follow the Riccati
# identity A' = 1 - A/k - A^2; ten terms are accurate at the k=50 switch.
_LARGE_COEFFICIENTS = (
    -1 / 2, -1 / 8, -1 / 8, -25 / 128, -13 / 32,
    -1073 / 1024, -103 / 32, -375733 / 32768,
    -23797 / 512, -55384775 / 262144,
)


def bessel_ratio_terms(kappa: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return A=I1/I0, A', and A/k with continuous k=0 limits.

    Uses float64 on the input device. Scaled Bessel functions prevent
    exponential overflow in the middle regime. A separate derivative series
    avoids cancellation of 1-A/k-A^2 at high concentration. The origin series
    also avoids 0/0 and preserves the isotropic posterior Jacobian there.
    """
    kappa = kappa.to(torch.float64)
    if not bool((torch.isfinite(kappa) & (kappa >= 0)).all()):
        raise ValueError("kappa must be finite and nonnegative")
    flat = kappa.reshape(-1)
    ratio = torch.empty_like(flat)
    derivative = torch.empty_like(flat)
    over_kappa = torch.empty_like(flat)
    small = flat < 1e-3
    large = flat >= 50
    middle = ~(small | large)
    if bool(small.any()):
        k = flat[small]
        square = k.square()
        a_over_k = 0.5 - square / 16 + square.square() / 96 - 11 * square.pow(3) / 6144
        ratio[small] = k * a_over_k
        over_kappa[small] = a_over_k
        derivative[small] = 0.5 - 3 * square / 16 + 5 * square.square() / 96 - 77 * square.pow(3) / 6144
    if bool(middle.any()):
        k = flat[middle]
        a = torch.special.i1e(k) / torch.special.i0e(k)
        ratio[middle] = a
        over_kappa[middle] = a / k
        derivative[middle] = 1 - a / k - a.square()
    if bool(large.any()):
        inverse = flat[large].reciprocal()
        a = torch.ones_like(inverse)
        da = torch.zeros_like(inverse)
        for order, coefficient in enumerate(_LARGE_COEFFICIENTS, start=1):
            a = a + coefficient * inverse.pow(order)
            da = da - order * coefficient * inverse.pow(order + 1)
        ratio[large] = a
        derivative[large] = da
        over_kappa[large] = a * inverse
    shape = kappa.shape
    return ratio.reshape(shape), derivative.reshape(shape), over_kappa.reshape(shape)


@dataclass(frozen=True)
class CirclePosterior:
    mean: Tensor
    jacobian: Tensor
    response: Tensor
    correction: Tensor
    full: Tensor
    radial_gain: Tensor
    tangent_gain: Tensor
    radial_direction: Tensor
    tangent_direction: Tensor
    concentration: Tensor


def circle_posterior(
    query: Tensor,
    noise_ratio: Tensor | float,
    *,
    radius: float = math.sqrt(2.0),
    center: Tensor | None = None,
) -> CirclePosterior:
    """Population posterior/readouts at arbitrary clean or noisy 2D queries.

    For rho=||query-center|| and k=R*rho/lambda^2, posterior angle is von
    Mises: mean-center = R*A(k)*radial_direction. Canonical Jacobian gains
    are R^2/lambda^2 * A'(k) radially and R^2/lambda^2 * A(k)/k tangentially.
    At rho=0 the posterior is uniform and both gains are R^2/(2 lambda^2).
    No gain clipping is appropriate: off-manifold posterior gains can exceed 1.
    """
    if query.ndim < 1 or query.shape[-1] != 2:
        raise ValueError("query must have final dimension 2")
    if not torch.is_floating_point(query):
        raise TypeError("query must be floating point")
    query = query.to(torch.float64)
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query must be finite")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if center is None:
        center = query.new_zeros(2)
    else:
        center = torch.as_tensor(center, dtype=query.dtype, device=query.device)
    if center.shape != (2,) or not bool(torch.isfinite(center).all()):
        raise ValueError("center must be a finite vector of shape [2]")
    scale = torch.as_tensor(noise_ratio, dtype=query.dtype, device=query.device)
    if scale.ndim == query.ndim and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    scale = torch.broadcast_to(scale, query.shape[:-1])
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError("noise_ratio must be finite and positive")
    relative = query - center
    distance = relative.norm(dim=-1)
    variance = scale.square()
    concentration = radius * distance / variance
    _, ratio_derivative, ratio_over_kappa = bessel_ratio_terms(concentration)
    tangent_gain = radius**2 / variance * ratio_over_kappa
    radial_gain = radius**2 / variance * ratio_derivative
    # This expression has a smooth, division-free rho=0 limit.
    mean = center + tangent_gain[..., None] * relative
    safe_distance = torch.where(distance > 0, distance, torch.ones_like(distance))
    direction = relative / safe_distance[..., None]
    origin_direction = torch.zeros_like(direction)
    origin_direction[..., 0] = 1
    direction = torch.where((distance > 0)[..., None], direction, origin_direction)
    tangent = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    radial_projector = direction[..., :, None] * direction[..., None, :]
    tangent_projector = tangent[..., :, None] * tangent[..., None, :]
    # Sum positive components rather than subtracting two O(1) radial terms
    # when the true radial covariance is O(lambda^2) near the clean circle.
    jacobian = (radial_gain[..., None, None] * radial_projector
                + tangent_gain[..., None, None] * tangent_projector)
    response = radial_gain + tangent_gain
    correction = (mean - query).square().sum(dim=-1) / variance
    return CirclePosterior(
        mean=mean, jacobian=jacobian, response=response, correction=correction,
        full=response + correction, radial_gain=radial_gain,
        tangent_gain=tangent_gain, radial_direction=direction,
        tangent_direction=tangent, concentration=concentration,
    )


def circle_value_and_jacobian(
    query: Tensor, noise_ratio: Tensor | float, *, radius: float = math.sqrt(2.0),
    center: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Tuple convenience adapter matching fixed-checkpoint geometry diagnostics."""
    result = circle_posterior(query, noise_ratio, radius=radius, center=center)
    return result.mean, result.jacobian
