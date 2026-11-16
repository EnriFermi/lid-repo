"""Exact-law check: identical density flow need not identify the posterior.

Time is physical Gaussian standard deviation sigma, not variance or log sigma.
No training or sampling-quality experiment is performed.  The constructed
field is deliberately NOT the population-optimal independent-FM velocity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.response_full_oracle_20260907 import tube_point, tube_posterior


A = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def gauge_check(query, sigma, *, angular_speed=1., order=256):
    x = np.asarray(query, dtype=np.float64)
    result = tube_posterior(x, sigma, order=order)
    score = (result.mean - x) / sigma**2
    velocity = -sigma * score
    added_velocity = angular_speed * (A @ x)
    modified_velocity = velocity + added_velocity
    # Canonical alpha=1,beta=sigma: v=(x-b)/sigma, b=x-sigma*v.
    adapted = x - sigma * modified_velocity
    error = -sigma * added_velocity
    jacobian = result.posterior_covariance / sigma**2
    adapted_jacobian = jacobian - sigma * angular_speed * A
    correction = float(np.sum((adapted-x)**2) / sigma**2)
    response = float(np.trace(adapted_jacobian))
    predicted_inflation = float(angular_speed**2 * np.sum((A @ x)**2))
    # Rotated density and posterior equivariance, independent of the gauge
    # continuity identity expressed through Tweedie's score.
    angle = .37
    rotation = np.eye(3)
    rotation[1:, 1:] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
    rotated = tube_posterior(rotation @ x, sigma, order=order)
    h = 1e-4 * sigma
    flux_divergence = 0.
    for index in range(3):
        direction = np.eye(3)[index]
        plus, minus = x+h*direction, x-h*direction
        plus_mass = np.exp(tube_posterior(plus, sigma, order=order).log_mass-result.log_mass)
        minus_mass = np.exp(tube_posterior(minus, sigma, order=order).log_mass-result.log_mass)
        flux_divergence += (plus_mass*(angular_speed*A@plus)[index]
                           - minus_mass*(angular_speed*A@minus)[index])/(2*h)
    return {
        'query': x.tolist(), 'sigma': sigma, 'angular_speed_per_sigma': angular_speed,
        'response_original': result.response, 'response_adapted': response,
        'correction_original': result.correction, 'correction_adapted': correction,
        'full_original': result.full, 'full_adapted': response+correction,
        'predicted_correction_inflation': predicted_inflation,
        'adapter_error_residual': float(np.max(np.abs(adapted-result.mean-error))),
        'weighted_divergence_over_density': float(angular_speed*np.trace(A)+added_velocity@score),
        'finite_difference_weighted_divergence_over_density': float(flux_divergence),
        'correction_identity_residual': correction-result.correction-predicted_inflation,
        'cross_term': float(2*(result.mean-x)@error/sigma**2),
        'rotation_log_mass_residual': rotated.log_mass-result.log_mass,
        'rotation_posterior_residual': float(np.max(np.abs(rotated.mean-rotation@result.mean))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [gauge_check(tube_point(u, .7), .0442778) for u in [0., 2., 4., 6.]]
    # Off-support locations check a density identity, not just support algebra.
    cases += [gauge_check(tube_point(4., .7)+np.array([.03, .01, -.02]), .0442778)]
    report = {
        'scope': 'Exact Exp law, sigma-time rotational probability-flow gauge. '
                 'Same marginals; deliberately not the population-optimal posterior/FM field. '
                 'Not an attribution of observed model error.',
        'cases': cases,
    }
    (args.output/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
