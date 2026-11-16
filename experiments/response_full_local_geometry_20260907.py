"""Independent local heat-coefficient checks on the original Exp population.

Classical smooth-manifold convolution coefficients, not a new endpoint
theorem. Exact-law numerical checks use fixed interior points and decreasing
sigma; they do not replace the finite-scale population references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.response_full_oracle_20260907 import tube_point, tube_posterior


def local_coefficients(ii_squared, h_squared, *, laplacian_log_density=0., gradient_log_density_squared=0.):
    scalar = h_squared-ii_squared
    response = laplacian_log_density-scalar/2
    correction = gradient_log_density_squared+h_squared/4
    return {'scalar_curvature': scalar, 'response_coefficient': response,
            'correction_coefficient': correction, 'full_coefficient': response+correction,
            'twice_response_plus_correction': 2*response+correction}


def exp_coefficients(u, prior='uniform_u'):
    t = float(9*np.exp(-2*u)); g = 1+t
    h_squared = 1/(t*g**3)
    ii_squared = (1+2*t+2*t*t)/(t*g**3)
    if prior == 'uniform_u':
        laplacian = -(1+4*t)/g**3
        gradient = (1+2*t)**2/g**3
    elif prior == 'uniform_area':
        laplacian = gradient = 0.
    else:
        raise ValueError('Unknown original/counterfactual prior')
    return {'u': float(u), 'radius': float(np.sqrt(t)), 'prior': prior,
            'ii_squared': ii_squared, 'h_squared': h_squared,
            'laplacian_log_density': laplacian, 'gradient_log_density_squared': gradient,
            **local_coefficients(ii_squared, h_squared, laplacian_log_density=laplacian,
                                 gradient_log_density_squared=gradient)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for u in [1., 2., 4.]:
        for prior in ['uniform_u', 'uniform_area']:
            coefficients = exp_coefficients(u, prior)
            for relative_scale in [.02, .01, .005]:
                sigma = relative_scale*min(1., coefficients['radius'])
                q = tube_point(u, .7)
                result = tube_posterior(q, sigma, order=256, cutoff=14., prior=prior)
                check = tube_posterior(q, sigma, order=128, cutoff=12., prior=prior)
                numeric_r = (result.response-2)/sigma**2
                numeric_c = result.correction/sigma**2
                rows.append({**coefficients, 'sigma': sigma, 'relative_scale': relative_scale,
                             'numeric_response_coefficient': numeric_r,
                             'numeric_correction_coefficient': numeric_c,
                             'numeric_full_coefficient': (result.full-2)/sigma**2,
                             'response_coefficient_error': numeric_r-coefficients['response_coefficient'],
                             'correction_coefficient_error': numeric_c-coefficients['correction_coefficient'],
                             'quadrature_response_difference': abs(result.response-check.response),
                             'quadrature_correction_difference': abs(result.correction-check.correction)})
    (args.output/'exp_coefficients.json').write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(json.dumps({key: row[key] for key in ['u', 'prior', 'sigma', 'response_coefficient',
                                                  'numeric_response_coefficient', 'correction_coefficient',
                                                  'numeric_correction_coefficient']}))


if __name__ == '__main__':
    main()
