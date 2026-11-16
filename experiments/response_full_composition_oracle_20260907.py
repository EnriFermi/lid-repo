"""Two applications at the SAME Gaussian scale: exact geometry controls only.

Composition is an altered field, not the original posterior. Its Jacobian
uses the second posterior derivative at the first, generally off-support mean.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from experiments.response_full_oracle_20260907 import tube_posterior, tube_point
from experiments.response_full_sphere_20260907 import sphere_component
from experiments.response_full_richardson_20260907 import sha256


def compose(field, x, sigma):
    x = np.asarray(x, dtype=float)
    b, j = field(x, sigma)
    g, j_at_b = field(b, sigma)
    jg = j_at_b@j
    r, c = float(np.trace(j)), float(np.sum((b-x)**2)/sigma**2)
    rg, cg = float(np.trace(jg)), float(np.sum((g-x)**2)/sigma**2)
    return {'posterior': b, 'jacobian': j, 'composed': g,
            'second_jacobian': j_at_b, 'composed_jacobian': jg,
            'response': r, 'correction': c, 'full': r+c,
            'composed_response': rg, 'composed_correction': cg, 'composed_full': rg+cg}


def halfline(y, sigma):
    y = np.asarray(y, dtype=float)
    t = y[0]/sigma
    m = np.exp(-t*t/2)/np.sqrt(2*np.pi)/ndtr(t)
    return np.array([y[0]+sigma*m]), np.array([[1-t*m-m*m]])


def sphere_field(dimension, radius=1.):
    def field(y, sigma):
        b, covariance, _ = sphere_component(y, sigma, np.zeros(dimension+1), radius)
        return b, covariance/sigma**2
    return field


def exp_field(order=256, cutoff=12., fixed_radius=None):
    def field(y, sigma):
        out = tube_posterior(y, sigma, order=order, cutoff=cutoff, fixed_radius=fixed_radius)
        return out.mean, out.posterior_covariance/sigma**2
    return field


def row(name, intrinsic_d, x, sigma, field):
    out = compose(field, x, sigma)
    summary = {'case': name, 'd': intrinsic_d, 'sigma': float(sigma),
               'query': np.asarray(x).tolist()}
    for key, value in out.items():
        summary[key] = value.tolist() if isinstance(value, np.ndarray) else value
    for kind in ['response', 'full']:
        summary[kind+'_error'] = abs(out[kind]-intrinsic_d)
        summary['composed_'+kind+'_error'] = abs(out['composed_'+kind]-intrinsic_d)
    return summary


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    dependency_paths = [Path(__file__), Path('experiments/response_full_oracle_20260907.py'),
                        Path('experiments/response_full_sphere_20260907.py')]
    hashes = {str(path): sha256(path) for path in dependency_paths}
    rows = [row('halfline_x_over_sigma_'+str(t), 1, [t], 1., halfline) for t in [0., .5, 4.]]
    raw_rms = .4427780508995
    interior_sigma = .1*raw_rms
    interior_k = (3*np.exp(-4)/interior_sigma)**2
    for d in [1, 2, 5]:
        for name, k in [('resolved_k100', 100.), ('Exp_u4_radius_ratio', interior_k)]:
            x = np.zeros(d+1); x[0] = 1.
            out = row(f'S{d}_{name}', d, x, k**-.5, sphere_field(d))
            out['kappa'] = float(k)
            rows.append(out)
    for u, sigma in [(0., .01*raw_rms), (4., interior_sigma), (6., interior_sigma)]:
        x = tube_point(u, .7)
        primary = row('actual_Exp_u'+str(u), 2, x, sigma, exp_field())
        checks = [compose(exp_field(order=128), x, sigma),
                  compose(exp_field(order=256, cutoff=14.), x, sigma)]
        primary['quadrature_consistency_max'] = max(
            abs(primary[key]-check[key]) for check in checks
            for key in ['response', 'correction', 'full', 'composed_response', 'composed_correction', 'composed_full'])
        if primary['quadrature_consistency_max'] > 1e-7:
            raise RuntimeError('Exp composition quadrature mismatch')
        rows.append(primary)
    x = tube_point(4., .7)
    rows.append(row('Exp_u4_frozen_radius_cylinder', 2, x, interior_sigma,
                    exp_field(fixed_radius=3*np.exp(-4))))
    assert all(sha256(path) == value for path, value in hashes.items())
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({
        'input_hashes': hashes, 'all_sources_unchanged': True,
        'selection': 'three previous Exp points and fixed halfline/sphere mathematical controls, no learned model or fitted coefficient',
        'scope': 'g=b_lambda composed with b_lambda, not the original posterior or a new claimed density',
        'chain_rule': 'Jg(x)=Jb(b(x)) Jb(x), second call retains the same sigma and is generally off support',
        'halfline': 'uniform Lebesgue halfline kernel, a local flat-boundary control not a normalized global data law',
        'actual_Exp': 'original uniform-u/angle law; original posterior recomputed at its first mean; full axial/angular covariance retained',
    }, indent=2)+'\n')
    print(json.dumps([{key: r[key] for key in ['case','d','response','correction','full',
        'composed_response','composed_correction','composed_full']} for r in rows], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
