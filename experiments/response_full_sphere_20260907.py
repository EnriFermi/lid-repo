"""Exact continuous Gaussian posterior for the benchmark's four S5 components.

The radial/angular integrals are analytic von Mises--Fisher moments. Component
covariances are centered before mixing to avoid subtracting large center terms.
No nearest-component approximation, finite training-cloud oracle, or d-based
readout selection is used. Generic single-sphere helpers support independent
circle and S2 controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, hyp0f1, ive, logsumexp

from experiments.response_full_models_20260907_span import sha256


def sphere_component(query, sigma, center, radius):
    query, center = np.asarray(query, dtype=float), np.asarray(center, dtype=float)
    m = query.shape[-1]
    if sigma <= 0 or radius <= 0 or m < 2:
        raise ValueError("positive scale/radius and ambient component dimension >=2 required")
    offset = query-center
    rho = np.linalg.norm(offset, axis=-1)
    k = rho*radius/sigma**2
    nu = m/2-1
    small = k < 1e-3
    safe = np.where(small, 1., k)
    a = ive(nu+1, safe)/ive(nu, safe)
    a = np.where(small, k/m-k**3/(m*m*(m+2)), a)
    a_over_k = np.divide(a, k, out=np.full_like(k, 1./m), where=k > 0)
    log_normalizer_minus_k = gammaln(m/2)+nu*np.log(2/safe)+np.log(ive(nu, safe))
    log_normalizer_minus_k = np.where(small, np.log(hyp0f1(m/2, np.where(small, k, 0.)**2/4))-k,
                                     log_normalizer_minus_k)
    # exp(-(rho-a)^2/(2 sigma²)) already includes the leading exp(k).
    log_density = -(rho-radius)**2/(2*sigma**2)+log_normalizer_minus_k
    unit = np.divide(offset, rho[..., None], out=np.zeros_like(offset), where=rho[..., None] > 0)
    mean = center+radius*a[..., None]*unit
    radial = 1-a*a-(m-1)*a_over_k
    radial = np.where(small, 1./m-3*k*k/(m*m*(m+2)), radial)
    outer = unit[..., :, None]*unit[..., None, :]
    covariance = radius**2*(a_over_k[..., None, None]*np.eye(m)
                           +(radial-a_over_k)[..., None, None]*outer)
    return mean, covariance, log_density


def mixture_posterior(query, sigma, centers, radii, weights=None):
    query = np.asarray(query, dtype=float)
    centers, radii = np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)
    weights = np.ones(len(radii))/len(radii) if weights is None else np.asarray(weights, dtype=float)
    if len(centers) != len(radii) or len(weights) != len(radii) or np.any(weights <= 0):
        raise ValueError("invalid positive mixture weights/components")
    component = [sphere_component(query, sigma, c, a) for c, a in zip(centers, radii)]
    means = np.stack([v[0] for v in component], axis=-2)
    covariances = np.stack([v[1] for v in component], axis=-3)
    logits = np.stack([v[2] for v in component], axis=-1)+np.log(weights/weights.sum())
    log_normalizer = logsumexp(logits, axis=-1)
    posterior_weights = np.exp(logits-log_normalizer[..., None])
    mean = np.einsum("...k,...ki->...i", posterior_weights, means)
    centered = means-mean[..., None, :]
    covariance = np.einsum("...k,...kij->...ij", posterior_weights,
                          covariances+centered[..., :, :, None]*centered[..., :, None, :])
    response = np.trace(covariance, axis1=-2, axis2=-1)/sigma**2
    correction = np.sum((mean-query)**2, axis=-1)/sigma**2
    return {"posterior": mean, "covariance": covariance, "response": response,
            "correction": correction, "full": response+correction,
            "weights": posterior_weights, "log_evidence_without_gaussian_constant": log_normalizer}


def benchmark_components():
    centers = np.zeros((4, 6))
    centers[:, :2] = [[3, 3], [-3, -3], [3, -3], [-3, 3]]
    radii = 3./3.**np.arange(4)
    return centers, radii


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = np.load(args.arrays)
    centers, radii = benchmark_components()
    raw = data['query_raw'][:, :6].astype(float)
    rms = float(data['normalization_rms'])
    normalization_mean = data['normalization_mean'].astype(float)
    queries = data['queries']
    nearest_component = np.argmin(np.abs(np.linalg.norm(raw[:, None, :]-centers, axis=-1)-radii), axis=-1)
    records, saved = [], {}
    for i, scale in enumerate(data['scales']):
        sigma = float(scale)*rms
        result = mixture_posterior(raw, sigma, centers, radii)
        fullraw = np.zeros_like(queries)
        fullraw[:, :6] = result['posterior']
        normalized_posterior = (fullraw-normalization_mean)/rms
        # Query normalization follows the checkpoint's original float32 map.
        correction = ((normalized_posterior-queries)**2).sum(-1)/float(scale)**2
        response = result['response']
        covariance = np.zeros((len(raw), queries.shape[1], queries.shape[1]))
        covariance[:, :6, :6] = result['covariance']/sigma**2
        e = data['posterior'][i]-normalized_posterior
        geometric_displacement = normalized_posterior-queries
        cross = 2*(geometric_displacement*e).sum(-1)/float(scale)**2
        energy = (e*e).sum(-1)/float(scale)**2
        np.testing.assert_allclose(data['correction_original'][i], correction+cross+energy, atol=2e-10)
        row = {'lambda': float(scale), 'raw_sigma': sigma,
               'oracle_response_mae': float(np.abs(response-5).mean()),
               'oracle_full_mae': float(np.abs(response+correction-5).mean()),
               'oracle_correction_mean': float(correction.mean()),
               'learned_response_mae': float(np.abs(data['response'][i]-5).mean()),
               'learned_full_mae': float(np.abs(data['full'][i]-5).mean()),
               'learned_response_oracle_mae': float(np.abs(data['response'][i]-response).mean()),
               'learned_full_oracle_mae': float(np.abs(data['full'][i]-response-correction).mean()),
               'learned_correction_mean': float(data['correction_original'][i].mean()),
               'correction_cross_mean': float(cross.mean()), 'posterior_error_energy': float(energy.mean()),
               'component_metrics': []}
        for k in range(4):
            mask = nearest_component == k
            row['component_metrics'].append({'component': k, 'radius': float(radii[k]), 'count': int(mask.sum()),
                'oracle_response_mae': float(np.abs(response[mask]-5).mean()),
                'oracle_full_mae': float(np.abs(response[mask]+correction[mask]-5).mean()),
                'other_component_mass': float((1-result['weights'][mask, k]).mean())})
        records.append(row)
        for key, value in [('posterior', normalized_posterior), ('response', response),
                           ('correction', correction), ('full', response+correction),
                           ('jacobian', covariance), ('mixture_weights', result['weights']),
                           ('posterior_error', e), ('correction_cross', cross), ('error_energy', energy)]:
            saved.setdefault(key, []).append(value)
    np.savez_compressed(args.output/'arrays.npz', queries=queries, queries_raw=raw,
        query_ids=data['query_ids'], scales=data['scales'], raw_sigma=data['raw_sigma'],
        component=nearest_component, normalization_mean=normalization_mean, normalization_scale=rms,
        **{key: np.stack(value) for key, value in saved.items()})
    (args.output/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'arrays_sha256': sha256(args.arrays), 'prior': 'equal mixture of uniform S5 surface measures',
        'radii': radii.tolist(), 'centers': centers.tolist(), 'true_local_d': 5,
        'oracle': 'exact analytic angular moments, full mixture weights, no nearest-component substitution',
        'normalization': 'checkpoint-owned float32 queries; raw population law retained'}, indent=2)+'\n')
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
