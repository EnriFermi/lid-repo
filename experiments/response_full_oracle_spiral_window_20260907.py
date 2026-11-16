"""Exact-latent Spiral geometric-resolution window, not a model-scale selector."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.response_full_oracle_20260907 import spiral_curve, spiral_posterior
from experiments.response_full_richardson_20260907 import sha256


SCALES = np.logspace(-5, -2, 13)


def exact_latents(query_ids):
    ids = np.asarray(query_ids)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= 1000):
        raise ValueError('Expected original test IDs in [0,1000)')
    return np.random.RandomState(0).uniform(1., 100., 102000)[101000+ids]


def prepare(source):
    ids = source['query_ids'][:64]
    t = exact_latents(ids)
    raw = np.zeros((len(ids), 30))
    raw[:, :2] = spiral_curve(t)
    mean, rms = source['normalization_mean'].astype(float), float(source['normalization_rms'])
    queries = (raw-mean)/rms
    frozen = source['queries'][:64].astype(float)
    difference = np.linalg.norm(queries-frozen, axis=1)
    return {'query_ids': ids, 'query_raw': raw, 'queries': queries, 'frozen_queries': frozen,
            'latent_t': t, 'scales': SCALES, 'raw_sigma': SCALES*rms,
            'normalization_mean': mean, 'normalization_rms': np.array(rms),
            'canonical_rounding_difference_norm': difference,
            'rounding_difference_over_smallest_scale': difference/SCALES[0]}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = {'ve': args.root/'ve_shared256/e1_spiral_pca/arrays.npz',
                    'vp': args.root/'vp_spiral_span/noise_average_arrays.npz'}
    hashes = {str(path): sha256(path) for path in source_paths.values()}
    fixtures = {name: prepare(dict(np.load(path))) for name, path in source_paths.items()}
    np.testing.assert_array_equal(fixtures['ve']['query_ids'], fixtures['vp']['query_ids'])
    np.testing.assert_array_equal(fixtures['ve']['query_raw'], fixtures['vp']['query_raw'])
    for name, fixture in fixtures.items():
        np.savez_compressed(args.output/(name+'_fixture.npz'), **fixture)
    print('FIXTURES_READY '+str(args.output), flush=True)
    summaries = {}
    for name, fixture in fixtures.items():
        shape = (len(SCALES), len(fixture['query_ids']))
        arrays = {key: np.zeros(shape) for key in ['response', 'correction', 'full',
            'other_turn_mass', 'response_order_error', 'correction_order_error',
            'response_cutoff_error', 'correction_cutoff_error']}
        posterior_raw = np.zeros(shape+(30,))
        for i, sigma in enumerate(fixture['raw_sigma']):
            for j, query in enumerate(fixture['query_raw'][:, :2]):
                low = spiral_posterior(query, sigma, order=48, cutoff=12.)
                high = spiral_posterior(query, sigma, order=96, cutoff=12.)
                wide = spiral_posterior(query, sigma, order=96, cutoff=14.)
                for key in ['response', 'correction', 'full']:
                    arrays[key][i, j] = getattr(high, key)
                for key in ['response', 'correction']:
                    arrays[key+'_order_error'][i, j] = abs(getattr(low, key)-getattr(high, key))
                    arrays[key+'_cutoff_error'][i, j] = abs(getattr(wide, key)-getattr(high, key))
                arrays['other_turn_mass'][i, j] = high.metadata['other_turn_mass']
                posterior_raw[i, j, :2] = high.mean
            print(name, float(SCALES[i]), 'full_mae', float(np.abs(arrays['full'][i]-1).mean()),
                  'fraction_abs_error_lt_.1', float((np.abs(arrays['full'][i]-1) < .1).mean()), flush=True)
        max_error = max(float(arrays[key].max()) for key in arrays if key.endswith('_error'))
        if max_error > 1e-6:
            raise RuntimeError(f'Quadrature consistency failed: {name} {max_error}')
        posterior = (posterior_raw-fixture['normalization_mean'])/float(fixture['normalization_rms'])
        np.savez_compressed(args.output/(name+'_oracle.npz'), **fixture, posterior=posterior,
                            posterior_raw=posterior_raw, **arrays)
        summaries[name] = {'count': shape[1], 'max_quadrature_component_difference': max_error,
            'normalization_rms': float(fixture['normalization_rms']),
            'maximum_exact_vs_frozen_query_distance': float(fixture['canonical_rounding_difference_norm'].max()),
            'maximum_query_difference_over_smallest_scale': float(fixture['rounding_difference_over_smallest_scale'].max()),
            'mean_query_difference_over_smallest_scale': float(fixture['rounding_difference_over_smallest_scale'].mean()),
            'rows': [{'lambda': float(scale), 'raw_sigma': float(fixture['raw_sigma'][i]),
                'response_mae': float(np.abs(arrays['response'][i]-1).mean()),
                'full_mae': float(np.abs(arrays['full'][i]-1).mean()),
                'full_absolute_error_lt_point1_fraction': float((np.abs(arrays['full'][i]-1) < .1).mean()),
                'response_absolute_error_lt_point1_fraction': float((np.abs(arrays['response'][i]-1) < .1).mean()),
                'other_turn_mass_mean': float(arrays['other_turn_mass'][i].mean()),
                'other_turn_mass_max': float(arrays['other_turn_mass'][i].max())} for i, scale in enumerate(SCALES)]}
    assert all(sha256(path) == value for path, value in hashes.items())
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'oracle_source_sha256': sha256(Path(__file__).with_name('response_full_oracle_20260907.py')),
        'inputs_sha256': hashes, 'all_inputs_unchanged': True,
        'queries': 'Original first64 testIDs, exact RandomState seed0 continuous latent t, not inversion of rounded coordinates',
        'normalization': 'Each model retains its own source mean/RMS; exact canonical queries are float64',
        'geometry': 'true uniform-t complete Spiral, all nearby turns retained',
        'selection': '13 predeclared quarter-decade scales1e-5..1e-2, fixed absolute-error threshold.1 descriptive only',
        'training_scope': 'All scales below.01 are outside original training support; no deployable scale choice inferred',
        'numerical_gate': '48vs96 quadrature order atcutoff12 and12vs14 cutoff; everyquery every scale component difference <=1e-6'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
