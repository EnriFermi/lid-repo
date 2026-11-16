"""Fixed adjacent-scale Richardson readouts, with no dimension-based selection.

This signed extrapolation is not a posterior or its full readout. All eight
existing adjacent scale pairs are retained. Each output is indexed by the
finer scale, and no scale below the training interval is requested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr


DIMENSIONS = {'spiral': 1, 'exp': 2, 'crescent': 3, 'spaghetti': 1, 'sphere4': 5}
ORACLES = {
    'spiral': 'population_oracle_ve_spiral256/fixture_arrays.npz',
    'exp': 'population_oracle_ve_exp256/fixture_arrays.npz',
    'crescent': 'population_oracle_ve_crescent256/arrays.npz',
    'spaghetti': 'population_oracle_ve_spaghetti256/arrays.npz',
    'sphere4': 'population_oracle_ve_sphere4_256/arrays.npz',
}
LEARNED_VE = {
    'spiral': 've_shared256/e1_spiral_pca/arrays.npz',
    'exp': 've_shared256/e6_exp_pca/arrays.npz',
    'crescent': 've_shared256/e7_crescent_moon_radius3.0/arrays.npz',
    'spaghetti': 've_spaghetti_shared256/e8_spaghetti_pca/arrays.npz',
    'sphere4': 've_sphere4_shared256/e8_sphere4_pca/arrays.npz',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extrapolate(values, scales):
    values, scales = np.asarray(values, dtype=float), np.asarray(scales, dtype=float)
    if (scales.ndim != 1 or len(scales) < 2 or values.shape[0] != len(scales)
            or np.any(scales <= 0) or np.any(np.diff(scales) <= 0)):
        raise ValueError('Require positive increasing scales and aligned values')
    a = (scales[1:]/scales[:-1])**2
    a = a.reshape((len(a),)+(1,)*(values.ndim-1))
    return (a*values[:-1]-values[1:])/(a-1)


def halfline_readouts(distance_over_sigma):
    """Locally uniform half-line boundary control, query x>=0, not a proper prior."""
    t = np.asarray(distance_over_sigma, dtype=float)
    if np.any(t < 0):
        raise ValueError('This control is evaluated on clean support')
    mills = np.exp(-t*t/2)/np.sqrt(2*np.pi)/ndtr(t)
    full = 1-t*mills
    return full-mills*mills, full


def metrics(original, proposed, target):
    error, new_error = original-target, proposed-target
    return {
        'original_mae': float(np.abs(error).mean()),
        'extrapolated_mae': float(np.abs(new_error).mean()),
        'original_rmse': float(np.sqrt(np.square(error).mean())),
        'extrapolated_rmse': float(np.sqrt(np.square(new_error).mean())),
        'original_mean': float(original.mean()),
        'extrapolated_mean': float(proposed.mean()),
        'pointwise_strict_improvement_fraction': float((np.abs(new_error) < np.abs(error)).mean()),
        'extrapolated_min': float(proposed.min()),
        'extrapolated_max': float(proposed.max()),
    }


def analyze_pairwise(name, law, data, response, full, oracle=None):
    scales = np.asarray(data['scales'], dtype=float)
    response, full = np.asarray(response), np.asarray(full)
    if response.shape != full.shape or response.shape != (len(scales), len(data['query_ids'])):
        raise ValueError('Readout/query shape mismatch')
    arrays = {'query_ids': data['query_ids'], 'queries': data['queries'],
              'fine_scales': scales[:-1], 'coarse_scales': scales[1:],
              'squared_scale_ratio': (scales[1:]/scales[:-1])**2}
    rows = []
    if oracle is not None:
        np.testing.assert_array_equal(data['query_ids'], oracle['query_ids'])
        np.testing.assert_array_equal(data['queries'], oracle['queries'])
        np.testing.assert_allclose(scales, oracle['scales'], rtol=1e-9, atol=0)
    for readout, values in [('response', response), ('full', full)]:
        proposed = extrapolate(values, scales)
        arrays['original_'+readout] = values[:-1]
        arrays['extrapolated_'+readout] = proposed
        for i in range(len(scales)-1):
            row = {'series': name, 'law': law, 'dimension': DIMENSIONS[law],
                   'readout': readout, 'count': len(data['query_ids']),
                   'fine_scale': float(scales[i]), 'coarse_scale': float(scales[i+1]),
                   **metrics(values[i], proposed[i], DIMENSIONS[law])}
            if oracle is not None:
                # Distinguish original-point target from the modified oracle operator.
                reference = oracle[readout]
                oracle_proposed = extrapolate(reference, scales)
                for key, target in [('same_scale_oracle', reference[i]),
                                    ('extrapolated_oracle_operator', oracle_proposed[i])]:
                    row['original_'+key+'_mae'] = float(np.abs(values[i]-target).mean())
                    row['extrapolated_'+key+'_mae'] = float(np.abs(proposed[i]-target).mean())
                row['oracle_scale_max_relative_mismatch'] = float(np.max(np.abs(scales/oracle['scales']-1)))
                arrays['same_scale_oracle_'+readout] = reference[:-1]
                arrays['extrapolated_oracle_'+readout] = oracle_proposed
            rows.append(row)
    return arrays, rows


def diagnostic_regions(law, data):
    count = len(data['query_ids'])
    for i, coarse_sigma in enumerate(data['raw_sigma'][1:]):
        if law == 'spiral':
            yield i, 'other_turn_mass_coarse_le_0.001', data['other_turn_mass'][i+1] <= .001
            yield i, 'other_turn_mass_fine_gt_0.5', data['other_turn_mass'][i] > .5
        elif law == 'exp':
            u, radius = data['latent_u'], data['local_radius']
            ends = np.minimum(np.hypot(u, radius-3), np.hypot(8-u, radius-3*np.exp(-8)))
            yield i, 'resolved_interior_coarse_sigma_le_radius_over10_ends_ge5sigma', (
                (coarse_sigma <= .1*radius) & (ends >= 5*coarse_sigma))
            yield i, 'unresolved_fiber_fine_sigma_ge_radius', data['raw_sigma'][i] >= radius
            yield i, 'end_boundary_distance_le_coarse_sigma', ends <= coarse_sigma
        elif law == 'crescent':
            q = data['queries_raw']
            distance = np.minimum.reduce([3-np.linalg.norm(q[:, :2], axis=1),
                np.linalg.norm(q[:, :2]-np.array([0, .3]), axis=1)-2.697,
                3-np.abs(q[:, 2])])
            yield i, 'boundary_distance_le_coarse_sigma', distance <= coarse_sigma
            yield i, 'boundary_distance_ge5_coarse_sigma', distance >= 5*coarse_sigma
        elif law == 'sphere4':
            for component in range(4):
                yield i, f'component_{component}', data['component'] == component
        elif law == 'spaghetti':
            yield i, 'remote_phase_mass_coarse_le_0.001', data['remote_phase_mass'][i+1] <= .001
        assert count > 0


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader(); writer.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources, original_hashes, loaded = {}, {}, {}
    def load(relative):
        path = args.root/relative
        if str(path) not in loaded:
            original_hashes[str(path)] = sha256(path)
            loaded[str(path)] = dict(np.load(path))
        return loaded[str(path)]
    oracle_rows, regional_rows, learned_rows = [], [], []
    for law, relative in ORACLES.items():
        data = load(relative)
        np.testing.assert_allclose(data['full'], data['response']+data['correction'], atol=1e-12, rtol=0)
        arrays, rows = analyze_pairwise('oracle_'+law, law, data, data['response'], data['full'])
        oracle_rows.extend(rows)
        np.savez_compressed(args.output/f'oracle_{law}.npz', **arrays)
        for i, label, mask in diagnostic_regions(law, data):
            if not np.any(mask):
                continue
            for readout in ['response', 'full']:
                regional_rows.append({'law': law, 'region': label, 'readout': readout,
                    'fine_scale': float(data['scales'][i]), 'coarse_scale': float(data['scales'][i+1]),
                    'count': int(mask.sum()), **metrics(arrays['original_'+readout][i, mask],
                        arrays['extrapolated_'+readout][i, mask], DIMENSIONS[law])})
    write_csv(args.output/'oracle_metrics.csv', oracle_rows)
    write_csv(args.output/'oracle_regions.csv', regional_rows)
    boundary_rows = []
    ratio = 10**.25
    for distance in [0, .25, .5, 1, 2, 3, 4, 5]:
        r, f = halfline_readouts(np.array([distance, distance/ratio]))
        boundary_rows.append({'distance_over_fine_sigma': distance, 'coarse_over_fine_sigma': ratio,
            'response': float(r[0]), 'full': float(f[0]),
            'extrapolated_response': float(extrapolate(r, [1, ratio])[0]),
            'extrapolated_full': float(extrapolate(f, [1, ratio])[0])})
    write_csv(args.output/'halfline_control.csv', boundary_rows)

    if args.learned:
        registry = [(f've_{law}', law, relative, ORACLES[law]) for law, relative in LEARNED_VE.items()]
        for law in ['spiral', 'exp', 'spaghetti']:
            oracle_path = ('population_oracle_spiral_fixture_v2/fixture_arrays.npz'
                           if law == 'spiral' else ORACLES[law])
            for head in ['posterior', 'noise']:
                registry.append((f'vp_{law}_{head}_average', law,
                    f'vp_{law}_span/{head}_average_arrays.npz', oracle_path))
        for law in ['spaghetti', 'exp']:
            for lr in ['original_lr', 'low_lr']:
                registry.append((f've_{law}_refinement_{lr}_average', law,
                    f've_{law}_refinement/{lr}_average_arrays.npz', ORACLES[law]))
        for name, law, relative, oracle_path in registry:
            data, oracle = load(relative), load(oracle_path)
            variants = [('original', data['response'], data['full'])]
            if 'response_span' in data:
                variants.append(('projected', data['response_span'], data['full_projected']))
            for variant, response, full in variants:
                label = name+'_'+variant
                arrays, rows = analyze_pairwise(label, law, data, response, full, oracle)
                np.savez_compressed(args.output/f'{label}.npz', **arrays)
                learned_rows.extend(rows)
            sources[name] = {'arrays': str(args.root/relative), 'oracle': str(args.root/oracle_path),
                'variants': [v[0] for v in variants]}
        write_csv(args.output/'learned_metrics.csv', learned_rows)

    for path, before in original_hashes.items():
        if sha256(path) != before:
            raise RuntimeError(f'Source changed while reading: {path}')
    summary = {}
    for name, rows in [('oracle', oracle_rows), ('learned', learned_rows)]:
        summary[name] = []
        for series in dict.fromkeys(row['series'] for row in rows):
            for readout in ['response', 'full']:
                group = [r for r in rows if r['series'] == series and r['readout'] == readout]
                summary[name].append({'series': series, 'readout': readout, 'pairs': len(group),
                    'mae_improved_pairs': sum(r['extrapolated_mae'] < r['original_mae'] for r in group),
                    'rmse_improved_pairs': sum(r['extrapolated_rmse'] < r['original_rmse'] for r in group)})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({
        'source_sha256': sha256(__file__), 'input_arrays_sha256': original_hashes,
        'all_sources_unchanged': True, 'learned_sources': sources,
        'formula': '(a*H_fine-H_coarse)/(a-1), a=(lambda_coarse/lambda_fine)^2',
        'scope': 'signed readout extrapolation, not a posterior; no claim it equals full readout of an extrapolated posterior',
        'selection': 'all eight existing adjacent pairs; fixed exponent2; no labels in coefficient or pair selection',
        'pointwise_oracle_errors': 'both original-scale population readout and extrapolated population operator retained',
        'regions': 'fixed diagnostic geometry/posterior-mass conditions; not deployable selectors and may overlap',
        'oracle_metric_rows': len(oracle_rows), 'learned_metric_rows': len(learned_rows),
        'scale_matching': 'exact queries and IDs; scales rtol1e-9, actual mismatch recorded for each learned row',
    }, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--learned', action='store_true')
    run(parser.parse_args())
