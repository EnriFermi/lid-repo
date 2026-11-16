"""Within-query finite-trajectory R/C covariance; no stationarity assumption."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from experiments.response_full_richardson_20260907 import sha256


def temporal_covariance(response, correction, truth):
    r, c = np.asarray(response, dtype=float), np.asarray(correction, dtype=float)
    if r.ndim != 2 or c.shape != r.shape:
        raise ValueError('Expected matching [snapshot, query] arrays')
    f = r+c
    rm, cm, fm = r.mean(0), c.mean(0), f.mean(0)
    rc, cc, fc = r-rm, c-cm, f-fm
    vr, vc, vf = ((x*x).mean(0) for x in [rc, cc, fc])
    cov = (rc*cc).mean(0)
    identity = np.max(np.abs(vf-vr-vc-2*cov))
    variance_diff = vc+2*cov
    tolerance = 1e-12*(vr+vc)
    correlation_denom = float(np.sqrt(vr.mean()*vc.mean()))
    row = {
        'var_response': float(vr.mean()), 'var_correction': float(vc.mean()),
        'cov_response_correction': float(cov.mean()), 'var_full': float(vf.mean()),
        'full_minus_response_variance': float(variance_diff.mean()),
        'ratio_full_response_variance': float(vf.mean()/vr.mean()) if vr.mean() else None,
        'within_query_temporal_correlation': float(cov.mean()/correlation_denom) if correlation_denom else None,
        'negative_temporal_covariance_query_fraction': float(np.mean(cov < -tolerance)),
        'full_less_variable_query_fraction': float(np.mean(variance_diff < -tolerance)),
        'response_mean_signed_dimension_error': float((rm-truth).mean()),
        'full_mean_signed_dimension_error': float((fm-truth).mean()),
        'mean_correction': float(cm.mean()),
        'mean_snapshot_response_dimension_mae': float(np.abs(r-truth).mean()),
        'mean_snapshot_full_dimension_mae': float(np.abs(f-truth).mean()),
        'response_of_mean_dimension_mae': float(np.abs(rm-truth).mean()),
        'mean_full_dimension_mae': float(np.abs(fm-truth).mean()),
        'response_mean_bias_squared': float(np.square(rm-truth).mean()),
        'full_mean_bias_squared': float(np.square(fm-truth).mean()),
        'response_snapshot_mse': float(np.square(r-truth).mean()),
        'full_snapshot_mse': float(np.square(f-truth).mean()),
        'variance_identity_max_error': float(identity),
        'response_mse_identity_error': float(abs(np.square(r-truth).mean()-vr.mean()-np.square(rm-truth).mean())),
        'full_mse_identity_error': float(abs(np.square(f-truth).mean()-vf.mean()-np.square(fm-truth).mean())),
    }
    arrays = {'variance_response': vr, 'variance_correction': vc, 'variance_full': vf,
              'covariance_response_correction': cov,
              'full_minus_response_variance': variance_diff,
              'response_time_mean': rm, 'correction_time_mean': cm, 'full_time_mean': fm,
              'response_snapshot_mae': np.abs(r-truth).mean(0),
              'full_snapshot_mae': np.abs(f-truth).mean(0)}
    return row, arrays


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources, rows = {}, []
    for law, d in [('spaghetti', 1), ('exp', 2)]:
        for arm in ['original_lr', 'low_lr']:
            path = args.root/f've_{law}_refinement/{arm}_temporal.npz'
            sources[str(path)] = sha256(path)
            a = np.load(path)
            np.testing.assert_allclose(a['full'], a['response']+a['correction_original'], atol=1e-12)
            if a['response'].shape != (16, 4, 64):
                raise RuntimeError('Unexpected fixed snapshot contract')
            per_query = {}
            for i, scale in enumerate(a['scales']):
                summary, arrays = temporal_covariance(a['response'][:, i], a['correction_original'][:, i], d)
                rows.append({'law': law, 'arm': arm, 'lambda': float(scale), 'd': d,
                    'snapshots': len(a['steps']), 'queries': len(a['query_ids']),
                    'first_step': int(a['steps'][0]), 'last_step': int(a['steps'][-1]), **summary})
                for key, value in arrays.items():
                    per_query.setdefault(key, []).append(value)
            np.savez_compressed(args.output/f'{law}_{arm}.npz', query_ids=a['query_ids'],
                scales=a['scales'], steps=a['steps'], **{key: np.stack(value) for key, value in per_query.items()})
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assert all(sha256(path) == value for path, value in sources.items())
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'input_hashes': sources, 'all_inputs_unchanged': True,
        'selection': 'all4 scales,2 laws,2 arms from paired4096-step LR continuations; no other state',
        'variance': 'population-normalized finite T16 variance within each query, then averaged over64 queries; not pooled acrossqueries',
        'scope': 'descriptive sparse deterministic trajectory identities, notstationarity, significance, seed uncertainty or every-update covariance',
        'stabilizes_full': 'VarC+2CovRC<0 perquery; negativeCov alone is notsufficient',
        'accuracy': 'known geometric d1/d2 for diagnostic errors, no fitted coefficients or selected model',
    }, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
