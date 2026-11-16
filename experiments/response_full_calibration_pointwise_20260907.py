"""Pointwise direction/gain attribution of immutable noisy displacement arrays.

This is not a new fitted readout. Pointwise oracle scalars minimize vector-value
error only; a spatially varying scalar also changes a field's Jacobian.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.response_full_calibration_geometry_20260907 import REGISTRY
from experiments.response_full_richardson_20260907 import sha256


def pointwise_alignment(v, u, frozen_gamma=1.):
    v, u = np.asarray(v, dtype=float), np.asarray(u, dtype=float)
    if v.shape != u.shape or v.ndim < 2:
        raise ValueError('Matching arrays with a last vector dimension required')
    vv, uu, uv = (np.sum(a*b, axis=-1) for a, b in [(v, v), (u, u), (u, v)])
    if uu.mean() <= 0 or vv.mean() <= 0:
        raise ValueError('Positive aggregate oracle and model energy required')
    beta = uv.mean()/uu.mean()
    local_beta = np.divide(uv, uu, out=np.zeros_like(uv), where=uu > 0)
    w = v-local_beta[..., None]*u
    direction_energy = np.sum(w*w, axis=-1)
    varying_gain_energy = (local_beta-beta)**2*uu
    global_residual = v-beta*u
    residual_energy = np.sum(global_residual**2, axis=-1)
    # Pointwise Pythagoras holds because w_i is perpendicular to u_i;
    # averaging yields the global residual-energy decomposition as well.
    split_error = np.max(np.abs(residual_energy-direction_energy-varying_gain_energy))
    gamma_global = uv.mean()/vv.mean()
    gamma_point = np.divide(uv, vv, out=np.zeros_like(uv), where=vv > 0)
    oracle_point_error = np.sum((gamma_point[..., None]*v-u)**2, axis=-1)
    formula_error = uu-np.divide(uv*uv, vv, out=np.zeros_like(uv), where=vv > 0)
    error_identity = np.max(np.abs(oracle_point_error-formula_error))
    global_error = np.sum((gamma_global*v-u)**2, axis=-1)
    frozen_error = np.sum((float(frozen_gamma)*v-u)**2, axis=-1)
    d = v.shape[-1]
    um, vm = float(uu.mean()), float(vv.mean())
    dm, gm, wm = (float(a.mean()) for a in [direction_energy, varying_gain_energy, residual_energy])
    summary = {
        'oracle_energy': um, 'model_energy': vm, 'global_beta': float(beta),
        'global_unaligned_energy': wm,
        'global_unaligned_fraction_of_model_energy': wm/vm,
        'pointwise_direction_energy': dm,
        'pointwise_direction_fraction_of_model_energy': dm/vm,
        'varying_gain_energy': gm, 'varying_gain_fraction_of_model_energy': gm/vm,
        'direction_share_of_global_unaligned_energy': dm/wm if wm else None,
        'varying_gain_share_of_global_unaligned_energy': gm/wm if wm else None,
        'oracle_energy_weighted_beta_std': float(np.sqrt(gm/um)),
        'negative_pointwise_beta_fraction': float(np.mean(local_beta < 0)),
        'zero_oracle_vector_count': int(np.sum(uu == 0)),
        'zero_model_vector_count': int(np.sum(vv == 0)),
        'gamma_global_oracle_ratio': float(gamma_global),
        'frozen_validation_gamma': float(frozen_gamma),
        'original_value_error_per_coordinate': float(np.mean(np.sum((v-u)**2, axis=-1))/d),
        'frozen_gamma_value_error_per_coordinate': float(frozen_error.mean()/d),
        'global_oracle_gamma_value_error_per_coordinate': float(global_error.mean()/d),
        'pointwise_oracle_gamma_value_error_per_coordinate': float(oracle_point_error.mean()/d),
        'pointwise_irreducible_fraction_of_oracle_energy': float(oracle_point_error.mean()/um),
        'split_identity_max_error': float(split_error),
        'pointwise_lower_bound_identity_max_error': float(error_identity),
    }
    saved = {'beta_pointwise': local_beta, 'gamma_pointwise_oracle': gamma_point,
        'oracle_energy_pointwise': uu, 'model_energy_pointwise': vv,
        'pointwise_direction_energy': direction_energy,
        'varying_gain_energy': varying_gain_energy,
        'global_residual_energy': residual_energy,
        'pointwise_oracle_value_error': oracle_point_error,
        'global_oracle_value_error': global_error,
        'frozen_gamma_value_error': frozen_error}
    return summary, saved


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    loaded, hashes = {}, {}
    def load(relative):
        path = args.root/relative
        if str(path) not in loaded:
            hashes[str(path)] = sha256(path)
            loaded[str(path)] = dict(np.load(path))
        return loaded[str(path)]
    tape = load('nb_spaghetti_review/tape.npz')
    oracle = load('population_oracle_spaghetti_nb/arrays.npz')
    source = load('ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz')
    old_path = args.root/'calibration_geometry_v2/summary.json'
    hashes[str(old_path)] = sha256(old_path)
    old = json.loads(old_path.read_text())
    p = source['span_basis']@source['span_basis'].T
    np.testing.assert_allclose(p, np.diag(np.r_[np.ones(20), np.zeros(10)]), atol=2e-13)
    for key in ['queries', 'query_ids', 'scales', 'noise']:
        np.testing.assert_array_equal(tape[key], oracle[key])
    rows = []
    for name in REGISTRY:
        learned = load('nb_spaghetti_review/'+name+'_arrays.npz')
        for key in ['query_ids', 'scales']:
            np.testing.assert_array_equal(tape[key], learned[key])
        saved = {}
        for i, scale in enumerate(tape['scales']):
            previous = next(row for row in old if row['model'] == name and row['lambda'] == float(scale))
            v = scale*(learned['scores'][i]@p)
            u = scale*(oracle['scores'][i]@p)
            summary, arrays = pointwise_alignment(v, u, previous['frozen_clean_target_gamma'])
            np.testing.assert_allclose(summary['model_energy'], previous['model_energy'], rtol=1e-14)
            np.testing.assert_allclose(summary['oracle_energy'], previous['oracle_energy'], rtol=1e-14)
            np.testing.assert_allclose(summary['gamma_global_oracle_ratio'], previous['oracle_ratio'], rtol=1e-14)
            if max(summary['split_identity_max_error'], summary['pointwise_lower_bound_identity_max_error']) > 1e-10:
                raise RuntimeError('Numerical attribution identity failed')
            rows.append({'model': name, 'lambda': float(scale), 'queries': len(tape['queries']),
                         'noise_replicates': tape['noise'].shape[1], **summary})
            for key, value in arrays.items():
                saved.setdefault(key, []).append(value)
        np.savez_compressed(args.output/(name+'_arrays.npz'),
            query_ids=tape['query_ids'], scales=tape['scales'],
            **{key: np.stack(value) for key, value in saved.items()})
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assert all(sha256(path) == value for path, value in hashes.items())
    (args.output/'manifest.json').write_text(json.dumps({
        'source_sha256': sha256(__file__), 'input_hashes': hashes, 'all_inputs_unchanged': True,
        'selection': 'all pre-existing five models and three NB scales, no new evaluation or training',
        'law': 'finite64 clean-query bank times120 fixed Gaussian vectors, evaluated against continuous-law posterior; not the finite-bank conditional posterior',
        'vectors': 'v=(A(b_theta)-A(y))/lambda, u=(b_star-A(y))/lambda; retained in containing20D span within30D',
        'direction_definition': 'pointwise Euclidean perpendicularity to the actual finite-noise oracle displacement, not tangent versus normal geometry',
        'scalar_bound': 'minimum over unrestricted real scalar per saved observation of value error only; oracle diagnostic, not deployable or a field/readout',
        'derivative_warning': 'spatially varying gamma changes divergence by grad(gamma) dot displacement; no response or LID inferred from this value bound',
        'error_units': 'mean squared displacement difference divided by lambda squared and ambient dimension30',
    }, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
