"""Finite-noise calibration/gain interpretation on immutable matched NB arrays.

No coefficient is fitted for deployment here: the original held-out coefficient
is frozen. Oracle-based and clean-trace coefficients are explanatory targets.
The finite 64-clean-point bank is not assigned the continuous population's
conditional law; the continuous oracle ratio is an independent empirical
diagnostic of the population identity, not that bank's exact risk optimum.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.response_full_richardson_20260907 import sha256


REGISTRY = {
    'vp_posterior': ('calibration_vp_spaghetti_posterior/arrays.npz', 'vp_spaghetti_posterior_local/arrays.npz'),
    'vp_noise': ('calibration_vp_spaghetti_noise/arrays.npz', 'vp_spaghetti_noise_local/arrays.npz'),
    've_original': ('calibration_ve_spaghetti/arrays.npz', 've_spaghetti_local/arrays.npz'),
    've_refined_original_lr': ('calibration_ve_spaghetti_refined/original_lr_arrays.npz', 've_spaghetti_refinement_audit/original_lr_average.npz'),
    've_refined_low_lr': ('calibration_ve_spaghetti_refined/low_lr_arrays.npz', 've_spaghetti_refinement_audit/low_lr_average.npz'),
}


def regression_components(v, u, tangent):
    """Tangent is the fixed clean-query frame, even at noisy observations."""
    v, u, tangent = np.asarray(v), np.asarray(u), np.asarray(tangent)
    np.testing.assert_allclose(np.square(tangent).sum(-1), 1., atol=1e-12)
    vt, ut = np.einsum('qkd,qd->qk', v, tangent), np.einsum('qkd,qd->qk', u, tangent)
    v_energy, u_energy = np.square(v).sum(-1).mean(), np.square(u).sum(-1).mean()
    numerator = (v*u).sum(-1).mean()
    vt_energy, ut_energy = np.square(vt).mean(), np.square(ut).mean()
    tn = (vt*ut).mean()
    vn_energy, nn = v_energy-vt_energy, numerator-tn
    gamma_t = tn/vt_energy if vt_energy > 0 else 0.
    gamma_n = nn/vn_energy if vn_energy > 0 else 0.
    error = v-u
    cross_error = (u*error).sum(-1).mean()
    error_energy = np.square(error).sum(-1).mean()
    gamma = numerator/v_energy
    return {'oracle_ratio': float(gamma), 'normal_ratio': float(gamma_n),
            'tangent_ratio': float(gamma_t), 'normal_model_energy_fraction': float(vn_energy/v_energy),
            'normal_oracle_energy_fraction': float(1-ut_energy/u_energy),
            'model_energy': float(v_energy), 'oracle_energy': float(u_energy),
            'oracle_error_cross': float(cross_error), 'error_energy': float(error_energy),
            'weighted_ratio_identity_error': float(abs(gamma-(gamma_t*vt_energy+gamma_n*vn_energy)/v_energy)),
            'error_ratio_identity_error': float(abs(gamma-(u_energy+cross_error)/(u_energy+2*cross_error+error_energy)))}


def calibrated_local_trace(original, gamma, rank):
    return rank+gamma*(np.asarray(original)-rank)


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
    oracle_nb = load('population_oracle_spaghetti_nb/arrays.npz')
    oracle_clean = load('population_oracle_ve_spaghetti256/arrays.npz')
    source = load('ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz')
    p = source['span_basis']@source['span_basis'].T
    p_expected = np.diag(np.r_[np.ones(20), np.zeros(10)])
    projection_error = float(np.max(np.abs(p-p_expected)))
    np.testing.assert_allclose(p, p_expected, atol=2e-13)
    for key in ['queries', 'query_ids', 'scales', 'noise']:
        np.testing.assert_array_equal(tape[key], oracle_nb[key])
    np.testing.assert_array_equal(tape['queries'], oracle_clean['queries'][:len(tape['queries'])])
    np.testing.assert_array_equal(tape['query_ids'], oracle_clean['query_ids'][:len(tape['queries'])])
    tangent = tape['tangent']
    np.testing.assert_allclose(tangent@p, tangent, atol=2e-14)
    rows = []
    for model, (calibration_path, local_path) in REGISTRY.items():
        learned = load('nb_spaghetti_review/'+model+'_arrays.npz')
        calibration, local = load(calibration_path), load(local_path)
        np.testing.assert_array_equal(learned['query_ids'], tape['query_ids'])
        np.testing.assert_array_equal(learned['scales'], tape['scales'])
        np.testing.assert_array_equal(calibration['query_ids'], oracle_clean['query_ids'])
        np.testing.assert_array_equal(calibration['queries'], oracle_clean['queries'])
        np.testing.assert_array_equal(local['query_ids'], oracle_clean['query_ids'])
        saved = {}
        for i, scale in enumerate(tape['scales']):
            j = int(np.argmin(np.abs(calibration['scales']-scale)))
            np.testing.assert_allclose(calibration['scales'][j], scale, rtol=1e-9, atol=0)
            gamma = float(calibration['projected_gamma'][j])
            # lambda*score = displacement/lambda; ratio unaffected, moments dimensionless.
            v = (learned['scores'][i]@p)*scale
            u = (oracle_nb['scores'][i]@p)*scale
            result = regression_components(v, u, tangent)
            y = tape['queries'][:, None, :]+scale*tape['noise']
            ay = source['span_center']+(y-source['span_center'])@p
            target_clean = (tape['queries'][:, None, :]-ay)/scale
            bank_ratio = (v*target_clean).sum()/np.square(v).sum()
            # Exact finite-noise oracle Jacobian, not I on the tangent and 0 on normals.
            tangent20 = local['tangent']
            jo = oracle_clean['jacobian_active'][j]
            ort = np.einsum('qi,qij,qj->q', tangent20, jo, tangent20)
            orn = np.trace(jo, axis1=-2, axis2=-1)-ort
            rt, rn = local['response_tangent'][j], local['response_curve_normal'][j]
            rt_new, rn_new = calibrated_local_trace(rt, gamma, 1), calibrated_local_trace(rn, gamma, 19)
            trace_identity = float(np.max(np.abs(rt_new+rn_new-calibration['projected_response'][j])))
            if trace_identity > 2e-10:
                raise RuntimeError(f'Calibrated trace replay failed: {model}, {scale}, {trace_identity}')
            row = {'model': model, 'lambda': float(scale), 'noisy_clean_queries': len(tangent),
                'noise_replicates': tape['noise'].shape[1], 'clean_derivative_queries': len(rt),
                'frozen_clean_target_gamma': gamma,
                'same_finite_bank_clean_target_ratio': float(bank_ratio),
                **result,
                'posterior_oracle_error_before': float(np.square(v-u).sum(-1).mean()/30),
                'posterior_oracle_error_after_frozen_gamma': float(np.square(gamma*v-u).sum(-1).mean()/30),
                'clean_normal_trace': float(rn.mean()), 'oracle_clean_normal_trace': float(orn.mean()),
                'calibrated_clean_normal_trace': float(rn_new.mean()),
                'normal_trace_matching_gamma_analysis_only': float((19-orn.mean())/(19-rn.mean())),
                'clean_tangent_trace': float(rt.mean()), 'oracle_clean_tangent_trace': float(ort.mean()),
                'calibrated_clean_tangent_trace': float(rt_new.mean()),
                'normal_trace_oracle_mae_before': float(np.abs(rn-orn).mean()),
                'normal_trace_oracle_mae_after': float(np.abs(rn_new-orn).mean()),
                'tangent_trace_oracle_mae_before': float(np.abs(rt-ort).mean()),
                'tangent_trace_oracle_mae_after': float(np.abs(rt_new-ort).mean()),
                'total_trace_identity_max_error': trace_identity}
            rows.append(row)
            for key, value in [('normal_trace', rn), ('normal_trace_calibrated', rn_new),
                               ('normal_trace_oracle', orn), ('tangent_trace', rt),
                               ('tangent_trace_calibrated', rt_new), ('tangent_trace_oracle', ort)]:
                saved.setdefault(key, []).append(value)
        np.savez_compressed(args.output/(model+'_clean_trace.npz'),
            query_ids=oracle_clean['query_ids'], scales=tape['scales'],
            **{key: np.stack(value) for key, value in saved.items()})
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    assert all(sha256(p) == digest for p, digest in hashes.items())
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'input_hashes': hashes, 'all_inputs_unchanged': True,
        'projection_max_error_from_active20': projection_error,
        'law': 'Continuous uniform phase benchmark population; exact b* is this law posterior, not posterior under the fixed64-point bank',
        'empirical_scope': '120 independent Gaussian corruptions at each fixed64 held-out cleanqueries approximate population ratios; frozen gamma was fitted earlier on separate1000 selectionX times8noise',
        'frames': 'exact clean-point tangent held fixed for noisy displacement decomposition; finite-noise oracle tangent component is retained',
        'error_units': 'mean squared model-minus-continuous-oracle posterior error over30 coordinates divided by lambda squared; not exact excess clean-target risk under the finite bank law',
        'clean_trace': 'all256 held-out queries; finite-scale exact oracle Jacobian; not a d-derived fitted gamma',
        'selection': 'all five saved models and three existing NB scales; no new coefficient used for readout',
    }, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
