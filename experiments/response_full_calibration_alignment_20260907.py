"""Exact scalar-alignment decomposition of the saved noisy posterior moments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from experiments.response_full_richardson_20260907 import sha256


def alignment(oracle_energy, model_energy, oracle_error_cross):
    u, v = float(oracle_energy), float(model_energy)
    if u <= 0 or v <= 0:
        raise ValueError('Positive vector-field energies required')
    cross = u+float(oracle_error_cross)
    beta = cross/u
    residual = v-cross*cross/u
    if residual < -1e-10*max(u, v):
        raise ValueError('Moments violate Cauchy-Schwarz')
    inverse = 1/beta if beta != 0 else None
    gamma = cross/v
    return {'oracle_aligned_gain_beta': beta, 'inverse_aligned_gain': inverse,
        'unaligned_energy': residual, 'unaligned_model_energy_fraction': residual/v,
        'gamma_from_alignment': gamma,
        'gamma_alignment_identity_error': abs(gamma-inverse*(1-residual/v)) if inverse is not None else 0.,
        'posterior_oracle_error_if_inverse_gain': (inverse*inverse*v-2*inverse*cross+u)/30 if inverse is not None else None,
        'posterior_oracle_error_at_oracle_ratio': (gamma*gamma*v-2*gamma*cross+u)/30}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha256(args.input)
    rows = json.loads(args.input.read_text())
    output = []
    for row in rows:
        result = alignment(row['oracle_energy'], row['model_energy'], row['oracle_error_cross'])
        if abs(result['gamma_from_alignment']-row['oracle_ratio']) > 1e-12:
            raise RuntimeError('Original oracle ratio did not replay')
        output.append({'model': row['model'], 'lambda': row['lambda'],
            'frozen_clean_target_gamma': row['frozen_clean_target_gamma'],
            'clean_normal_trace_gain': 1/row['normal_trace_matching_gamma_analysis_only'], **result})
    assert sha256(args.input) == before
    (args.output/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'input_sha256': before, 'input': str(args.input), 'input_unchanged': True,
        'decomposition': 'v=beta*u+w, empirical mean inner product <u,w>=0; normal/tangent geometry is not the definition of w',
        'selection': 'all fifteen pre-existing model/scale cells',
        'inverse_gain': 'analysis-only counterfactual, not a newly deployed or d-selected coefficient'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/response_full_20260907/calibration_geometry_v2/summary.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
