"""Two-scale correction diagnostic, without LID-label parameter selection.

Uses C_1 = ||4(b_{lambda/2}-x)-(b_lambda-x)||^2/lambda^2.
This cancels a quadratic displacement term but amplifies a scale-independent
model offset. Neither a universal improvement nor a boundary certificate is
assumed. The original response is unchanged. Projection, when requested,
acts on the canonical posterior and its corresponding cached response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve import canonical_posterior
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS
from models.training import load_checkpoint


def first_order_correction(b, b_half, x, scale):
    return (4 * (b_half - x) - (b - x)).square().sum(-1) / scale**2


def project(values, center, basis):
    return center + ((values-center) @ basis) @ basis.T


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    cached = np.load(args.arrays)
    before = sha256(args.checkpoint)
    x = torch.from_numpy(cached['queries']).double().to(args.device)
    center = torch.from_numpy(cached['span_center']).double().to(args.device)
    basis = torch.from_numpy(cached['span_basis']).double().to(args.device)
    if args.family == 'vp':
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = load_field(payload, args.kind, args.mode, args.device)
        predict = lambda points, scale: posterior(model, points, scale, args.kind)
        low = payload['config']['lambda_min']
    else:
        result = load_checkpoint(args.checkpoint, device=args.device)
        result.model.double().eval()
        predict = lambda points, scale: canonical_posterior(result, points, scale)
        low = .01  # Immutable source campaign's declared sigma_min.
    scales = cached['scales']
    half, rows, output_arrays = [], [], {}
    d = DATASET_SPECS[args.dataset].true_lid
    with torch.no_grad():
        for index, scale in enumerate(scales):
            b = predict(x, float(scale))
            b_half = predict(x, float(scale)/2)
            np.testing.assert_allclose(b.cpu().numpy(), cached['posterior'][index], atol=1e-9, rtol=1e-9)
            half.append(b_half.cpu().numpy())
            for variant in ['original', 'projected']:
                bx = b if variant == 'original' else project(b, center, basis)
                bh = b_half if variant == 'original' else project(b_half, center, basis)
                r = torch.from_numpy(cached['response' if variant == 'original' else 'response_span'][index]).to(x)
                c = (bx-x).square().sum(-1)/float(scale)**2
                c_first = first_order_correction(bx, bh, x, float(scale))
                full, proposed = r+c, r+c_first
                for name, value in [('correction', c), ('correction_first', c_first), ('full', full), ('full_first', proposed)]:
                    output_arrays.setdefault(variant+'_'+name, []).append(value.cpu().numpy())
                rows.append({'dataset': args.dataset, 'family': args.family, 'kind': args.kind,
                    'mode': args.mode, 'variant': variant, 'lambda': float(scale),
                    'half_scale_in_training_range': bool(scale/2 >= low),
                    'response_mae': float((r-d).abs().mean()), 'full_mae': float((full-d).abs().mean()),
                    'full_first_mae': float((proposed-d).abs().mean()), 'correction_mean': float(c.mean()),
                    'correction_first_mean': float(c_first.mean()),
                    'first_beats_full_fraction': float(((proposed-d).abs() < (full-d).abs()).double().mean())})
    assert sha256(args.checkpoint) == before
    np.savez_compressed(args.output/'arrays.npz', queries=cached['queries'], query_ids=cached['query_ids'],
        scales=scales, posterior_half=np.stack(half), **{k: np.stack(v) for k,v in output_arrays.items()})
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'checkpoint_sha256': before,
        'checkpoint_unchanged': True, 'arrays_sha256': sha256(args.arrays), 'source_sha256': sha256(__file__),
        'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'training_scale_min': low, 'readout': 'R_lambda + norm(4*(b_half-x)-(b_lambda-x))^2/lambda^2',
        'selection': 'fixed coefficient4 and ratio2 derived from linear/quadratic scale expansion; no labels used',
        'warning': 'Not guaranteed outside small-scale expansion; constant model offset amplified by factor3'}, indent=2)+'\n')
    print(json.dumps(rows, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--arrays', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--family', choices=['vp','ve'], required=True)
    p.add_argument('--kind', choices=['posterior','noise'], default='posterior')
    p.add_argument('--mode', choices=['last','average'], default='average')
    p.add_argument('--device', default='cpu')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
