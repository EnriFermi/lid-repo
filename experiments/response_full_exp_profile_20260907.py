"""Saved-query Exp geometry profiles, with exact orthogonal value-error accounting.

No inference, fitted grouping, scale selection, or new posterior is constructed.
All models use the verified common normalized query array and continuous oracle.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

PARTS = ('meridional', 'circumferential', 'surface_normal', 'ambient')
MODELS = {
    'vp_posterior': 'vp_exp_span/posterior_average_arrays.npz',
    'vp_noise': 'vp_exp_span/noise_average_arrays.npz',
    've_original': 've_shared256/e6_exp_pca/arrays.npz',
    've_low_lr_average': 've_exp_refinement/low_lr_average_arrays.npz',
}
TOL = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tube_geometry(raw):
    """True surface frame at same u/theta; retain tiny saved-query round-off."""
    q = np.asarray(raw)[:, :3]
    u = q[:, 0] + 4
    theta = np.arctan2(q[:, 1], q[:, 2])
    r = 3 * np.exp(-u)
    sin, cos = np.sin(theta), np.cos(theta)
    norm = np.sqrt(1+r*r)
    meridional = np.column_stack((np.ones_like(r), -r*sin, -r*cos))/norm[:, None]
    circumferential = np.column_stack((np.zeros_like(r), cos, -sin))
    normal = np.column_stack((r, sin, cos))/norm[:, None]
    frame = np.stack((meridional, circumferential, normal), axis=-1)
    surface = np.column_stack((u-4, r*sin, r*cos))
    # Exact Euclidean distances to the two boundary circles, not to the axis.
    edge0 = np.sqrt(u*u+(r-3)**2)
    edge8 = np.sqrt((8-u)**2+(r-3*np.exp(-8))**2)
    return dict(u=u, theta=theta, radius=r, frame=frame,
                boundary_distance=np.minimum(edge0, edge8),
                boundary0_distance=edge0, boundary8_distance=edge8,
                frame_point_raw=surface, frame_query_difference=np.linalg.norm(q-surface, axis=-1))


def status(advantage):
    return np.where(advantage > TOL, 1, np.where(advantage < -TOL, -1, 0))


def decompose(q, b, bstar, response, oracle_response, scales, frame):
    sigma = np.asarray(scales)[:, None, None]
    delta = (bstar-q[None])/sigma
    error = (b-bstar)/sigma
    model = delta+error
    result = {}
    for label, vec in [('oracle', delta), ('error', error), ('model', model)]:
        coordinates = np.einsum('sqj,qjk->sqk', vec[..., :3], frame)
        for i, part in enumerate(PARTS[:3]):
            result[f'{label}_energy_{part}'] = coordinates[..., i]**2
        result[f'{label}_energy_ambient'] = (vec[..., 3:]**2).sum(-1)
        result[f'{label}_energy'] = (vec**2).sum(-1)
        np.testing.assert_allclose(sum(result[f'{label}_energy_{p}'] for p in PARTS),
                                   result[f'{label}_energy'], rtol=2e-12, atol=2e-12)
    dc = np.einsum('sqj,qjk->sqk', delta[..., :3], frame)
    ec = np.einsum('sqj,qjk->sqk', error[..., :3], frame)
    for i, part in enumerate(PARTS[:3]):
        result[f'cross_{part}'] = 2*dc[..., i]*ec[..., i]
    result['cross_ambient'] = 2*(delta[..., 3:]*error[..., 3:]).sum(-1)
    result['cross'] = 2*(delta*error).sum(-1)
    for part in (*PARTS, ''):
        suffix = '_'+part if part else ''
        np.testing.assert_allclose(result['model_energy'+suffix],
            result['oracle_energy'+suffix]+result['cross'+suffix]+result['error_energy'+suffix],
            rtol=2e-11, atol=2e-11)
    c, cstar = result['model_energy'], result['oracle_energy']
    result.update(response=response, correction=c, full=response+c,
        oracle_response=oracle_response, oracle_correction=cstar,
        oracle_full=oracle_response+cstar, response_error=response-oracle_response)
    for prefix, r, correction in [('', response, c), ('oracle_', oracle_response, cstar),
                                   ('substitute_exact_c_', response, cstar)]:
        advantage = np.abs(r-2)-np.abs(r+correction-2)
        result[prefix+'advantage'] = advantage
        result[prefix+'status'] = status(advantage)
    return result


def summarize(arrays, geometry, raw_sigma, series):
    rows = []
    for i, sigma in enumerate(raw_sigma):
        u = geometry['u']
        masks = {f'u_{lo}_{lo+2}': (u >= lo) & ((u < lo+2) if lo < 6 else (u <= 8))
                 for lo in (0, 2, 4, 6)}
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks.values()), np.ones(len(u)))
        # Fixed secondary partition: a descriptive distance criterion, not a LID selector.
        boundary = geometry['boundary_distance'] <= 2*sigma
        masks.update(all=np.ones(len(u), bool), boundary_2sigma=boundary, outside_boundary_2sigma=~boundary)
        for group, mask in masks.items():
            n = int(mask.sum())
            if not n:
                rows.append(dict(series=series, scale_index=i, group=group, count=0))
                continue
            a = {k: v[i, mask] for k, v in arrays.items()}
            row = dict(series=series, scale_index=i, group=group, count=n,
                raw_sigma=float(sigma), u_mean=float(u[mask].mean()),
                radius_over_sigma_median=float(np.median(geometry['radius'][mask]/sigma)),
                boundary_layer_fraction=float(boundary[mask].mean()))
            for k, v in a.items():
                if 'status' not in k:
                    row[k+'_mean'] = float(v.mean())
            for name in ('response', 'full', 'oracle_response', 'oracle_full'):
                row[name+'_d_mae'] = float(np.abs(a[name]-2).mean())
            row['substitute_exact_c_d_mae'] = float(np.abs(a['response']+a['oracle_correction']-2).mean())
            for prefix in ('', 'oracle_', 'substitute_exact_c_'):
                for v, label in [(-1, 'harms'), (0, 'tie'), (1, 'helps')]:
                    row[prefix+label+'_fraction'] = float(np.mean(a[prefix+'status'] == v))
            for oi, oname in [(-1, 'harms'), (0, 'tie'), (1, 'helps')]:
                for mi, mname in [(-1, 'harms'), (0, 'tie'), (1, 'helps')]:
                    joint = (a['oracle_status'] == oi) & (a['status'] == mi)
                    row[f'oracle_{oname}_model_{mname}_count'] = int(joint.sum())
            reverse = (a['oracle_status'] == 1) & (a['status'] == -1)
            row['reversal_rescued_by_exact_c_count'] = int((reverse & (a['substitute_exact_c_status'] == 1)).sum())
            rows.append(row)
    return rows


def plot_profile(rows, scales, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#28639b', '#119b84', '#db9433', '#ae496e']
    groups = [f'u_{i}_{i+2}' for i in (0, 2, 4, 6)]
    fig, axes = plt.subplots(1, 5, figsize=(15.6, 3.2), sharex=True, sharey=True)
    for ax, series in zip(axes, ['oracle', *MODELS]):
        source_series = 'vp_posterior' if series == 'oracle' else series
        prefix = 'oracle_' if series == 'oracle' else ''
        for color, group in zip(colors, groups):
            selected = [r for r in rows if r['series'] == source_series and r['group'] == group]
            selected.sort(key=lambda r:r['scale_index'])
            ax.plot(scales, [r[prefix+'advantage_mean'] for r in selected], marker='.', color=color,
                    label=group.replace('u_', 'u∈[').replace('_', ',')+')')
        ax.axhline(0, color='.5', lw=.8)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=.025)
        ax.set_title(series.replace('_', ' ')); ax.set_xlabel('physical λ (normalized data)')
        ax.grid(alpha=.16)
    axes[0].set_ylabel('MAE(response) − MAE(full)\npositive: correction helps')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle('Actual Exp: the same correction has different consequences along the surface', y=1.01)
    fig.tight_layout(); fig.savefig(output/'profile.pdf', bbox_inches='tight')
    fig.savefig(output/'profile.png', dpi=180, bbox_inches='tight'); plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    oracle_path = args.base/'population_oracle_ve_exp256/fixture_arrays.npz'
    oracle = dict(np.load(oracle_path)); hashes = {str(oracle_path): sha(oracle_path)}
    geometry = tube_geometry(oracle['queries_raw'])
    np.testing.assert_allclose(geometry['frame'].transpose(0,2,1)@geometry['frame'],
        np.broadcast_to(np.eye(3), (256,3,3)), atol=1e-14)
    rows, checks = [], {}
    for series, path in MODELS.items():
        path = args.base/path; hashes[str(path)] = sha(path); model = dict(np.load(path))
        np.testing.assert_array_equal(model['query_ids'], oracle['query_ids'])
        np.testing.assert_array_equal(model['queries'], oracle['queries'])
        np.testing.assert_allclose(model['scales'], oracle['scales'], rtol=1e-9, atol=0)
        np.testing.assert_allclose(model['query_raw'][:,:3], oracle['queries_raw'], atol=3e-7, rtol=0)
        arrays = decompose(model['queries'], model['posterior'], oracle['posterior'],
            model['response'], oracle['response'], model['scales'], geometry['frame'])
        np.testing.assert_allclose(arrays['correction'], model['correction_original'], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(arrays['full'], model['full'], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(arrays['oracle_correction'], oracle['correction'], rtol=1e-9, atol=1e-9)
        raw_sigma = model['scales']*float(oracle['normalization_scale'])
        group_rows = summarize(arrays, geometry, raw_sigma, series)
        for r in group_rows:r['lambda'] = float(model['scales'][r['scale_index']])
        rows.extend(group_rows)
        np.savez_compressed(args.output/f'{series}.npz', **arrays, **geometry,
            queries=model['queries'], queries_raw=oracle['queries_raw'], query_ids=model['query_ids'],
            scales=model['scales'], raw_sigma=raw_sigma)
        checks[series] = {'saved_c_max_difference':float(np.max(np.abs(arrays['correction']-model['correction_original']))),
            'scale_relative_difference':float(np.max(np.abs(model['scales']/oracle['scales']-1))),
            'decomposition_max_difference':float(np.max(np.abs(arrays['correction']-arrays['oracle_correction']-arrays['cross']-arrays['error_energy'])))}
    (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=keys);writer.writeheader();writer.writerows(rows)
    plot_profile(rows, model['scales'], args.output)
    for path, digest in hashes.items():assert sha(path) == digest
    (args.output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__), input_sha256=hashes,
        queries=256, models=list(MODELS), all_scales=model['scales'].tolist(), rows=len(rows), checks=checks,
        u_bins=[0,2,4,6,8], boundary_criterion='distance to nearer actual boundary circle <=2 raw sigma',
        frame_point_max_query_difference=float(geometry['frame_query_difference'].max()),
        frame_rounding_scope='Analytic true surface frame at same u and polar angle as the saved round-trip query; query unchanged.',
        oracle_scope='Continuous original uniform-u, uniform-angle narrowing surface; no cylinder substitution.',
        scale_scope='Oracle b/R reused at stored scales within relative1e-9; C recomputed at model scale for exact signed identities.',
        status_tolerance=TOL, substitution_scope='Replacing C alone is algebraic attribution, not a realizable field or deployable selector.',
        no_inference_or_training=True, inputs_unchanged=True), indent=2)+'\n')
    print(json.dumps({'rows':len(rows), 'checks':checks, 'output':str(args.output)}, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('artifacts/response_full_20260907'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
