"""Explanatory fixed-data replays for the adjacent-scale extrapolation audit."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.response_full_richardson_20260907 import extrapolate, metrics, sha256, DIMENSIONS


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {'spiral': args.root/'population_oracle_ve_spiral256/fixture_arrays.npz',
             'spaghetti': args.root/'population_oracle_ve_spaghetti256/arrays.npz',
             'vp': args.root/'vp_spaghetti_span/posterior_average_arrays.npz',
             'oracle_csv': args.replay/'oracle_metrics.csv',
             'learned_csv': args.replay/'learned_metrics.csv'}
    before = {str(p): sha256(p) for p in paths.values()}
    spiral = np.load(paths['spiral'])
    scales = spiral['scales']
    local = spiral['local_response']+spiral['local_correction']
    actual = spiral['full']
    local_g, actual_g = extrapolate(local, scales), extrapolate(actual, scales)
    branch_rows = []
    for i in range(len(scales)-1):
        for name, values, proposed in [('full_law', actual, actual_g), ('same_turn_only', local, local_g)]:
            branch_rows.append({'law': name, 'fine_scale': float(scales[i]),
                               'coarse_scale': float(scales[i+1]),
                               **metrics(values[i], proposed[i], 1)})
    np.savez_compressed(args.output/'spiral_branch_counterfactual.npz',
        query_ids=spiral['query_ids'], queries=spiral['queries'], scales=scales,
        local_full=local, actual_full=actual, extrapolated_local_full=local_g,
        extrapolated_actual_full=actual_g, other_turn_mass=spiral['other_turn_mass'])

    oracle, learned = np.load(paths['spaghetti']), np.load(paths['vp'])
    np.testing.assert_array_equal(oracle['queries'], learned['queries'])
    np.testing.assert_array_equal(oracle['query_ids'], learned['query_ids'])
    np.testing.assert_array_equal(oracle['scales'], learned['scales'])
    a = (learned['scales'][1]/learned['scales'][0])**2
    fine = learned['response'][0]-oracle['response'][0]
    coarse = learned['response'][1]-oracle['response'][1]
    propagated = (a*fine-coarse)/(a-1)
    replay = extrapolate(learned['response'], learned['scales'])[0]-extrapolate(oracle['response'], oracle['scales'])[0]
    np.testing.assert_allclose(propagated, replay, rtol=1e-14, atol=2e-15)
    error = {'squared_scale_ratio': float(a), 'fine_error_mean': float(fine.mean()),
             'coarse_error_mean': float(coarse.mean()),
             'positive_fine_contribution_mean': float((a*fine/(a-1)).mean()),
             'subtracted_coarse_contribution_mean': float((coarse/(a-1)).mean()),
             'propagated_error_mean': float(propagated.mean()),
             'fine_error_min': float(fine.min()), 'propagated_error_min': float(propagated.min()),
             'identity_max_error': float(np.abs(propagated-replay).max())}
    np.savez_compressed(args.output/'vp_spaghetti_error_propagation.npz',
        query_ids=learned['query_ids'], fine_error=fine, coarse_error=coarse,
        propagated_error=propagated, squared_scale_ratio=a)

    oracle_rows = list(csv.DictReader(paths['oracle_csv'].open()))
    learned_rows = list(csv.DictReader(paths['learned_csv'].open()))
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.4), sharex=True)
    for column, law in enumerate(DIMENSIONS):
        for row, rows, series in [(0, oracle_rows, 'oracle_'+law), (1, learned_rows, 've_'+law+'_original')]:
            selected = [v for v in rows if v['series'] == series and v['readout'] == 'full']
            x = np.array([float(v['fine_scale']) for v in selected])
            ax = axes[row, column]
            for key, label, style, color in [('original_mae', 'Original full', '-', '#243b53'),
                    ('extrapolated_mae', 'Adjacent-scale extrapolation', '--', '#c44e00')]:
                ax.loglog(x, [float(v[key]) for v in selected], style, marker='o',
                          markersize=3, color=color, linewidth=1.4, label=label)
            ax.grid(alpha=.18, which='both')
            if row == 0:
                ax.set_title(f'{law.title()} (d={DIMENSIONS[law]})')
            if column == 0:
                ax.set_ylabel(('Population oracle' if row == 0 else 'Preserved VE model')+'\nLID MAE')
            if row == 1:
                ax.set_xlabel('Finer scale λ')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Canceling population finite-scale bias need not cancel learned readout error', y=.98)
    fig.subplots_adjust(left=.065, right=.99, bottom=.15, top=.89, hspace=.23, wspace=.36)
    fig.savefig(args.output/'population_vs_learned.pdf', bbox_inches='tight')
    fig.savefig(args.output/'population_vs_learned.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    assert all(sha256(p) == digest for p, digest in before.items())
    (args.output/'details.json').write_text(json.dumps({'spiral_branch_counterfactual': branch_rows,
        'vp_spaghetti_first_pair_response_error': error}, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'helper_sha256': sha256(Path(__file__).with_name('response_full_richardson_20260907.py')),
        'inputs_sha256': before, 'inputs_unchanged': True,
        'scope': 'fixed stored local-turn population restriction and exact learned-error linear propagation; no new training or selection',
        'figure': 'full MAE at every finer adjacent scale; oracle versus fixed original VE, projections remain in main audit CSV'}, indent=2)+'\n')
    print(json.dumps(error, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/response_full_20260907'))
    parser.add_argument('--replay', type=Path, default=Path('artifacts/response_full_20260907/adjacent_richardson'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
