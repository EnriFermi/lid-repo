"""Matched exact-query Spiral scale window, plotting the fixed comparison ledger."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.response_full_richardson_20260907 import sha256


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha256(args.input)
    rows = json.loads(args.input.read_text())
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    for family, label, color in [('ve_edm', 'Preserved VE / EDM', '#c44e00'),
            ('vp_noise', 'VP noise head', '#008b8b'), ('vp_posterior', 'VP posterior head', '#9652a0')]:
        chosen = [r for r in rows if r['family'] == family]
        x = np.array([r['lambda'] for r in chosen])
        axes[0].semilogx(x, [r['model_response_mean'] for r in chosen], color=color, marker='o', markersize=3, label=label)
        axes[1].loglog(x, [r['model_full_d_mae'] for r in chosen], color=color, marker='o', markersize=3)
    chosen = [r for r in rows if r['family'] == 've_edm']
    axes[0].semilogx(x, [r['oracle_response_mean'] for r in chosen], color='#243b53', linewidth=2, label='Exact population oracle')
    error = np.array([r['oracle_full_d_mae'] for r in chosen])
    axes[1].loglog(x, np.maximum(error, 1e-6), color='#243b53', linewidth=2)
    axes[1].scatter(x[error < 1e-6], np.full((error < 1e-6).sum(), 1e-6), marker='v', color='#243b53')
    axes[0].axhline(1, color='black', linewidth=.8, linestyle=':', label='True local dimension 1')
    axes[1].axhline(.1, color='gray', linewidth=1, linestyle=':', label='Fixed accuracy threshold.1')
    axes[0].set_ylabel('Mean response'); axes[1].set_ylabel('Full-readout MAE to dimension 1')
    for ax in axes:
        ax.set_xlabel('Canonical noise scale λ')
        ax.axvline(.01, color='gray', linestyle='--', linewidth=.8)
        ax.grid(alpha=.18, which='both')
    axes[0].set_ylim(-.5, 33)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .05), ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Spiral: exact geometric resolution versus fixed-model extrapolation', y=.99)
    fig.text(.52, .91, 'All scales λ < .01 are outside the models’ training range.', ha='center', fontsize=9)
    fig.text(.52, .012, 'Oracle MAE below 1e−6 is shown at the numerical reporting floor (downward markers).', ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.28, top=.86, wspace=.23)
    fig.savefig(args.output/'spiral_scale_window.pdf', bbox_inches='tight')
    fig.savefig(args.output/'spiral_scale_window.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    assert sha256(args.input) == before
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': sha256(__file__),
        'input': str(args.input), 'input_sha256': before, 'input_unchanged': True,
        'selection': 'all13 predefined scales and3 fixed models; oracle shown for VE normalization, VP oracle differs negligibly and remains in source ledger',
        'display_floor': 'oracle full MAE only,1e-6; underlying numeric values preserved'}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('artifacts/response_full_20260907/spiral_exact_endpoint_comparison/summary.json'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
