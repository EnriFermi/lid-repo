"""Complete E1/E5 (unknown-LID) cells from already trained checkpoints, without retraining.

The September 16 native campaign trained every E1/E5 cell, but the reference-mean Kneedle failed
for three variants, so every dependent cell inherited `reference_selection_failed`:

* ``t_flowmatching``: native support is (0, inf); the evaluation window replaced inf by 64 and the
  reference mean curve is still falling there (knee_at_boundary). Training samples t ~ U(0, 1)
  with lambda = k (1 - t) / t, so about 1.9% of training draws lie in [64, 999]. This script
  re-measures the same checkpoints on 29 geometric scales from the model's lower bound to
  ``--upper`` (default 999, the RF window) and applies the unchanged reference Kneedle.
* ``scale_conditioned_nf``: the trained epsilon range with the OLS5 stencil ends at 64, so the
  window cannot be widened. The reference curve has a plateau at the ambient dimension, and kneed
  returns its first point (knee_at_boundary). With ``--fallback upper_bound`` the failed reference
  is replaced by the window's upper bound; the cell is recorded as ``scale_unresolved`` and the
  receipt keeps the Kneedle failure. This is a forced operating point, not a successful selection.
* ``schrodinger_bridge`` is not handled: its support [0.14, 0.63] is fixed by the bridge and its
  mean curve rises above the ambient dimension at the upper end.

Everything else follows ``experiments.fair_campaign.run``: the same checkpoints (SHA-256 checked
against the archived receipts), data files, partition and holdout queries, readouts, frozen
reference scale for dependent cells and ``fair_outputs`` metrics. Outputs are completion rows in the
campaign layout, so ``scripts.score_cells --skip-verify`` scores them with the table code.
``fair_campaign verify`` does not apply: these rows are re-measurements, marked by ``reevaluation``.

    python -m scripts.complete_unknown_lid --variant t_flowmatching \\
        --archive-cells <run>/cells/t_flowmatching --checkpoints <ckpts>/cells/t_flowmatching \\
        --canonical-root data/lid_benchmarks_exact/benchmarks --output runs/e1e5_reeval --upper 999
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from datasets.registry import load_split
from experiments import fair_data
from experiments import fair_measurements as measurement
from experiments import fair_outputs as outputs
from experiments import global_campaign as v1
from experiments import global_campaign_v2 as v2
from experiments.fair_campaign import dataset_spec, inventory, prediction_spec
from models import training
from models.native_tasks import scale_support, valid_scales

ROOT = Path(__file__).resolve().parents[1]
FALLBACK_RULE = 'window_upper_bound_after_failed_reference_kneedle'


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_dir(key: str) -> str:
    return key.replace('/', '__')


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def evaluation_grid(config, upper: float | None) -> np.ndarray:
    """The campaign grid, or 29 geometric scales up to `upper` for an infinite native support."""
    if upper is None:
        return measurement.model_scales(config)
    lo_native, hi_native = scale_support(config)
    if not math.isinf(hi_native):
        raise ValueError('--upper only applies to models whose native scale support is unbounded')
    lo, _ = measurement.model_scale_bounds(config)
    grid = np.geomspace(lo, upper, 29)
    grid[0], grid[-1] = lo, upper
    if not valid_scales(config, grid).all():
        raise ValueError('extended grid leaves the native support')
    return grid


def load_queries(cell, root: Path, split: str, archived: dict, cfg):
    loaded = load_split(root, dataset_spec(cell), split, representation=cell.representation, mmap_mode='r')
    hashes = fair_data.validate_loaded(cell, loaded, split)
    if hashes != archived[f'{split}_files_sha256']:
        raise ValueError(f'{cell.key}: {split} files differ from the archived receipt')
    features = loaded.features.reshape(len(loaded.features), -1)
    if split == 'test':
        labels = np.asarray([], dtype=np.int64) if loaded.labels is None else np.asarray(loaded.labels).reshape(-1)
        return features, labels
    part = v1.partition_source_train(features, loaded.lid, selection=cfg['campaign']['selection'], seed=0)
    holdout = part.selection_features
    if v1._array_sha(part.selection_indices[:len(holdout)]) != archived['effective_holdout_indices_sha256']:
        raise ValueError(f'{cell.key}: holdout queries differ from the archived receipt')
    return holdout, part.selection_indices[:len(holdout)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['t_flowmatching', 'scale_conditioned_nf'])
    parser.add_argument('--archive-cells', type=Path, required=True, help='<run>/cells/<variant> with receipts')
    parser.add_argument('--checkpoints', type=Path, required=True, help='<root>/cells/<variant> with model.pt')
    parser.add_argument('--canonical-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upper', type=float, help='extended upper scale for unbounded native support')
    parser.add_argument('--fallback', choices=['none', 'upper_bound'], default='none')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    measurement.validate_runtime()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, cells = inventory()
    by_key = {c.key: c for c in cells}
    keys = [c.key for c in cells if c.suite_id in ('e1', 'e5') and c.target_policy != 'known_lid']
    readouts = list(outputs.readouts(args.variant))
    primary = readouts[0]
    code = dict(script_sha256=sha(Path(__file__)),
                git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip())
    out_root = args.output / args.variant
    out_root.mkdir(parents=True, exist_ok=True)
    finished = {}
    # references (reference_dataset == dataset) first
    for key in sorted(keys, key=lambda k: by_key[k].reference_dataset != by_key[k].dataset):
        cell = by_key[key]
        directory = out_root / cell_dir(key)
        if (directory / 'complete.json').is_file():
            finished[key] = json.loads((directory / 'complete.json').read_text())
            print(json.dumps(dict(cell=key, skipped='already complete')), flush=True)
            continue
        start = time.monotonic()
        archived_dir = args.archive_cells / cell_dir(key)
        archived = json.loads((archived_dir / 'complete.json').read_text())
        if archived['variant'] != args.variant or archived['cell_key'] != key:
            raise ValueError(f'{key}: archived receipt belongs to another cell')
        checkpoint = args.checkpoints / cell_dir(key) / 'model.pt'
        if sha(checkpoint) != archived['checkpoint_sha256']:
            raise ValueError(f'{key}: checkpoint differs from the archived receipt')
        trained = training.load_checkpoint(checkpoint, device=args.device)
        config = trained.config
        grid = evaluation_grid(config, args.upper)
        spec = prediction_spec(args.variant, archived['resolved']['geometry'], config)
        spec['lambda_support'] = [float(grid[0]), float(grid[-1])]

        def predict(query, scales, readout=primary):
            return v2._prediction_curve(training.predict_lid, trained, query, np.asarray(scales), model=spec,
                                        seed=0, batch_size=128, readout=readout)

        reference_key = f'{cell.suite_id}/{cell.reference_dataset}/{cell.representation}'
        is_reference = reference_key == key
        directory.mkdir(parents=True)
        row = dict(archived)
        provenance = dict(code, source_receipt_sha256=sha(archived_dir / 'complete.json'),
                          archived_selected_lambda=archived['selected_lambda'],
                          archived_selection=archived['selection'],
                          archived_measurement_status=archived['measurement_status'],
                          candidate_lambdas=grid.tolist(), upper=args.upper, fallback=args.fallback,
                          retrained=False)
        if is_reference:
            holdout, query_ids = load_queries(cell, args.canonical_root, 'train', archived, cfg)
            curve = predict(holdout, grid)
            np.savez_compressed(directory / 'holdout_curve.npz', scales=grid, prediction=curve,
                                target=np.asarray([]), query_ids=query_ids)
            if args.upper is None:
                with np.load(archived_dir / 'holdout_curve.npz', allow_pickle=False) as z:
                    measurement.require_grid(z['scales'], grid)
                    provenance['max_abs_diff_vs_archived_holdout'] = float(np.max(np.abs(z['prediction'] - curve)))
            index, selection = measurement.reference_selection(curve, grid)
            if index is not None:
                selected, status = float(grid[index]), 'selected'
            elif args.fallback == 'upper_bound':
                selected, status = float(grid[-1]), 'scale_unresolved'
                selection = dict(selection, fallback_rule=FALLBACK_RULE, selected_index=len(grid) - 1,
                                 selected_lambda=selected)
            else:
                selected, status = None, 'selection_failed'
            row['holdout_curve_sha256'] = sha(directory / 'holdout_curve.npz')
            reference = None
        else:
            reference = finished[reference_key]
            selected, status = reference['selected_lambda'], reference['measurement_status']
            selection = dict(status=reference['selection']['status'] if status != 'scale_unresolved' else 'scale_unresolved',
                             criterion='reuse_reference_mean_kneedle', reference_dataset=cell.reference_dataset,
                             protocol=measurement.REFERENCE_PROTOCOL, coordinate='log_lambda',
                             candidate_lambdas=grid.tolist())
            if status == 'selection_failed':
                selection.update(status='selection_failed', failure_reason='reference_selection_failed',
                                 reference_failure_reason=reference['selection']['failure_reason'])
            if 'fallback_rule' in reference['selection']:
                selection['reference_fallback_rule'] = reference['selection']['fallback_rule']
            row['holdout_curve_sha256'] = None
            ref_dir = out_root / cell_dir(reference_key)
            (directory / 'reference_complete.json').write_bytes((ref_dir / 'complete.json').read_bytes())
            (directory / 'reference_test_predictions.npz').write_bytes((ref_dir / 'test_predictions.npz').read_bytes())
            row['reference_receipt_sha256'] = sha(ref_dir / 'complete.json')
        queries, labels = load_queries(cell, args.canonical_root, 'test', archived, cfg)
        if v1._array_sha(labels) != archived['test_labels_sha256'] or len(queries) != archived['test_n']:
            raise ValueError(f'{key}: test queries differ from the archived receipt')
        predictions = dict(query_ids=np.arange(len(queries), dtype=np.int64), labels=labels,
                           target=np.asarray([]))
        metrics = {}
        if selected is not None:
            for readout in readouts:
                if args.variant == 'scale_conditioned_nf' and readout == 'fixed_likelihood':
                    trained.model.double()
                values = predict(queries, [selected], readout)[:, 0]
                predictions[readout] = values
                metrics[readout] = measurement.metric(values, None,
                    'float64' if args.variant == 'scale_conditioned_nf' and readout == 'fixed_likelihood' else 'float32')
        np.savez_compressed(directory / 'test_predictions.npz', **predictions)
        row.update(selected_lambda=selected, selection=selection, measurement_status=status, metrics=metrics,
                   test_predictions_sha256=sha(directory / 'test_predictions.npz'),
                   reevaluation=dict(provenance, seconds=time.monotonic() - start))
        if reference is None:
            row['reference_metrics'] = outputs.reference_metrics(row, predictions, row, predictions)
        else:
            with np.load(directory / 'reference_test_predictions.npz', allow_pickle=False) as base:
                row['reference_metrics'] = outputs.reference_metrics(row, predictions, reference, base)
        write_json(directory / 'selection.json', dict(cell=row['cell'], variant=args.variant,
                   selected_lambda=selected, selection=selection, measurement_status=status))
        write_json(directory / 'complete.json', row)
        finished[key] = row
        print(json.dumps(dict(cell=key, status=status, selected_lambda=selected,
                              failure=selection.get('failure_reason'), metrics=metrics,
                              seconds=round(time.monotonic() - start))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
