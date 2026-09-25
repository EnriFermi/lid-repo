"""Table scores for the cells present in a run directory.

`fair_campaign aggregate` requires the complete matrix. When only part of the matrix has been
trained (for example the known-LID cells without E1/E5), this script computes the same scores from
the same receipts and the same `fair_outputs` code, and states which score is out of scope.

    python -m scripts.score_cells --root runs/logistic --variant logistic_flowmatching
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from experiments import fair_outputs as outputs
from experiments.fair_campaign import verify_measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()
    entries, cells = [], []
    for path in sorted(args.root.rglob('complete.json')):
        row = json.loads(path.read_text())
        if args.variant and row['variant'] != args.variant:
            continue
        if not args.skip_verify:
            verify_measurements(path, row)
        entries.append((row, path.parent))
        cells.append(row['cell'])
    if not entries:
        raise SystemExit('no receipts found')
    records = outputs.result_records(entries)
    scores = [s for s in outputs.table_scores(records, cells)
              if (not args.variant or s['model_variant'] == args.variant)
              and s['status'] != 'not_in_scope']
    value = dict(root=str(args.root), variant=args.variant, cells=len(entries),
                 cell_keys=sorted({row['cell_key'] for row, _ in entries}),
                 result_records=records, table_scores=scores)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        outputs.write_csv(args.output.with_name(args.output.stem+'.results.csv'), records)
        outputs.write_csv(args.output.with_name(args.output.stem+'.scores.csv'), scores)
    for score in scores:
        print(json.dumps({k: score[k] for k in ('model_variant', 'readout', 'known_selection_protocol',
                                                'score', 'value', 'status', 'expected_cells',
                                                'recorded_cells', 'boundary_selected_cells')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
