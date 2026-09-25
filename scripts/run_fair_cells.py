"""Local runner for `experiments.fair_campaign run` over several cells and GPUs.

The cluster submitter targets a remote scheduler; this is its local equivalent. It keeps one
subprocess per slot, assigns GPUs round-robin, resumes by skipping cells that already hold a
complete receipt, and never modifies an existing output directory.

    python -m scripts.run_fair_cells --variant logistic_flowmatching \
        --cells e6/e6_exp_pca/dataset,e6/e6_exp_pca/coefficients \
        --canonical-root data/lid_benchmarks_exact/benchmarks \
        --output runs/logistic --gpus 0,1 --per-gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def cell_directory(output: Path, variant: str, cell: str) -> Path:
    return output/variant/cell.replace('/', '__')


def completed(directory: Path) -> bool:
    receipt = directory/'complete.json'
    if not receipt.is_file():
        return False
    return json.loads(receipt.read_text()).get('status') == 'complete'


def command(args, cell: str, directory: Path) -> list[str]:
    value = [sys.executable, '-m', 'experiments.fair_campaign', 'run',
             '--variant', args.variant, '--cell', cell, '--output', str(directory)]
    if args.canonical_root:
        value += ['--canonical-root', str(args.canonical_root)]
    if args.generated_root:
        value += ['--generated-root', str(args.generated_root)]
    if args.steps:
        value += ['--steps', str(args.steps)]
    if args.preflight:
        value += ['--preflight']
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--cells', required=True, help='comma-separated cell keys, or a file with one per line')
    parser.add_argument('--canonical-root', type=Path)
    parser.add_argument('--generated-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--per-gpu', type=int, default=1)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    source = Path(args.cells)
    cells = [line.strip() for line in source.read_text().splitlines() if line.strip()] \
        if source.is_file() else [c.strip() for c in args.cells.split(',') if c.strip()]
    slots = [gpu for gpu in args.gpus.split(',') if gpu.strip() for _ in range(args.per_gpu)]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'logs').mkdir(exist_ok=True)
    pending = [c for c in cells if not completed(cell_directory(args.output, args.variant, c))]
    print(json.dumps(dict(cells=len(cells), pending=len(pending), slots=len(slots))), flush=True)
    running: dict[str, tuple[subprocess.Popen, str, float]] = {}
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < len(slots):
            gpu = slots[len(running)] if len(running) < len(slots) else slots[0]
            busy = {g for _, g, _ in running.values()}
            free = [g for g in slots if slots.count(g) > sum(1 for _, other, _ in running.values() if other == g)]
            gpu = free[0] if free else gpu
            cell = pending.pop(0)
            directory = cell_directory(args.output, args.variant, cell)
            if directory.exists() and not completed(directory):
                print(json.dumps(dict(cell=cell, skipped='existing incomplete output directory')), flush=True)
                failures.append(cell)
                continue
            log = (args.output/'logs'/f"{cell.replace('/', '__')}.log").open('a')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                               CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'))
            process = subprocess.Popen(command(args, cell, directory), cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            running[cell] = (process, gpu, time.monotonic())
            print(json.dumps(dict(started=cell, gpu=gpu, pid=process.pid)), flush=True)
        time.sleep(10)
        for cell, (process, gpu, started) in list(running.items()):
            if process.poll() is None:
                continue
            del running[cell]
            elapsed = round(time.monotonic()-started)
            ok = process.returncode == 0 and completed(cell_directory(args.output, args.variant, cell))
            if not ok:
                failures.append(cell)
            print(json.dumps(dict(finished=cell, gpu=gpu, seconds=elapsed, ok=ok,
                                  returncode=process.returncode, remaining=len(pending)+len(running))), flush=True)
    print(json.dumps(dict(status='failed' if failures else 'complete', failed=failures)), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
