"""Post-hoc diagnostic that separates probe radius from VE noise scale.

The prospective campaign used perturbations ``x +/- sigma*z``.  This script
keeps the denoiser condition at ``sigma`` but probes its input Jacobian with
``x +/- epsilon*z``, where ``epsilon = ratio*sigma``.  It therefore tests
whether disagreement with the clean-query FLIPD response is caused by shell
averaging rather than by stochastic-trace noise.  It is explicitly diagnostic:
its ratios were examined after the primary five-manifold outcome and cannot
replace that outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from experiments.ve_known_lid_recoverability_campaign import (  # noqa: E402
    DEFAULT_DATASETS,
    SELECTOR_ANCHORS,
    _normalise,
    frozen_sparse_selector,
    make_splits,
)
from models.training import (  # noqa: E402
    TrainingResult,
    diffusion_denoiser_output,
    load_checkpoint,
)


def _read_curve(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty curve file: {path}")
    return rows


def _local_response(
    result: TrainingResult,
    raw_query: np.ndarray,
    noise: torch.Tensor,
    sigma: float,
    *,
    probe_ratio: float,
    batch_size: int,
) -> float:
    query = _normalise(result, np.ascontiguousarray(raw_query, dtype=np.float32))
    epsilon = probe_ratio * sigma
    plus = query[:, None, :] + epsilon * noise
    minus = query[:, None, :] - epsilon * noise
    inputs = torch.stack((plus, minus), dim=2).reshape(-1, query.shape[1])
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            condition = torch.full(
                (len(batch),), sigma, device=batch.device, dtype=batch.dtype
            )
            outputs.append(
                diffusion_denoiser_output(
                    result.model,
                    batch,
                    condition,
                    preconditioning=result.config.diffusion_preconditioning,
                    data_std=result.config.diffusion_data_std,
                )
            )
    denoised = torch.cat(outputs).reshape(
        query.shape[0], noise.shape[1], 2, query.shape[1]
    )
    directional = (
        ((denoised[:, :, 0] - denoised[:, :, 1]) / (2.0 * epsilon) * noise)
        .sum(dim=(1, 2))
        .div(noise.square().sum(dim=(1, 2)))
    )
    return float(query.shape[1] * directional.mean().item())


def run_dataset(args: argparse.Namespace, name: str) -> dict[str, Any]:
    dataset_dir = args.output_dir.expanduser().resolve() / name
    result = load_checkpoint(dataset_dir / "model.pt", device=args.device)
    _, validation, _ = make_splits(
        name,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        data_seed=args.data_seed,
    )
    query_rng = np.random.default_rng(args.query_seed)
    indices = query_rng.choice(len(validation), args.query_count, replace=False)
    query = np.ascontiguousarray(validation[indices])
    parameter = next(result.model.parameters())
    noise = torch.as_tensor(
        np.random.default_rng(args.noise_seed)
        .standard_normal(
            (args.query_count, args.probe_count, result.model.config.ambient_dim)
        )
        .astype(np.float32),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    source_rows = _read_curve(dataset_dir / "curves.csv")
    scales = np.asarray([float(row["sigma"]) for row in source_rows])
    exact_response = np.asarray(
        [float(row["validation_response_mean"]) for row in source_rows]
    )
    records: list[dict[str, Any]] = []
    for ratio in args.probe_ratios:
        estimate = np.asarray(
            [
                _local_response(
                    result,
                    query,
                    noise,
                    float(sigma),
                    probe_ratio=ratio,
                    batch_size=args.batch_size,
                )
                for sigma in scales
            ]
        )
        anchor_response: list[float] = []
        for anchor in SELECTOR_ANCHORS:
            index = int(np.argmin(np.abs(np.log(scales / anchor))))
            if not math.isclose(float(scales[index]), anchor, rel_tol=1.0e-12):
                raise RuntimeError("diagnostic curve is missing a selector anchor")
            anchor_response.append(float(estimate[index]))
        q = 1.0 - np.asarray(anchor_response) / result.model.config.ambient_dim
        selection = frozen_sparse_selector(
            SELECTOR_ANCHORS, q, ambient_dim=result.model.config.ambient_dim
        )
        records.append(
            {
                "probe_ratio": float(ratio),
                "pearson_to_exact_clean_response": float(
                    np.corrcoef(estimate, exact_response)[0, 1]
                ),
                "mae_to_exact_clean_response": float(
                    np.mean(np.abs(estimate - exact_response))
                ),
                "rmse_to_exact_clean_response": float(
                    np.sqrt(np.mean(np.square(estimate - exact_response)))
                ),
                "anchor_response_dimension_units": anchor_response,
                "selector_if_substituted_posthoc": selection,
            }
        )
    report = {
        "schema_version": 1,
        "dataset": name,
        "scope": "posthoc mechanism diagnostic; not a prospective selector result",
        "definition": (
            "Central finite-difference denoiser gain at x +/- epsilon*z with "
            "epsilon=probe_ratio*sigma and denoiser condition sigma"
        ),
        "query_count": args.query_count,
        "probe_count": args.probe_count,
        "query_seed": args.query_seed,
        "noise_seed": args.noise_seed,
        "records": records,
    }
    (dataset_dir / "local_probe_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-ratios", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--query-seed", type=int, default=20_260_906)
    parser.add_argument("--noise-seed", type=int, default=20_260_907)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32_768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = [run_dataset(args, name) for name in args.datasets]
    summary = {
        "schema_version": 1,
        "scope": "posthoc mechanism diagnostic; primary campaign remains unchanged",
        "datasets": [report["dataset"] for report in reports],
        "records": {
            report["dataset"]: report["records"] for report in reports
        },
    }
    path = args.output_dir.expanduser().resolve() / "local_probe_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for report in reports:
        best = min(
            report["records"], key=lambda item: item["mae_to_exact_clean_response"]
        )
        print(
            f"{report['dataset']}: best ratio={best['probe_ratio']:.3g} "
            f"corr={best['pearson_to_exact_clean_response']:.4f} "
            f"mae={best['mae_to_exact_clean_response']:.4f} "
            f"selector={best['selector_if_substituted_posthoc']['status']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
