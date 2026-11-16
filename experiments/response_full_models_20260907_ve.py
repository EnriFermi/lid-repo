"""Exact query-level VE readout audit on shared complex-geometry fixtures.

The only inference intervention is projection onto an affine data span fitted
from training observations, without intrinsic-dimension labels. The ordinary
EDM denoiser includes its analytic skip and output factors in every derivative.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import (
    correction_components, fit_affine_span, projected_trace, sha256,
)
from experiments.ve_known_lid_recoverability_campaign import (
    DATASET_SPECS, _array_sha256, make_splits,
)
from models.training import diffusion_denoiser_output, load_checkpoint


SCALES = [.01, .0177827941, .0316227766, .0562341325, .1,
          .177827941, .316227766, .562341325, 1.]


def canonical_posterior(result, points, scale):
    return diffusion_denoiser_output(result.model, points, scale,
        preconditioning=result.config.diffusion_preconditioning,
        data_std=result.config.diffusion_data_std)


def shared_queries(fixture, dataset, test, count):
    ids = np.asarray(fixture["query_ids"], dtype=np.int64)
    if count is not None:
        ids = ids[:count]
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(test)):
        raise ValueError("Invalid shared test IDs")
    raw = np.ascontiguousarray(test[ids])
    if fixture["dataset"] == dataset:
        np.testing.assert_array_equal(raw, np.asarray(fixture["query_raw"])[:len(ids)])
    return ids, raw


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_hash = sha256(args.queries_fixture)
    fixture = torch.load(args.queries_fixture, map_location="cpu", weights_only=False)
    rows, records = [], []
    for dataset in args.datasets:
        source = args.source / dataset
        output = args.output / dataset
        output.mkdir()
        source_manifest = json.loads((source / "dataset_manifest.json").read_text())
        train, val, test = make_splits(dataset,
            train_size=source_manifest["splits"]["train"]["shape"][0],
            validation_size=source_manifest["splits"]["validation"]["shape"][0],
            test_size=source_manifest["splits"]["test"]["shape"][0],
            data_seed=source_manifest["data_seed"])
        for name, array in [("train", train), ("validation", val), ("test", test)]:
            if _array_sha256(array) != source_manifest["splits"][name]["sha256"]:
                raise ValueError(f"Preserved {dataset}/{name} generator hash mismatch")
        checkpoint = source / "model.pt"
        before = sha256(checkpoint)
        result = load_checkpoint(checkpoint, device=args.device)
        result.model.double().eval()
        mean, rms = result.normalization_mean.cpu().float(), result.normalization_scale
        ids, raw = shared_queries(fixture, dataset, test, args.count)
        # Preserve the same float32 normalized observations used by the fresh VP
        # fixture, then evaluate derivatives and squared norms in float64.
        queries = ((torch.from_numpy(raw) - mean) / rms).double()
        fit_ids = np.arange(min(args.fit_count, len(train)))
        normalized_train = (torch.from_numpy(train[fit_ids]) - mean) / rms
        center, basis, singular = fit_affine_span(normalized_train, args.relative_rank_threshold)
        rank, dimension = basis.shape[1], queries.shape[1]
        train_residual = normalized_train.double() - center
        train_residual -= (train_residual @ basis) @ basis.T
        query_residual = queries - center - ((queries - center) @ basis) @ basis.T
        arrays = {"query_ids": ids, "query_raw": raw, "queries": queries.numpy(),
            "scales": np.asarray(args.scales), "raw_sigma": np.asarray(args.scales)*rms,
            "normalization_mean": mean.numpy(), "normalization_rms": np.asarray(rms),
            "span_center": center.numpy(), "span_basis": basis.numpy()}
        accumulated = {}
        points = queries.to(args.device)
        identity = torch.eye(dimension, dtype=points.dtype, device=args.device)
        for scale in args.scales:
            predict = lambda q: canonical_posterior(result, q, scale)
            values, response = projected_trace(predict, points, identity, args.batch_size)
            projected_values, span_response = projected_trace(predict, points,
                basis.to(args.device), args.batch_size)
            torch.testing.assert_close(projected_values, values, rtol=0, atol=0)
            values, response, span_response = values.cpu(), response.cpu(), span_response.cpu()
            terms = correction_components(values, queries, center, basis, scale)
            full = response + terms["correction_original"]
            projected_full = span_response + terms["correction_projected"]
            value_projected = center + ((values-center) @ basis) @ basis.T
            data = {"posterior": values, "posterior_projected": value_projected,
                "response": response, "response_span": span_response,
                "response_offspan": response-span_response,
                "full": full, "full_projected": projected_full, **terms}
            for key, value in data.items():
                accumulated.setdefault(key, []).append(value.numpy())
            d = DATASET_SPECS[dataset].true_lid
            skip = result.config.diffusion_data_std**2 / (scale**2 + result.config.diffusion_data_std**2)
            row = {"dataset": dataset, "lambda": scale, "count": len(queries),
                "true_lid": d, "span_rank": rank,
                "response_mae": float((response-d).abs().mean()),
                "full_mae": float((full-d).abs().mean()),
                "projected_response_mae": float((span_response-d).abs().mean()),
                "projected_full_mae": float((projected_full-d).abs().mean()),
                "response_mean": float(response.mean()),
                "span_response_mean": float(span_response.mean()),
                "offspan_response_mean": float((response-span_response).mean()),
                "analytic_skip_offspan_response": (dimension-rank)*skip if result.config.diffusion_preconditioning == "edm" else 0.,
                "correction_harm_fraction": float(((full-d).abs()>(response-d).abs()).double().mean()),
                **{key+"_mean": float(value.mean()) for key,value in terms.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
        arrays.update({key: np.stack(value) for key,value in accumulated.items()})
        np.savez_compressed(output / "arrays.npz", **arrays)
        torch.save({"queries": queries, "query_raw": raw, "query_ids": ids,
            "mean": mean, "rms": rms, "center": center, "basis": basis,
            "singular_values": singular, "fit_ids": fit_ids}, output / "fixture.pt")
        assert sha256(checkpoint) == before
        records.append({"dataset": dataset, "checkpoint": str(checkpoint), "checkpoint_sha256": before,
            "source_dataset_manifest_sha256": sha256(source / "dataset_manifest.json"),
            "all_regenerated_split_hashes_match": True,
            "normalization_rms": rms, "normalization_source": "immutable VE checkpoint",
            "same_fixture_dataset": dataset == fixture["dataset"],
            "same_fixture_normalization": dataset == fixture["dataset"] and
                torch.equal(mean, fixture["mean"]) and rms == fixture["rms"],
            "query_ids": ids.tolist(), "fit_count": len(fit_ids), "rank": rank,
            "relative_rank_threshold": args.relative_rank_threshold,
            "train_offspan_rms": float(train_residual.square().mean().sqrt()),
            "query_offspan_rms": float(query_residual.square().mean().sqrt()),
            "preconditioning": result.config.diffusion_preconditioning,
            "data_std": result.config.diffusion_data_std, "best_epoch": result.best_epoch})
    assert sha256(args.queries_fixture) == fixture_hash
    (args.output / "manifest.json").write_text(json.dumps({
        "query_fixture": str(args.queries_fixture), "query_fixture_sha256": fixture_hash,
        "source_sha256": sha256(__file__), "span_helper_sha256": sha256(Path(__file__).with_name("response_full_models_20260907_span.py")),
        "training_source_sha256": sha256("models/training.py"),
        "dtype": "float32 canonical observations, float64 model and derivatives",
        "trace": "exact canonical-coordinate batched VJPs including all EDM factors",
        "scales": args.scales, "device": args.device,
        "intervention": "training-data global affine-span output projection, no local-d labels",
        "datasets": records, "all_checkpoints_unchanged": True}, indent=2)+"\n")
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"output": str(args.output), "rows": len(rows), "all_checkpoints_unchanged": True}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queries-fixture", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["e1_spiral_pca", "e6_exp_pca", "e7_crescent_moon_radius3.0"])
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--fit-count", type=int, default=4096)
    p.add_argument("--relative-rank-threshold", type=float, default=1e-5)
    p.add_argument("--scales", nargs="+", type=float, default=SCALES)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())), flush=True)
