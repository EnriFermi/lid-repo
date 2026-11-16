"""Global-span readout audit of immutable paired complex VP checkpoints.

Project the canonical posterior, including the noise model's analytic identity
skip, never its raw output. Cached exact full traces are independently replayed
with generic VJPs at every scale on a fixed first-query subset.
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
from experiments.response_full_training_20260907 import posterior
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def load_field(payload, kind, mode, device):
    field = ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    field.load_state_dict(payload["models" if mode == "last" else "averages"][kind], strict=True)
    return field.to(device).double().eval()


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.source / "final.pt"
    before = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fixture = torch.load(args.source / "fixture.pt", map_location="cpu", weights_only=False)
    cached = np.load(args.source / "predictions.npz")
    config, dataset = payload["config"], payload["config"]["dataset"]
    train, _, test = make_splits(dataset, train_size=config["train_count"],
        validation_size=config["selection_count"], test_size=config["test_count"], data_seed=0)
    raw = np.asarray(fixture["query_raw"])
    ids = np.asarray(fixture["query_ids"])
    np.testing.assert_array_equal(raw, test[ids])
    mean, rms = fixture["mean"], fixture["rms"]
    queries = fixture["queries"].double()
    normalized_train = (torch.from_numpy(train[:args.fit_count]) - mean) / rms
    center, basis, singular = fit_affine_span(normalized_train, args.relative_rank_threshold)
    np.testing.assert_array_equal(queries.numpy(), cached["queries"])
    np.testing.assert_array_equal(ids, cached["query_ids"])
    rank, dimension = basis.shape[1], queries.shape[1]
    torch.save({"center": center, "basis": basis, "singular_values": singular,
        "queries": queries, "query_raw": raw, "query_ids": ids, "mean": mean, "rms": rms},
        args.output / "span_and_queries.pt")
    rows, replay = [], []
    for kind in args.kinds:
        for mode in args.modes:
            model = load_field(payload, kind, mode, args.device)
            arrays = {"query_ids": ids, "query_raw": raw, "queries": queries.numpy(),
                "scales": cached["scales"], "normalization_mean": mean.numpy(),
                "normalization_rms": np.asarray(rms), "span_center": center.numpy(), "span_basis": basis.numpy()}
            accumulated = {}
            for i, scale in enumerate(cached["scales"]):
                predict = lambda q: posterior(model, q, float(scale), kind)
                values, span_r = projected_trace(predict, queries.to(args.device), basis.to(args.device), args.batch_size)
                count = min(args.trace_replay_count, len(queries))
                _, replay_r = projected_trace(predict, queries[:count].to(args.device),
                    torch.eye(dimension, device=args.device, dtype=torch.float64), args.batch_size)
                values, span_r = values.cpu(), span_r.cpu()
                response = torch.from_numpy(cached[f"{kind}_{mode}_response"][i]).double()
                np.testing.assert_allclose(values.numpy(), cached[f"{kind}_{mode}_b"][i], rtol=1e-9, atol=1e-9)
                torch.testing.assert_close(replay_r.cpu(), response[:count], rtol=1e-9, atol=1e-9)
                terms = correction_components(values, queries, center, basis, float(scale))
                np.testing.assert_allclose(terms["correction_original"].numpy(), cached[f"{kind}_{mode}_correction"][i], rtol=1e-9, atol=1e-9)
                full = response+terms["correction_original"]
                projected_full = span_r+terms["correction_projected"]
                projected_b = center+((values-center)@basis)@basis.T
                data = {"posterior": values, "posterior_projected": projected_b,
                    "response": response, "response_span": span_r, "response_offspan": response-span_r,
                    "full": full, "full_projected": projected_full, **terms}
                for name, value in data.items():
                    accumulated.setdefault(name, []).append(value.numpy())
                d = DATASET_SPECS[dataset].true_lid
                rows.append({"dataset": dataset, "kind": kind, "mode": mode,
                    "lambda": float(scale), "count": len(queries), "span_rank": rank,
                    "response_mae": float((response-d).abs().mean()),
                    "full_mae": float((full-d).abs().mean()),
                    "projected_response_mae": float((span_r-d).abs().mean()),
                    "projected_full_mae": float((projected_full-d).abs().mean()),
                    "response_mean": float(response.mean()), "span_response_mean": float(span_r.mean()),
                    "offspan_response_mean": float((response-span_r).mean()),
                    **{name+"_mean": float(value.mean()) for name,value in terms.items()}})
                replay.append({"kind": kind, "mode": mode, "lambda": float(scale),
                    "trace_queries": count, "max_full_trace_replay_error": float((replay_r.cpu()-response[:count]).abs().max())})
                print(json.dumps(rows[-1]), flush=True)
            arrays.update({key: np.stack(value) for key,value in accumulated.items()})
            np.savez_compressed(args.output / f"{kind}_{mode}_arrays.npz", **arrays)
    assert sha256(checkpoint) == before
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "manifest.json").write_text(json.dumps({
        "source": str(args.source), "checkpoint_sha256": before, "source_sha256": sha256(__file__),
        "fixture_sha256": sha256(args.source / "fixture.pt"),
        "cached_predictions_sha256": sha256(args.source / "predictions.npz"),
        "training_adapter_sha256": sha256("experiments/response_full_training_20260907.py"),
        "dtype": "float64 model/derivatives, original float32 canonical observations",
        "fit_count": len(normalized_train), "fit_subset": "first training observations",
        "relative_rank_threshold": args.relative_rank_threshold, "span_rank": rank,
        "all_checkpoints_unchanged": True, "trace_replays": replay,
        "intervention": "Affine-span projection of canonical posterior, including noise identity skip"}, indent=2)+"\n")
    return {"output": str(args.output), "rows": len(rows), "checkpoint_unchanged": True}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--kinds", nargs="+", default=["posterior", "noise"])
    p.add_argument("--modes", nargs="+", default=["last", "average"])
    p.add_argument("--fit-count", type=int, default=4096)
    p.add_argument("--relative-rank-threshold", type=float, default=1e-5)
    p.add_argument("--trace-replay-count", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())), flush=True)
