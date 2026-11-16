"""Auditable decomposition of learned correction against a population oracle.

For e=b-b*, the exact learned correction is C* + 2(b*-x).e/lambda²
+ ||e||²/lambda². Oracle swaps below are diagnostic counterfactual readouts,
not implementable estimators or a new training intervention.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.response_full_models_20260907_span import sha256


def decompose(queries, learned_mean, oracle_mean, learned_response, oracle_response, scales):
    q, b, truth = (np.asarray(value, dtype=np.float64) for value in (queries, learned_mean, oracle_mean))
    if b.shape != truth.shape or b.shape[1:] != q.shape:
        raise ValueError("Posterior/query shapes disagree")
    scale2 = np.asarray(scales, dtype=np.float64)[:, None]**2
    displacement = truth-q[None]
    error = b-truth
    correction = ((b-q[None])**2).sum(-1)/scale2
    oracle_c = (displacement**2).sum(-1)/scale2
    cross = 2*(displacement*error).sum(-1)/scale2
    energy = (error**2).sum(-1)/scale2
    np.testing.assert_allclose(correction, oracle_c+cross+energy, rtol=1e-11, atol=1e-11)
    dr = learned_response-oracle_response
    full = learned_response+correction
    oracle_full = oracle_response+oracle_c
    np.testing.assert_allclose(full-oracle_full, dr+cross+energy, rtol=1e-10, atol=1e-10)
    return {"correction": correction, "oracle_correction": oracle_c,
        "oracle_cross": cross, "posterior_error_energy": energy,
        "response_error": dr, "full": full, "oracle_full": oracle_full,
        "oracle_displacement": displacement, "posterior_error": error,
        "oracle_response_learned_correction": oracle_response+correction,
        "learned_response_oracle_correction": learned_response+oracle_c}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model = np.load(args.model_arrays)
    oracle = np.load(args.oracle_arrays)
    scales = oracle["scales"]
    scale_relative_mismatch = 0.
    if "scales" in model:
        # Some campaigns serialize nine-decimal logspace values. Permit only
        # sub-part-per-billion serialization differences, and disclose them.
        np.testing.assert_allclose(model["scales"], scales, rtol=1e-9, atol=0)
        scale_relative_mismatch = float(np.max(np.abs(model["scales"]-scales)/scales))
        scales = model["scales"]
    if "query_ids" in model:
        np.testing.assert_array_equal(model["query_ids"], oracle["query_ids"])
    queries = oracle["queries"]
    if "queries" in model:
        np.testing.assert_array_equal(model["queries"], queries)
    results = decompose(queries, model[args.posterior_key], oracle["posterior"],
        model[args.response_key], oracle["response"], scales)
    np.testing.assert_allclose(results["correction"], model[args.correction_key], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(results["oracle_correction"], oracle["correction"], rtol=2e-7, atol=2e-7)
    if args.basis_arrays:
        import torch
        fixture = torch.load(args.basis_arrays, map_location="cpu", weights_only=False)
        basis = fixture["basis"].numpy()
        error = results["posterior_error"]
        span_error = (error@basis)@basis.T
        off_error = error-span_error
        results["span_posterior_error_energy"] = (span_error**2).sum(-1)/scales[:,None]**2
        results["offspan_posterior_error_energy"] = (off_error**2).sum(-1)/scales[:,None]**2
    rows = []
    for i, scale in enumerate(scales):
        d = args.true_lid
        r, rstar = model[args.response_key][i], oracle["response"][i]
        c, cstar, full, fstar = (results[key][i] for key in ["correction", "oracle_correction", "full", "oracle_full"])
        row = {"name": args.name, "lambda": float(scale), "count": len(queries),
            "learned_response_mean": float(r.mean()), "oracle_response_mean": float(rstar.mean()),
            "learned_response_d_mae": float(np.abs(r-d).mean()),
            "learned_full_d_mae": float(np.abs(full-d).mean()),
            "oracle_response_d_mae": float(np.abs(rstar-d).mean()),
            "oracle_full_d_mae": float(np.abs(fstar-d).mean()),
            "response_oracle_mae": float(np.abs(r-rstar).mean()),
            "full_oracle_mae": float(np.abs(full-fstar).mean()),
            "learned_correction_harm_fraction": float((np.abs(full-d)>np.abs(r-d)).mean()),
            "oracle_correction_harm_fraction": float((np.abs(fstar-d)>np.abs(rstar-d)).mean()),
            "oracle_response_learned_correction_d_mae": float(np.abs(results["oracle_response_learned_correction"][i]-d).mean()),
            "learned_response_oracle_correction_d_mae": float(np.abs(results["learned_response_oracle_correction"][i]-d).mean())}
        for key, values in results.items():
            if values.ndim == 2:
                row[key+"_mean"] = float(values[i].mean())
        rows.append(row)
        print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / "decomposition.npz", scales=scales,
        query_ids=oracle["query_ids"], **results)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "manifest.json").write_text(json.dumps({
        "name": args.name, "model_arrays": str(args.model_arrays), "model_arrays_sha256": sha256(args.model_arrays),
        "oracle_arrays": str(args.oracle_arrays), "oracle_arrays_sha256": sha256(args.oracle_arrays),
        "source_sha256": sha256(__file__), "true_lid": args.true_lid,
        "model_scales": scales.tolist(), "oracle_scales": oracle["scales"].tolist(),
        "maximum_relative_scale_serialization_difference": scale_relative_mismatch,
        "posterior_key": args.posterior_key, "response_key": args.response_key,
        "correction_key": args.correction_key,
        "swaps": "Diagnostic oracle swaps, not implementable estimators or training interventions"}, indent=2)+"\n")
    return {"rows": len(rows), "output": str(args.output)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-arrays", type=Path, required=True)
    p.add_argument("--oracle-arrays", type=Path, required=True)
    p.add_argument("--basis-arrays", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--true-lid", type=float, required=True)
    p.add_argument("--posterior-key", default="posterior")
    p.add_argument("--response-key", default="response")
    p.add_argument("--correction-key", default="correction_original")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args())), flush=True)
