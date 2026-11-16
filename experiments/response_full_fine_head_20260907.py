"""One fixed fine-band SVD decoder fit on previously frozen features/targets.

This deliberately replaces the shared all-scale objective by its lambda<=.03
restriction. It is a cause diagnostic, not a scale-conditioned/deployable head
selection rule. No ridge, alternate cutoff, nonlinear training or second fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_convex_head import (
    body_hash, capture_features, replaced_head, residual_targets, solve_head,
)
from experiments.response_full_models_20260907_convex_head_loader import load_convex_head
from experiments.response_full_models_20260907_oracle_audit import decompose
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import (
    active_jacobian, split_jacobian_error, tangent_from_phase,
)
from experiments.response_full_models_20260907_ve import SCALES
from experiments.response_full_models_20260907_ve_refinement import diagnostics, state_cpu
from experiments.response_full_models_20260907_ve_refinement_audit import value_components
from models.training import diffusion_denoiser_output

FINE_MAX = .03


def fine_mask(sigma):
    sigma = torch.as_tensor(sigma, dtype=torch.float64, device="cpu")
    if sigma.ndim != 1 or not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError("Expected finite positive vector of physical noise scales")
    mask = sigma <= FINE_MAX
    if not mask.any() or mask.all():
        raise ValueError("Need both fine and complementary rows")
    return mask


def fit_fine_head(design, targets, sigma):
    if design.dtype != targets.dtype or design.dtype != torch.float64:
        raise ValueError("Require saved float64 design and targets")
    if design.shape[0] != targets.shape[0] or len(sigma) != len(design):
        raise ValueError("Row counts differ")
    if not torch.equal(design[:, -1], torch.ones(len(design), dtype=design.dtype)):
        raise ValueError("Saved last design column is not the exact intercept")
    mask = fine_mask(sigma)
    coefficient, subdesign, spectrum, residual, receipt = solve_head(
        design[mask, :-1], targets[mask])
    torch.testing.assert_close(subdesign, design[mask], rtol=0, atol=0)
    receipt.update({"fine_max": FINE_MAX, "fine_rows": int(mask.sum()),
                    "rest_rows": int((~mask).sum()), "all_rows": len(mask)})
    return coefficient, mask, spectrum, residual, receipt


def objective_partitions(design, targets, coefficient, mask):
    residual = design @ coefficient - targets
    masks = {"fine": mask, "rest": ~mask, "all": torch.ones_like(mask)}
    return {key: {"count": int(selected.sum()),
                  "mse": float(residual[selected].square().mean())}
            for key, selected in masks.items()}


def write_rows(path, rows):
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    sources = [Path(__file__), args.parent / "models.pt", args.parent / "least_squares.npz",
               args.parent / "manifest.json", args.cache, args.queries, args.oracle, args.noisy_oracle]
    for suffix in ("convex_head", "convex_head_loader", "oracle_audit", "rb_training", "span",
                   "spaghetti_local", "ve", "ve_refinement", "ve_refinement_audit"):
        sources.append(Path(f"experiments/response_full_models_20260907_{suffix}.py"))
    sources.extend([Path("models/neural_fields.py"), Path("models/training.py"),
                    Path("experiments/fm_mechanism_fast_trace.py")])
    for name in ("original", "refit"):
        sources.extend([args.parent / f"{name}_arrays.npz", args.parent / f"{name}_noisy.npz"])
    hashes = {str(path): sha256(path) for path in sources}
    parent_manifest = json.loads((args.parent / "manifest.json").read_text())
    if parent_manifest["sources"][str(args.cache)] != hashes[str(args.cache)]:
        raise ValueError("Target cache differs from original convex fit")
    saved = np.load(args.parent / "least_squares.npz")
    ids = saved["objective_row_ids"]
    np.testing.assert_array_equal(ids, np.arange(516096, 524288))
    design = torch.from_numpy(saved["design"])
    targets = torch.from_numpy(saved["targets"])
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    tape = {key: cache[key][ids].double().to(args.device)
            for key in ("observed", "sigma", "oracle_target")}
    sigma = tape["sigma"].cpu()
    payload = torch.load(args.parent / "models.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(torch.as_tensor(cache["normalization_mean"]),
                               payload["normalization"]["mean"], rtol=0, atol=0)
    assert float(cache["normalization_scale"]) == float(payload["normalization"]["scale"])
    original = load_convex_head(args.parent / "models.pt", "original", args.device).requires_grad_(False)
    global_head = load_convex_head(args.parent / "models.pt", "refit", args.device).requires_grad_(False)
    original_body = body_hash(original)
    assert original_body == body_hash(global_head) == parent_manifest["source_body_sha256"]
    recaptured = capture_features(original, tape["observed"], tape["sigma"]).cpu()
    torch.testing.assert_close(recaptured, design[:, :-1], rtol=1e-12, atol=1e-12)
    target_replay = residual_targets(tape["observed"], tape["sigma"], tape["oracle_target"]).cpu()
    torch.testing.assert_close(target_replay, targets, rtol=0, atol=0)
    coefficient, mask, spectrum, residual, receipt = fit_fine_head(design, targets, sigma)
    if int(mask.sum()) != 1205:
        raise ValueError("Pre-existing fine-band mask no longer contains 1205 rows")
    models = {"original": original, "global_ols": global_head,
              "fine_ols": replaced_head(original, coefficient)}
    coefficients = {"original": torch.from_numpy(saved["old_coefficient"]),
                    "global_ols": torch.from_numpy(saved["coefficient"]), "fine_ols": coefficient}
    objective_rows = []
    for name, model in models.items():
        actual_coef = torch.cat([model.output[-1].weight.T, model.output[-1].bias[None]]).cpu()
        torch.testing.assert_close(actual_coef, coefficients[name], rtol=0, atol=0)
        assert body_hash(model) == original_body
        partitions = objective_partitions(design, targets, coefficients[name], mask)
        for part, record in partitions.items():
            selected = {"fine": mask, "rest": ~mask, "all": torch.ones_like(mask)}[part].to(args.device)
            with torch.no_grad():
                actual = float(cached_edm_loss(model, tape["observed"][selected],
                                               tape["sigma"][selected], tape["oracle_target"][selected]))
            np.testing.assert_allclose(actual, record["mse"], rtol=1e-9, atol=1e-12)
            objective_rows.append({"state": name, "partition": part, **record,
                                   "coefficient_norm": float(coefficients[name].norm())})
    receipt["coefficient_norms"] = {name: float(c.norm()) for name, c in coefficients.items()}
    np.savez_compressed(args.output / "least_squares.npz", coefficient=coefficient.numpy(),
                        fine_mask=mask.numpy(), sigma=sigma.numpy(), objective_row_ids=ids,
                        singular_values=spectrum.numpy(), fine_residual=residual.numpy())
    torch.save({**payload, "models": {name: state_cpu(model) for name, model in models.items()},
                "method": "One lambda<=.03-only SVD final head, shared frozen body"}, args.output / "models.pt")
    # Same supported loader, which converts to double BEFORE loading the SVD head.
    reloaded = load_convex_head(args.output / "models.pt", "fine_ols", args.device)
    for key, value in models["fine_ols"].state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])
    del reloaded, cache, recaptured, target_replay
    fixture = np.load(args.queries)
    oracle = np.load(args.oracle)
    for key in ("queries", "query_ids"):
        np.testing.assert_array_equal(fixture[key], oracle[key])
    np.testing.assert_allclose(SCALES, oracle["scales"], rtol=1e-9, atol=0)
    q = torch.from_numpy(fixture["queries"]).double().to(args.device)
    tangent = tangent_from_phase(oracle["query_phase"])
    noisy = np.load(args.noisy_oracle)
    noisy_q = torch.from_numpy(noisy["queries"]).double().to(args.device)
    noisy_z = torch.from_numpy(noisy["noise"]).double().to(args.device)
    rows, risk_rows = [], []
    for name, model in models.items():
        data = diagnostics(model, q, SCALES)
        if name != "fine_ols":
            old_name = "original" if name == "original" else "refit"
            old = np.load(args.parent / f"{old_name}_arrays.npz")
            for key in ("posterior", "response", "correction_original", "full"):
                np.testing.assert_allclose(data[key], old[key], rtol=1e-10, atol=1e-10)
        result = decompose(fixture["queries"], data["posterior"], oracle["posterior"],
                           data["response"], oracle["response"], SCALES)
        parts = value_components(result["posterior_error"], tangent, SCALES)
        jacobian_arrays = {}
        for i, scale in enumerate(SCALES):
            value, jac = active_jacobian(lambda x: diffusion_denoiser_output(
                model, x, float(scale), preconditioning="edm", data_std=1.), q, active=30)
            value, jac = value.cpu().numpy(), jac.cpu().numpy()
            np.testing.assert_allclose(value, data["posterior"][i], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(np.trace(jac, axis1=-2, axis2=-1), data["response"][i], rtol=1e-9, atol=1e-9)
            active = jac[:, :20, :20]
            truth = oracle["jacobian_active"][i]
            full_truth = np.zeros_like(jac)
            full_truth[:, :20, :20] = truth
            tangent_r = np.einsum("bi,bij,bj->b", tangent, active, tangent)
            extra = {"jacobian": jac, "response_tangent": tangent_r,
                     "response_curve_normal": np.trace(active, axis1=-2, axis2=-1) - tangent_r,
                     "response_ambient_normal": np.trace(jac[:, 20:, 20:], axis1=-2, axis2=-1),
                     "whole_jacobian_error_squared": ((jac-full_truth)**2).sum((-1, -2)),
                     **split_jacobian_error(active-truth, tangent)}
            for key, array in extra.items():
                jacobian_arrays.setdefault(key, []).append(array)
            rows.append({"state": name, "lambda": float(scale), "count": len(q),
                         "response_mean": float(data["response"][i].mean()),
                         "correction_mean": float(data["correction_original"][i].mean()),
                         "full_d_mae": float(abs(data["full"][i]-1).mean()),
                         "response_oracle_mae": float(abs(result["response_error"][i]).mean()),
                         "full_oracle_mae": float(abs(result["full"][i]-result["oracle_full"][i]).mean()),
                         "posterior_error_energy": float(result["posterior_error_energy"][i].mean()),
                         **{key+"_mean": float(array.mean()) for key, array in extra.items() if array.ndim == 1},
                         **{"posterior_error_"+key+"_energy": float(array[i].mean()) for key, array in parts.items()}})
        np.savez_compressed(args.output / f"{name}_arrays.npz", queries=fixture["queries"],
                            query_ids=fixture["query_ids"], scales=SCALES, **data,
                            **{key: np.stack(array) for key, array in jacobian_arrays.items()},
                            **{"posterior_error_"+key+"_energy": array for key, array in parts.items()},
                            posterior_error_energy=result["posterior_error_energy"], oracle_cross=result["oracle_cross"])
        predictions = []
        for i, scale in enumerate(noisy["scales"]):
            with torch.no_grad():
                b = diffusion_denoiser_output(model, noisy_q+float(scale)*noisy_z,
                    float(scale), preconditioning="edm", data_std=1.)
            reference = torch.from_numpy(noisy[f"oracle_{i}"]).to(args.device)
            predictions.append(b.cpu().numpy())
            risk_rows.append({"state": name, "lambda": float(scale), "count": len(noisy_q),
                              "noisy_clean_mse_over_Dlambda2": float((b-noisy_q).square().mean()/scale**2),
                              "noisy_oracle_error_over_Dlambda2": float((b-reference).square().mean()/scale**2)})
        if name != "fine_ols":
            previous = np.load(args.parent / f"{old_name}_noisy.npz")
            np.testing.assert_allclose(np.stack(predictions), previous["posterior"], rtol=1e-12, atol=1e-12)
        np.savez_compressed(args.output / f"{name}_noisy.npz", posterior=np.stack(predictions),
                            queries=noisy["queries"], noise=noisy["noise"], scales=noisy["scales"])
    write_rows(args.output / "objectives", objective_rows)
    write_rows(args.output / "metrics", rows)
    write_rows(args.output / "noisy_risk", risk_rows)
    (args.output / "solve.json").write_text(json.dumps(receipt, indent=2)+"\n")
    assert all(sha256(path) == digest for path, digest in hashes.items())
    manifest = {"sources": hashes, "completed": True, "all_sources_unchanged": True,
                "body_sha256_by_state": {name: body_hash(model) for name, model in models.items()},
                "source_body_sha256": original_body, "fit_rows": int(mask.sum()),
                "selection": "Only fixed physical lambda<=.03, inherited1205 mask; no ridge, tuning, alternate fit or LID selection",
                "objective_scope": "Restriction/reweighting of the frozen all-scale DATA objective; same posterior targets and nonlinear features, not same global objective",
                "heldout_scope": "Independent continuous-oracle noisy bank has256 observations per3 scales; clean256/all9 actual full spatial Jacobians",
                "precision": "Float64 solve and inference; supported double-before-load loader roundtrip bitwise",
                "oracle_scale_relative_mismatch": float(np.max(abs(SCALES-oracle["scales"])/oracle["scales"])),
                "elapsed_seconds": time.monotonic()-start}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"solve": receipt, "objectives": objective_rows, "noisy_risk": risk_rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("artifacts/response_full_20260907")
    parser.add_argument("--parent", type=Path, default=base / "convex_head_spaghetti")
    parser.add_argument("--cache", type=Path, default=base / "rb_spaghetti_cache/tape.pt")
    parser.add_argument("--queries", type=Path, default=base / "ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz")
    parser.add_argument("--oracle", type=Path, default=base / "population_oracle_ve_spaghetti256/arrays.npz")
    parser.add_argument("--noisy-oracle", type=Path, default=Path("../notes/response_full_20260907/spaghetti_noisy_risk/arrays.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())
