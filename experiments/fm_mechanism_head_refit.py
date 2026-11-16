"""Frozen-feature convex head refits for the Gaussian posterior-FM control.

No hidden parameters are trained. Geometry-informed tangent targets isolate the
final affine head from the representation learned by the original network.
All replacement checkpoints are diagnostic artifacts in a new directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from experiments.fm_mechanism_diagnostics import alpha_at, value_and_jacobian
from experiments.fm_mechanism_pilot import (
    canonical_posterior, gaussian_posterior, load_pilot_checkpoint,
)


def sample_data(fixture: dict[str, Any], count: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    basis = fixture["tangent_basis"].double()
    clean = float(fixture["latent_std"]) * torch.randn(count, basis.shape[1], generator=generator, dtype=torch.float64) @ basis.T
    noise = torch.randn(count, basis.shape[0], generator=generator, dtype=torch.float64)
    lam = (math.log(.01) + math.log(100) * torch.rand(count, generator=generator, dtype=torch.float64)).exp()
    return clean, noise, lam


@torch.no_grad()
def final_features(field: Any, points: Tensor, lam: Tensor, schedule: str, chunk: int = 512) -> Tensor:
    """Capture the actual input to the final Linear, including SiLU/LayerNorm."""
    captured: list[Tensor] = []
    handle = field.output[-1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        for start in range(0, len(points), chunk):
            canonical_posterior(field, points[start:start+chunk], lam[start:start+chunk], schedule)
    finally:
        handle.remove()
    return torch.cat(captured)


def solve_weighted_head(features: Tensor, targets: Tensor, lam: Tensor, rcond: float = 1e-10) -> tuple[Tensor, dict[str, Any]]:
    design = torch.cat([features, torch.ones(len(features), 1, dtype=features.dtype)], dim=1)
    weighted = design / lam[:, None]
    target = targets / lam[:, None]
    result = torch.linalg.lstsq(weighted, target, rcond=rcond, driver="gelsd")
    coefficient = result.solution
    residual = weighted @ coefficient - target
    gradient = weighted.T @ residual
    normalized_stationarity = gradient.norm() / (weighted.norm() * residual.norm()).clamp_min(1e-30)
    audit = {
        "driver": "gelsd", "rcond": rcond, "rank": int(result.rank),
        "rows": len(weighted), "columns": weighted.shape[1],
        "singular_values": result.singular_values.tolist(),
        "coefficient_norm": float(coefficient.norm()),
        "weighted_residual_mse_per_target": residual.square().mean(0).tolist(),
        "normalized_stationarity": float(normalized_stationarity),
    }
    return coefficient, audit


def replaced_head(field: Any, coefficient: Tensor, basis: Tensor) -> Any:
    """Round fitted head to training dtype, then use analysis float64 uniformly."""
    result = copy.deepcopy(field)
    head = result.output[-1]
    with torch.no_grad():
        head.weight.copy_((basis @ coefficient[:-1].T).float().double())
        head.bias.copy_((basis @ coefficient[-1]).float().double())
    return result


@torch.no_grad()
def predict(field: Any, points: Tensor, lam: Tensor | float, schedule: str, chunk: int = 512) -> Tensor:
    lam = torch.as_tensor(lam, dtype=points.dtype)
    if lam.ndim == 0:
        lam = lam.expand(len(points))
    return torch.cat([canonical_posterior(field, part, lam[start:start+len(part)], schedule)
                      for start, part in zip(range(0, len(points), chunk), points.split(chunk))])


def evaluate(
    field: Any, fixture: dict[str, Any], schedule: str, variant: str,
    heldout: tuple[Tensor, Tensor, Tensor], query_count: int, scales: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Tensor]]:
    basis = fixture["tangent_basis"].double()
    # Double-precision orthogonalization stays within the saved tangent span.
    basis, _ = torch.linalg.qr(basis)
    clean, noise, lam_mix = heldout
    noisy_mix = clean + lam_mix[:, None] * noise
    oracle_mix = gaussian_posterior(noisy_mix, lam_mix, fixture)
    predicted = predict(field, noisy_mix, lam_mix, schedule)
    tangent_prediction = (predicted @ basis) @ basis.T
    risk = {
        "schedule": schedule, "variant": variant,
        "heldout_weighted_target_risk": float(((tangent_prediction-clean).square().mean(-1)/lam_mix.square()).mean()),
        "heldout_weighted_oracle_excess": float(((tangent_prediction-oracle_mix).square().mean(-1)/lam_mix.square()).mean()),
        "heldout_oracle_target_risk": float(((oracle_mix-clean).square().mean(-1)/lam_mix.square()).mean()),
    }
    rows, arrays = [], {}
    # Fresh heldout queries, never the refit points; cap value diagnostics at 2048.
    value_clean, value_noise = clean[:2048], noise[:2048]
    for scale in scales:
        for where, points in (("clean", value_clean), ("noisy", value_clean + scale * value_noise)):
            value = predict(field, points, scale, schedule) @ basis
            oracle = gaussian_posterior(points, scale, fixture) @ basis
            row = {"schedule": schedule, "variant": variant, "lambda": scale, "location": where,
                   "posterior_tangent_error_energy_over_lambda2": float((value-oracle).square().sum(-1).mean()/scale**2)}
            rows.append(row)
            arrays[f"{variant}/{schedule}/{scale}/{where}/value_tangent"] = value
        tangent_queries = value_clean[:query_count] @ basis
        def tangent_fn(t: Tensor) -> Tensor:
            return canonical_posterior(field, t @ basis.T, scale, schedule) @ basis
        value, jac = value_and_jacobian(tangent_fn, tangent_queries, chunk=query_count)
        origin = tangent_fn(torch.zeros(1, basis.shape[1], dtype=torch.float64)).detach()
        gain = float(fixture["latent_std"])**2 / (float(fixture["latent_std"])**2 + scale**2)
        oracle_value = gain * tangent_queries
        response = jac.diagonal(dim1=-2, dim2=-1).sum(-1)
        correction = (value-tangent_queries).square().sum(-1)/scale**2
        full = response + correction
        oracle_full = basis.shape[1]*gain + (oracle_value-tangent_queries).square().sum(-1)/scale**2
        euler = torch.einsum("bij,bj->bi", jac, tangent_queries) - (value-origin)
        radius_sq = tangent_queries.square().sum(-1)
        radial_gain = ((value-origin)*tangent_queries).sum(-1)/radius_sq
        radial_derivative = torch.einsum("bi,bij,bj->b", tangent_queries, jac, tangent_queries)/radius_sq
        diagnostic = {
            "schedule": schedule, "variant": variant, "lambda": scale, "location": "clean_exact_tangent_jacobian",
            "query_count": query_count,
            "tangent_response_mean": float(response.mean()),
            "tangent_full_mean": float(full.mean()),
            "tangent_full_mae_lid": float((full-basis.shape[1]).abs().mean()),
            "tangent_full_oracle_mae": float((full-oracle_full).abs().mean()),
            "tangent_response_oracle_mae": float((response-basis.shape[1]*gain).abs().mean()),
            "relative_centered_euler_defect_mean": float((euler.norm(dim=-1)/radius_sq.sqrt()).mean()),
            "radial_gain_mean": float(radial_gain.mean()),
            "radial_derivative_mean": float(radial_derivative.mean()),
            "oracle_gain": gain,
        }
        rows.append(diagnostic)
        arrays[f"{variant}/{schedule}/{scale}/clean/tangent_jacobian"] = jac
    return risk, rows, arrays


def run(checkpoint: Path, output: Path, *, sample_count: int = 32768, heldout_count: int = 8192,
        query_count: int = 24, seed: int = 20260905, include_local: bool = True) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fixture = torch.load(checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    if fixture.get("data_law", "gaussian") != "gaussian":
        raise ValueError("This first refit diagnostic is restricted to the Gaussian control")
    output_coordinate = payload["config"].get("output_coordinate", "data")
    basis = fixture["tangent_basis"].double()
    start = time.monotonic()
    clean, noise, lam = sample_data(fixture, sample_count, seed + 701)
    heldout = sample_data(fixture, heldout_count, seed + 702)
    noisy = clean + lam[:, None] * noise
    oracle_target = gaussian_posterior(noisy, lam, fixture) @ basis
    finite_target = clean @ basis
    targets = torch.cat([oracle_target, finite_target], dim=1)
    saved = {"shared_oracle": {}, "shared_finite_target": {}, "local_lambda1_oracle": {}}
    coefficient_states, fit_audits, risk_rows, metric_rows, all_arrays = {}, [], [], [], {}
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(checkpoint, schedule, "cpu")
        field = field.double().eval()
        features = final_features(field, noisy, lam, schedule)
        output_alpha = alpha_at(lam, schedule) if output_coordinate == "native" else torch.ones_like(lam)
        coefficient, audit = solve_weighted_head(features, targets * output_alpha[:, None], lam * output_alpha)
        fit_audits.append({"schedule": schedule, "fit": "shared", **audit})
        d = basis.shape[1]
        variants = {"original_tangent_projected": field,
                    "shared_oracle": replaced_head(field, coefficient[:, :d], basis),
                    "shared_finite_target": replaced_head(field, coefficient[:, d:], basis)}
        coefficient_states[schedule] = {"shared_oracle": coefficient[:, :d], "shared_finite_target": coefficient[:, d:]}
        if include_local:
            local_lam = torch.ones_like(lam)
            local_noisy = clean + noise
            local_features = final_features(field, local_noisy, local_lam, schedule)
            local_target = gaussian_posterior(local_noisy, local_lam, fixture) @ basis
            local_alpha = alpha_at(local_lam, schedule) if output_coordinate == "native" else torch.ones_like(local_lam)
            local_coefficient, local_audit = solve_weighted_head(local_features, local_target * local_alpha[:, None], local_lam * local_alpha)
            fit_audits.append({"schedule": schedule, "fit": "lambda1", **local_audit})
            variants["local_lambda1_oracle"] = replaced_head(field, local_coefficient, basis)
            coefficient_states[schedule]["local_lambda1_oracle"] = local_coefficient
        for variant, model in variants.items():
            scales = (1.0,) if variant.startswith("local") else (.1, .316227766, 1.0)
            risk, metrics, arrays = evaluate(model, fixture, schedule, variant, heldout, query_count, scales)
            if variant.startswith("local"):
                risk["mixed_scale_risk_scope"] = "out_of_fit_distribution_diagnostic_only"
            risk_rows.append(risk)
            metric_rows.extend(metrics)
            all_arrays.update(arrays)
            if variant in saved:
                saved[variant][schedule] = {name: value.detach().float().cpu() for name, value in model.state_dict().items()}
            print(json.dumps({"schedule": schedule, "variant": variant, "risk": risk,
                              "lambda1": [row for row in metrics if row["lambda"] == 1 and row["location"] == "clean_exact_tangent_jacobian"]}), flush=True)
    torch.save(fixture, output / "data.pt")
    torch.save(coefficient_states, output / "head_coefficients.pt")
    torch.save(all_arrays, output / "evaluation_arrays.pt")
    for variant, models in saved.items():
        if not models:
            continue
        refit_payload = {"schema_version": payload["schema_version"], "step": payload["step"],
                         "config": payload["config"], "architecture": payload["architecture"],
                         "models": models, "optimizers": {}, "fixture_file": "data.pt",
                         "head_refit": {"variant": variant, "samples": sample_count, "source": str(checkpoint)}}
        torch.save(refit_payload, output / f"checkpoint_{variant}.pt")
    summary = {"source_checkpoint": str(checkpoint),
               "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "sample_count": sample_count, "heldout_count": heldout_count, "query_count": query_count,
               "seed": seed, "elapsed_seconds": time.monotonic()-start,
               "output_coordinate": output_coordinate,
               "fit_audits": fit_audits, "heldout_risks": risk_rows, "metrics": metric_rows,
               "claim_boundary": "Frozen-body geometry-informed head diagnostic; replacement is not a deployable label-free method. All reported original comparisons are tangent-projected; no normal-repair credit.",
               "optimization": "CPU float64 SVD gelsd, rcond1e-10, no ridge; head rounded to float32 before float64 diagnostics",
               "target": "Rao-Blackwellized population posterior mean versus independent finite clean-X targets on identical feature rows",
               "native_output_risk_contract": "For native output only: head target alpha*b*, design/targets weighted by 1/(alpha*lambda); evaluation divides head output by alpha exactly once"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in metric_rows for key in row))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(metric_rows)
    print(json.dumps({"output": str(output), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32768)
    parser.add_argument("--heldout-count", type=int, default=8192)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.output, sample_count=args.sample_count, heldout_count=args.heldout_count,
        query_count=args.query_count, seed=args.seed, include_local=not args.skip_local)


if __name__ == "__main__":
    main()
