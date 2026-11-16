"""CPU frozen-feature probe of final LayerNorm amplitude information.

Every variant uses the same frozen trunk, fit/heldout draws and weighted
Gaussian-posterior objective. Only the representation supplied to a fitted
affine head changes. This is an oracle diagnostic, not deployable LID training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_diagnostics import alpha_at, value_and_jacobian
from experiments.fm_mechanism_head_refit import sample_data, solve_weighted_head
from experiments.fm_mechanism_pilot import (
    canonical_posterior, gaussian_posterior, load_pilot_checkpoint,
)


VARIANTS = ("normalized", "restore_std", "remove_final_ln", "pre_ln_linear")


def feature_dictionary(field, points, lam, schedule):
    lam = torch.as_tensor(lam, dtype=points.dtype, device=points.device)
    if lam.ndim == 0:
        lam = lam.expand(len(points))
    alpha = alpha_at(lam, schedule)
    embedded = field.condition_embedding(lam.log())
    input_part = field.input(alpha[:, None] * points)
    condition_part = field.input_condition(embedded)
    state = input_part + condition_part
    for block in field.blocks:
        state = block(state, embedded)
    mean = state.mean(-1, keepdim=True)
    std = ((state-mean).square().mean(-1, keepdim=True) + field.output[0].eps).sqrt()
    normalized = field.output[1](field.output[0](state))
    return {
        "normalized": normalized,
        "restore_std": normalized * std,
        "remove_final_ln": field.output[1](state),
        "pre_ln_linear": state,
        "state_std": std,
        "state_mean": mean,
        "input_part_norm": input_part.norm(dim=-1, keepdim=True),
        "condition_part_norm": condition_part.norm(dim=-1, keepdim=True),
    }


@torch.no_grad()
def extract(field, points, lam, schedule):
    collected = {}
    for start in range(0, len(points), 512):
        part = feature_dictionary(field, points[start:start+512], lam[start:start+512], schedule)
        for name, value in part.items():
            collected.setdefault(name, []).append(value)
    return {name: torch.cat(parts) for name, parts in collected.items()}


def linear_readout(features, coefficient):
    return features @ coefficient[:-1] + coefficient[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    fixture = torch.load(args.checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    assert fixture.get("data_law", "gaussian") == "gaussian"
    basis = fixture["tangent_basis"].double()
    dim, intrinsic = basis.shape
    clean, noise, lam = sample_data(fixture, 32768, 20260905 + 701)
    held_clean, held_noise, held_lam = sample_data(fixture, 8192, 20260905 + 702)
    noisy, held_noisy = clean + lam[:, None]*noise, held_clean + held_lam[:, None]*held_noise
    target = gaussian_posterior(noisy, lam, fixture) @ basis
    held_oracle = gaussian_posterior(held_noisy, held_lam, fixture) @ basis
    held_target = held_clean @ basis
    args.output.mkdir(parents=True, exist_ok=False)
    rows, norm_rows, audits, coefficients = [], [], [], {}
    original_equivalence_error = 0.0
    for schedule in ("log_noise", "rectified_linear", "vp_trigonometric"):
        field, _ = load_pilot_checkpoint(args.checkpoint, schedule, "cpu")
        field = field.double().eval()
        assert getattr(field, "pilot_output_coordinate", "data") == "data"
        fit = extract(field, noisy, lam, schedule)
        held = extract(field, held_noisy, held_lam, schedule)
        expected = canonical_posterior(field, noisy[:64], lam[:64], schedule)
        reconstructed = field.output[-1](fit["normalized"][:64])
        equivalence = float((expected-reconstructed).abs().max().detach())
        assert equivalence < 1e-10, equivalence
        original_equivalence_error = max(original_equivalence_error, equivalence)
        fitted = {}
        for variant in VARIANTS:
            coefficient, audit = solve_weighted_head(fit[variant], target, lam)
            fitted[variant] = coefficient.float().double()
            coefficients[f"{schedule}/{variant}"] = fitted[variant]
            audits.append({"schedule": schedule, "variant": variant, **audit})
        for variant in ("original_tangent_projected",) + VARIANTS:
            if variant == "original_tangent_projected":
                prediction = field.output[-1](held["normalized"]).detach() @ basis
            else:
                prediction = linear_readout(held[variant], fitted[variant])
            risk = float(((prediction-held_oracle).square().sum(-1)/(dim*held_lam.square())).mean())
            total_risk = float(((prediction-held_target).square().sum(-1)/(dim*held_lam.square())).mean())
            for scale in (0.316227766, 1.0):
                points_t = held_clean[:24] @ basis
                def predict_tangent(t):
                    features = feature_dictionary(field, t @ basis.T, scale, schedule)
                    if variant == "original_tangent_projected":
                        return field.output[-1](features["normalized"]) @ basis
                    return linear_readout(features[variant], fitted[variant])
                value, jac = value_and_jacobian(predict_tangent, points_t, chunk=24)
                origin = predict_tangent(torch.zeros(1, intrinsic, dtype=torch.float64)).detach()
                tau2 = float(fixture["latent_std"])**2
                gain = tau2/(tau2+scale**2)
                oracle = gain*points_t
                response = jac.diagonal(dim1=-2, dim2=-1).sum(-1)
                correction = (value-points_t).square().sum(-1)/scale**2
                oracle_correction = (oracle-points_t).square().sum(-1)/scale**2
                full_error = response+correction-intrinsic*gain-oracle_correction
                euler = torch.einsum("bij,bj->bi", jac, points_t)-(value-origin)
                row = {
                    "schedule": schedule, "variant": variant, "lambda": scale,
                    "heldout_oracle_excess_risk": risk, "heldout_total_target_risk": total_risk,
                    "clean_query_count": 24,
                    "clean_value_error_energy_over_lambda2": float((value-oracle).square().sum(-1).mean()/scale**2),
                    "centered_euler_relative_mean": float((euler.norm(dim=-1)/points_t.norm(dim=-1)).mean()),
                    "response_oracle_mae": float((response-intrinsic*gain).abs().mean()),
                    "full_oracle_mae": float(full_error.abs().mean()),
                    "response_mean": float(response.mean()), "correction_mean": float(correction.mean()),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        for scale in (0.316227766, 1.0):
            points = held_clean[:2048]
            features = extract(field, points, torch.full((len(points),), scale, dtype=torch.float64), schedule)
            radius = points.norm(dim=-1)
            norm_rows.append({
                "schedule": schedule, "lambda": scale,
                "radius_std_correlation": float(torch.corrcoef(torch.stack([radius, features["state_std"].ravel()]))[0, 1]),
                "state_std_mean": float(features["state_std"].mean()),
                "state_std_std": float(features["state_std"].std()),
                "condition_input_norm_ratio_mean": float((features["condition_part_norm"]/features["input_part_norm"]).mean()),
            })
    torch.save(coefficients, args.output / "coefficients.pt")
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "source_checkpoint": str(args.checkpoint), "fit_count": 32768, "heldout_count": 8192,
        "fit_seed": 20260905+701, "heldout_seed": 20260905+702,
        "original_feature_equivalence_max_error": original_equivalence_error,
        "variants": {
            "normalized": "SiLU(LN(h)); existing head features",
            "restore_std": "std_eps(h)*SiLU(LN(h)); same feature count, explicit amplitude restoration",
            "remove_final_ln": "SiLU(h); removes final LN only at fixed frozen trunk",
            "pre_ln_linear": "h; removes both final LN and SiLU, same feature count",
        },
        "head_fitting": "Independent empirical LS per variant, shared scales and fit/heldout draws, exact Gaussian posterior targets, float64 SVD, coefficients rounded float32",
        "geometry_scope": "All comparisons tangent projected; exact 3x3 tangent Jacobian. No normal contribution is credited.",
        "claim_boundary": "Frozen representation interventions, not trunk retraining; raw h also removes SiLU and cannot isolate LayerNorm alone. Correlated norm statistics are descriptive.",
        "fit_audits": audits, "metrics": rows, "norm_statistics": norm_rows,
        "elapsed_seconds": time.monotonic()-started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
