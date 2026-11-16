"""Fixed-native conditioning versus input-path scale derivatives; frozen models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_mechanism_diagnostics import alpha_at
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint
from experiments.fm_mechanism_shell_probe import make_population


def relative_rms(vector, points):
    return float((vector.square().sum(-1).mean()/points.square().sum(-1).mean()).sqrt())


def correlation(left, right):
    if len(left) < 3 or left.std() < 1e-15 or right.std() < 1e-15:
        return None
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1])


def log_alpha_derivative(scale, schedule):
    if schedule == "log_noise":
        return 0.
    if schedule == "rectified_linear":
        return -scale/(1+scale)
    if schedule == "vp_trigonometric":
        return -scale**2/(1+scale**2)
    raise ValueError(schedule)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixture = torch.load(args.checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    law = fixture.get("data_law", "gaussian")
    population, clean, frame, *_ = make_population(fixture, args.queries, 20260905+2301)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, arrays = [], {}
    maximum_chain_error, maximum_oracle_fd_error = 0., 0.
    started = time.monotonic()
    for schedule in ("log_noise", "rectified_linear", "vp_trigonometric"):
        field, payload = load_pilot_checkpoint(args.checkpoint, schedule, "cpu")
        field = field.double().eval()
        assert payload["config"].get("output_coordinate", "data") == "data"
        assert payload["config"].get("input_coordinate", "native") == "native"
        for scale in (.1, .316227766, 1.):
            strata = {"population_clean": clean}
            if law == "uniform":
                generator = torch.Generator().manual_seed(20260905+431)
                halfwidth = population.latent_scale-3*scale
                assert bool((halfwidth > 0).all())
                latent = (2*torch.rand(args.queries, population.intrinsic_dim, generator=generator, dtype=torch.float64)-1)*halfwidth
                strata["interior_conditional"] = latent @ population.basis.T
            for stratum, points in strata.items():
                tau = points.new_full((len(points),), float(torch.log(torch.tensor(scale, dtype=torch.float64))))
                ones = torch.ones_like(tau)
                alpha = float(alpha_at(points.new_tensor(scale), schedule))
                native = alpha*points
                h = log_alpha_derivative(scale, schedule)
                value, condition_derivative = torch.autograd.functional.jvp(
                    lambda t: field(native, t), tau, ones)
                _, radial_derivative = torch.autograd.functional.jvp(
                    lambda r: canonical_posterior(field, r, scale, schedule), points, points)
                input_path_derivative = h*radial_derivative
                _, total_derivative = torch.autograd.functional.jvp(
                    lambda t: canonical_posterior(field, points, t.exp(), schedule), tau, ones)
                chain_error = float((total_derivative-condition_derivative-input_path_derivative).abs().max())
                maximum_chain_error = max(maximum_chain_error, chain_error)
                if population is not None:
                    moments = population.posterior(points, scale)
                    oracle = moments.mean
                    oracle_jr = (moments.tangent_gain*(points @ population.basis)) @ population.basis.T
                    _, oracle_total = torch.autograd.functional.jvp(
                        lambda t: population.posterior(points, t.exp()).mean, tau, ones)
                    oracle_fn = lambda t: population.posterior(points, t.exp()).mean
                else:
                    moments = circle_posterior(points, scale, radius=float(fixture["circle_radius"]))
                    oracle = moments.mean
                    oracle_jr = torch.einsum("nij,nj->ni", moments.jacobian, points)
                    oracle_total = -2*oracle_jr
                    oracle_fn = lambda t: circle_posterior(points, t.exp(), radius=float(fixture["circle_radius"])).mean
                finite_difference = (oracle_fn(tau+1e-4)-oracle_fn(tau-1e-4))/(2e-4)
                maximum_oracle_fd_error = max(maximum_oracle_fd_error, float((oracle_total-finite_difference).abs().max()))
                oracle_input = h*oracle_jr
                oracle_condition = oracle_total-oracle_input
                error = value-oracle
                condition_error = condition_derivative-oracle_condition
                input_error = input_path_derivative-oracle_input
                total_error = total_derivative-oracle_total
                def tangent(v):
                    return torch.einsum("ndi,nd->ni", frame, v)
                error_t = tangent(error)
                condition_error_t = tangent(condition_error)
                input_error_t = tangent(input_error)
                total_error_t = tangent(total_error)
                relative_query_value_error = error_t.square().sum(-1)/points.square().sum(-1)
                relative_query_total_error = total_error_t.square().sum(-1)/points.square().sum(-1)
                sum_condition_input_error = (condition_error_t.square().sum(-1)+input_error_t.square().sum(-1)).mean()
                row = {
                    "run": args.checkpoint.parent.name, "law": law, "schedule": schedule,
                    "lambda": scale, "stratum": stratum, "query_count": len(points),
                    "log_alpha_derivative": h,
                    "oracle_Jr_relative_rms": relative_rms(oracle_jr, points),
                    "oracle_input_path_relative_rms": relative_rms(oracle_input, points),
                    "oracle_condition_only_relative_rms": relative_rms(oracle_condition, points),
                    "oracle_total_scale_relative_rms": relative_rms(oracle_total, points),
                    "model_input_path_relative_rms": relative_rms(input_path_derivative, points),
                    "model_condition_only_relative_rms": relative_rms(condition_derivative, points),
                    "condition_mismatch_tangent_relative_rms": relative_rms(condition_error_t, points),
                    "input_path_mismatch_tangent_relative_rms": relative_rms(input_error_t, points),
                    "total_scale_mismatch_tangent_relative_rms": relative_rms(total_error_t, points),
                    "condition_input_mismatch_tangent_inner_product": float((condition_error_t*input_error_t).sum(-1).mean()),
                    "total_vs_uncancelled_tangent_error_energy_ratio": float(total_error_t.square().sum(-1).mean()/sum_condition_input_error) if sum_condition_input_error > 0 else None,
                    "value_error_energy_over_lambda2": float(error.square().sum(-1).mean()/scale**2),
                    "tangent_value_error_energy_over_lambda2": float(error_t.square().sum(-1).mean()/scale**2),
                    "correlation_relative_tangent_value_and_scale_error_energy": correlation(relative_query_value_error, relative_query_total_error),
                    "chain_identity_max_error": chain_error,
                }
                rows.append(row)
                prefix = f"{schedule}/{scale}/{stratum}"
                for name, tensor in {"points": points, "value": value, "oracle": oracle,
                    "condition_derivative": condition_derivative, "input_path_derivative": input_path_derivative,
                    "total_derivative": total_derivative, "oracle_condition": oracle_condition,
                    "oracle_input_path": oracle_input, "oracle_total": oracle_total,
                    "tangent_value_error": error_t, "tangent_total_scale_error": total_error_t}.items():
                    arrays[prefix+"/"+name] = tensor.detach()
            print(json.dumps({"law": law, "schedule": schedule, "lambda": scale, "elapsed": time.monotonic()-started}), flush=True)
    assert maximum_chain_error < 1e-10, maximum_chain_error
    assert maximum_oracle_fd_error < 1e-5, maximum_oracle_fd_error
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    torch.save(arrays, args.output / "query_arrays.pt")
    summary = {"source": str(args.checkpoint), "law": law, "metrics": rows,
        "chain_identity_max_error": maximum_chain_error, "oracle_scale_derivative_finite_difference_max_error": maximum_oracle_fd_error,
        "elapsed_seconds": time.monotonic()-started,
        "identity": "d_loglambda b_theta = (d_loglambda B_theta at native Y fixed) + (d_loglambda logalpha) J_btheta r. Required condition response = oracle total derivative minus oracle input-path term.",
        "scope": "Frozen derivative localization, not proof that mismatch caused value bias or that conditioning capacity is insufficient. Correlations use radius-normalized tangent energies. Uniform interior stratum is sampled conditionally from an inner box, not the ordinary query distribution."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"completed": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
