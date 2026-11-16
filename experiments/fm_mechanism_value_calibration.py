"""Held-out tangent-value calibration of fixed FM mechanism checkpoints.

This uses known population geometry and is a diagnostic, not a deployable LID
estimator. The first ``test_count`` saved queries are held out; the next 512
fit calibration maps. Gaussian and Uniform use their own finite-scale oracles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_oracles import AffineFixture
from experiments.fm_mechanism_pilot import (
    canonical_posterior,
    load_pilot_checkpoint,
)


def append_one(value):
    return torch.cat([value, value.new_ones((len(value), 1))], dim=-1)


def fit_error_models(inputs, errors, include_radial=True):
    mean_x, mean_e = inputs.mean(0), errors.mean(0)
    centered_x, centered_e = inputs - mean_x, errors - mean_e
    radial = (centered_x * centered_e).sum() / centered_x.square().sum()
    design = append_one(inputs)
    affine = torch.linalg.lstsq(design, errors).solution
    if not include_radial:
        return {"constant": mean_e, "affine": affine}
    # One additional shared cubic-radial coefficient tests norm dependence
    # without fitting a flexible nonlinear predictor to the calibration set.
    norm_scale = inputs.square().sum(-1).mean()
    cubic = inputs * (inputs.square().sum(-1) / norm_scale)[:, None]
    cubic_affine = torch.linalg.lstsq(design, cubic).solution
    cubic_residual = cubic - design @ cubic_affine
    affine_residual = errors - design @ affine
    cubic_gain = (cubic_residual * affine_residual).sum() / cubic_residual.square().sum()
    return {
        "constant": mean_e,
        "radial_gain": radial,
        "radial_offset": mean_e - radial * mean_x,
        "affine": affine,
        "cubic_gain": cubic_gain,
        "cubic_norm_scale": norm_scale,
        "cubic_affine": affine - cubic_gain * cubic_affine,
    }


def residuals(inputs, errors, fitted):
    if "radial_gain" not in fitted:
        return {
            "raw": errors,
            "constant": errors - fitted["constant"],
            "affine": errors - append_one(inputs) @ fitted["affine"],
        }
    return {
        "raw": errors,
        "constant": errors - fitted["constant"],
        "radial": errors - fitted["radial_offset"] - fitted["radial_gain"] * inputs,
        "affine": errors - append_one(inputs) @ fitted["affine"],
        "affine_cubic_radial": (
            errors - append_one(inputs) @ fitted["cubic_affine"]
            - fitted["cubic_gain"] * inputs
            * (inputs.square().sum(-1) / fitted["cubic_norm_scale"])[:, None]
        ),
    }


def scalar(value):
    return float(value.detach())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(checkpoint: Path, output: Path, derivative_count: int = 24, test_count: int = 128):
    torch.set_num_threads(4)
    dtype = torch.float64
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    law = data.get("data_law", "gaussian")
    if law not in {"gaussian", "uniform"}:
        raise ValueError("Exact oracle is available only for Gaussian and Uniform")
    if not 1 <= derivative_count <= test_count:
        raise ValueError("derivative_count must lie between 1 and test_count")
    d = int(payload["config"]["intrinsic_dim"])
    dimension = int(payload["config"]["ambient_dim"])
    q, _ = torch.linalg.qr(data["orthonormal_basis"].to(dtype))
    cache_path = checkpoint.parent / f"diagnostics_{payload['step']:06d}" / "query_arrays.npz"
    cache = np.load(cache_path) if cache_path.exists() else None
    if cache is not None:
        q = torch.from_numpy(cache["basis"]).to(dtype)
    tangent = q[:, :d]
    total_count = test_count + 512
    clean = data["clean_queries"][:total_count].to(dtype)
    noise = data["query_noise"][:total_count].to(dtype)
    if len(clean) != total_count:
        raise ValueError("Need test_count+512 saved queries")
    test, calibration = slice(0, test_count), slice(test_count, total_count)
    sigma_data = float(data["latent_std"])
    latent_scale = sigma_data if law == "gaussian" else float(data["latent_half_width"])
    oracle_fixture = AffineFixture(
        law, tangent, data["center"].to(dtype),
        torch.full((d,), latent_scale, dtype=dtype),
    )
    true_norm = torch.linalg.vector_norm(clean @ tangent, dim=-1) / sigma_data
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=dtype)
    bin_edges = torch.quantile(true_norm[calibration], quantiles)
    bin_edges[0], bin_edges[-1] = -torch.inf, torch.inf
    output.mkdir(parents=True, exist_ok=True)
    rows, bins, boundary_rows, derivative_rows, coefficients, arrays = [], [], [], [], {}, {}
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(checkpoint, schedule, "cpu")
        field = field.to(dtype).eval()
        for lam in (0.1, 0.316227766, 1.0):

            def fn(points):
                return canonical_posterior(
                    field, points, lam, schedule,
                    input_coordinate=payload["config"].get("input_coordinate", "native"),
                    output_coordinate=payload["config"].get("output_coordinate", "data"),
                )

            location_data = {}
            for location in ("clean", "noisy"):
                points = clean if location == "clean" else clean + lam * noise
                with torch.no_grad():
                    values = fn(points)
                input_t = points @ tangent
                pred_t = values @ tangent
                oracle = oracle_fixture.posterior(points, lam)
                oracle_t = oracle.tangent_mean
                error_t = pred_t - oracle_t
                fitted = fit_error_models(
                    input_t[calibration], error_t[calibration],
                    include_radial=law == "gaussian",
                )
                output_map = torch.linalg.lstsq(
                    append_one(pred_t[calibration]), oracle_t[calibration]
                ).solution
                label = f"{schedule}__{lam:.9g}__{location}"
                coefficients[label] = {
                    key: value.detach().tolist() for key, value in fitted.items()
                }
                coefficients[label]["output_map"] = output_map.tolist()
                for split, indices in (("calibration", calibration), ("test", test)):
                    residual = residuals(input_t[indices], error_t[indices], fitted)
                    row = {
                        "schedule": schedule, "lambda": lam,
                        "location": location, "split": split,
                        "count": len(input_t[indices]),
                        "constant_offset_norm": scalar(fitted["constant"].norm()),
                        "radial_error_gain": scalar(fitted["radial_gain"]) if "radial_gain" in fitted else "not_fitted_uniform",
                        "affine_linear_frobenius": scalar(fitted["affine"][:-1].norm()),
                        "affine_skew_frobenius": scalar(
                            (fitted["affine"][:-1] - fitted["affine"][:-1].T).norm() / 2
                        ),
                    }
                    for name, value in residual.items():
                        row[f"{name}_error_energy_over_lambda2"] = scalar(
                            value.square().sum(-1).mean() / lam**2
                        )
                    output_residual = append_one(pred_t[indices]) @ output_map - oracle_t[indices]
                    row["output_affine_error_energy_over_lambda2"] = scalar(
                        output_residual.square().sum(-1).mean() / lam**2
                    )
                    row["raw_mean_coordinate_error_over_lambda2"] = (
                        row["raw_error_energy_over_lambda2"] / dimension
                    )
                    row["normal_oracle_error_energy_over_lambda2"] = scalar(
                        (values[indices] @ q[:, d:]).square().sum(-1).mean() / lam**2
                    )
                    rows.append(row)
                test_residual = residuals(input_t[test], error_t[test], fitted)
                if law == "uniform":
                    face_ratio = (latent_scale-(clean @ tangent).abs()).min(-1).values / lam
                    strata = {
                        "boundary_lt1": face_ratio[test] < 1,
                        "transition_1to3": (face_ratio[test] >= 1) & (face_ratio[test] <= 3),
                        "interior_gt3": face_ratio[test] > 3,
                    }
                    for stratum, mask in strata.items():
                        if not bool(mask.any()):
                            continue
                        corrected_t = append_one(pred_t[test][mask]) @ output_map
                        boundary_rows.append({
                            "schedule": schedule, "lambda": lam, "location": location,
                            "stratum": stratum, "count": int(mask.sum()),
                            "face_distance_over_lambda_mean": scalar(face_ratio[test][mask].mean()),
                            "raw_oracle_error_energy_over_lambda2": scalar(error_t[test][mask].square().sum(-1).mean()/lam**2),
                            "constant_residual_energy_over_lambda2": scalar(test_residual["constant"][mask].square().sum(-1).mean()/lam**2),
                            "affine_residual_energy_over_lambda2": scalar(test_residual["affine"][mask].square().sum(-1).mean()/lam**2),
                            "output_affine_oracle_error_energy_over_lambda2": scalar((corrected_t-oracle_t[test][mask]).square().sum(-1).mean()/lam**2),
                            "population_posterior_bias_energy_over_lambda2": scalar((oracle_t[test][mask]-input_t[test][mask]).square().sum(-1).mean()/lam**2),
                            "oracle_response_mean": scalar(oracle.response[test][mask].mean()),
                            "oracle_full_mean": scalar(oracle.full[test][mask].mean()),
                        })
                for index in (range(len(bin_edges) - 1) if law == "gaussian" else []):
                    mask = (true_norm[test] > bin_edges[index]) & (true_norm[test] <= bin_edges[index + 1])
                    if not bool(mask.any()):
                        continue
                    selected_x = input_t[test][mask]
                    selected_error = error_t[test][mask]
                    radial_per_query = (selected_x * selected_error).sum(-1) / selected_x.square().sum(-1)
                    radial_part = radial_per_query[:, None] * selected_x
                    bins.append({
                        "schedule": schedule, "lambda": lam,
                        "location": location, "bin": index,
                        "count": int(mask.sum()),
                        "true_latent_norm_mean_in_std_units": scalar(true_norm[test][mask].mean()),
                        "radial_relative_error_mean": scalar(radial_per_query.mean()),
                        "raw_error_energy_over_lambda2": scalar(selected_error.square().sum(-1).mean()/lam**2),
                        "radial_component_energy_over_lambda2": scalar(radial_part.square().sum(-1).mean()/lam**2),
                        "nonradial_component_energy_over_lambda2": scalar((selected_error-radial_part).square().sum(-1).mean()/lam**2),
                        "affine_residual_energy_over_lambda2": scalar(test_residual["affine"][mask].square().sum(-1).mean()/lam**2),
                    })
                location_data[location] = {
                    "output_map": output_map, "points": points,
                    "values": values, "pred_t": pred_t, "oracle_t": oracle_t,
                }
                arrays[label + "__test_pred_t"] = pred_t[test].numpy()
                arrays[label + "__test_oracle_t"] = oracle_t[test].numpy()
                arrays[label + "__test_input_t"] = input_t[test].numpy()

            # Use exact full derivatives only once. The affine intervention
            # preserves the normal output function; explicitly differentiate
            # its calibrated tangent outputs using the same AD helper.
            derivative_points = clean[:derivative_count]
            key = f"{schedule}__{lam:.9g}__clean"
            if cache is not None and key+"__tangent_jacobian" in cache:
                if len(cache[key+"__normal_trace"]) < derivative_count:
                    raise ValueError("Cached exact trace has too few test queries")
                torch.testing.assert_close(
                    torch.from_numpy(cache["clean"][:derivative_count]), derivative_points,
                    atol=0, rtol=0,
                )
                with torch.no_grad():
                    value = fn(derivative_points)
                torch.testing.assert_close(
                    value, torch.from_numpy(cache[key+"__posterior"][:derivative_count]),
                    atol=2e-9, rtol=2e-9,
                )
                baseline_tangent_jac = torch.from_numpy(cache[key+"__tangent_jacobian"][:derivative_count])
                normal_trace = torch.from_numpy(cache[key+"__normal_trace"][:derivative_count])
            else:
                value, jac = value_and_jacobian(fn, derivative_points)
                baseline_tangent_jac = torch.einsum("di,bdk,kj->bij", tangent, jac, tangent)
                baseline_response = jac.diagonal(dim1=-2, dim2=-1).sum(-1)
                normal_trace = baseline_response - baseline_tangent_jac.diagonal(dim1=-2, dim2=-1).sum(-1)
            derivative_oracle = oracle_fixture.posterior(derivative_points, lam)
            oracle_tangent_jac = torch.diag_embed(derivative_oracle.tangent_gain)
            normal_displacement = (value - derivative_points) @ q[:, d:]
            normal_correction = normal_displacement.square().sum(-1) / lam**2
            for fit_location in ("identity", "clean", "noisy"):
                if fit_location == "identity":
                    matrix = torch.cat([torch.eye(d, dtype=dtype), torch.zeros(1, d, dtype=dtype)])
                else:
                    matrix = location_data[fit_location]["output_map"]

                def calibrated_tangent(points):
                    return append_one(fn(points) @ tangent) @ matrix

                calibrated_value, calibrated_jac = value_and_jacobian(
                    calibrated_tangent, derivative_points
                )
                derivative_t = calibrated_jac @ tangent
                chain_rule = torch.einsum("ij,bjk->bik", matrix[:-1].T, baseline_tangent_jac)
                torch.testing.assert_close(derivative_t, chain_rule, atol=2e-9, rtol=2e-9)
                response = derivative_t.diagonal(dim1=-2, dim2=-1).sum(-1) + normal_trace
                correction_t = (calibrated_value - derivative_points @ tangent).square().sum(-1)/lam**2
                full = response + correction_t + normal_correction
                derivative_rows.append({
                    "schedule": schedule, "lambda": lam,
                    "fit_location": fit_location, "test_count": derivative_count,
                    "response_mean": scalar(response.mean()),
                    "response_mae_lid": scalar((response-d).abs().mean()),
                    "tangent_correction_mean": scalar(correction_t.mean()),
                    "normal_correction_mean": scalar(normal_correction.mean()),
                    "full_mean": scalar(full.mean()),
                    "full_mae_lid": scalar((full-d).abs().mean()),
                    "full_mae_finite_scale_oracle": scalar((full-derivative_oracle.full).abs().mean()),
                    "response_mae_finite_scale_oracle": scalar((response-derivative_oracle.response).abs().mean()),
                    "oracle_full_mean": scalar(derivative_oracle.full.mean()),
                    "oracle_response_mean": scalar(derivative_oracle.response.mean()),
                    "tangent_jacobian_oracle_frobenius_sq": scalar(
                        (derivative_t-oracle_tangent_jac).square().sum((-1,-2)).mean()
                    ),
                    "chain_rule_max_error": scalar((derivative_t-chain_rule).abs().max()),
                })
            print(schedule, lam, "done", flush=True)

    write_csv(output / "value_error_decomposition.csv", rows)
    if bins:
        write_csv(output / "norm_bins.csv", bins)
    if boundary_rows:
        write_csv(output / "boundary_strata.csv", boundary_rows)
    write_csv(output / "readout_calibration.csv", derivative_rows)
    (output / "coefficients.json").write_text(json.dumps(coefficients, indent=2))
    arrays["test_true_latent_norm"] = true_norm[test].numpy()
    np.savez_compressed(output / "test_tangent_predictions.npz", **arrays)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256((checkpoint.parent / "data.pt").read_bytes()).hexdigest(),
        "calibration_indices": [test_count, total_count], "test_indices": [0, test_count],
        "derivative_test_indices": [0, derivative_count],
        "dtype": "float64", "device": "cpu", "torch_threads": 4,
        "norm_bin_edges": [float(v) if np.isfinite(float(v)) else None for v in bin_edges],
        "interpretation": "Posthoc known-oracle output calibration; not deployable, no retraining.",
        "oracle": law + " exact finite-scale posterior; Uniform moments include boundary asymmetry",
        "gaussian_radial_models_enabled": law == "gaussian",
        "exact_derivative_cache": str(cache_path) if cache is not None else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if law == "uniform":
        print(json.dumps({"output": str(output), "value_rows": len(rows), "derivative_rows": len(derivative_rows), "boundary_rows": len(boundary_rows)}))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"log_noise": "#2d6a4f", "rectified_linear": "#c44e52", "vp_trigonometric": "#3b75af"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for col, lam in enumerate((0.1, 0.316227766, 1.0)):
        for schedule in payload["models"]:
            selected = [b for b in bins if b["schedule"] == schedule and b["lambda"] == lam and b["location"] == "clean"]
            x = [b["true_latent_norm_mean_in_std_units"] for b in selected]
            axes[0, col].plot(x, [b["radial_relative_error_mean"] for b in selected], "o-", label=schedule, color=colors[schedule])
            axes[1, col].plot(x, [b["affine_residual_energy_over_lambda2"] for b in selected], "o-", color=colors[schedule])
        axes[0, col].axhline(0, color="grey", linewidth=0.8)
        axes[0, col].set_title(f"Physical scale = {lam:.3g}")
        axes[1, col].set_xlabel("True tangent norm / latent std")
    axes[0, 0].set_ylabel("Signed radial error / tangent norm")
    axes[1, 0].set_ylabel("Residual after affine fit / scale²")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed 8k-step checkpoint: held-out clean-query tangent distortion")
    fig.savefig(output / "tangent_norm_dependence.png", dpi=180)
    fig.savefig(output / "tangent_norm_dependence.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(output), "value_rows": len(rows), "derivative_rows": len(derivative_rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derivative-count", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=128)
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.derivative_count, args.test_count)
