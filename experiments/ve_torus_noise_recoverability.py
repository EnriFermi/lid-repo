"""Probe when a VE diffusion starts recognizing injected noise on a torus.

For a clean query ``x`` and known Gaussian probe ``z``, evaluate the denoiser
on the antithetic pair ``x +/- sigma z``.  The VE score implies a predicted
noise

    z_hat(y, sigma) = (y - m_theta(y, sigma)) / sigma.

The normalized alignment of the antithetic prediction with ``z`` measures the
fraction of the injected perturbation that the model itself recognizes as
noise.  This diagnostic uses ordinary denoiser forwards only: no input
derivatives, divergence estimates, tangent/normal directions, or LID labels.
Known torus geometry and the population denoiser are used only in explicitly
post-hoc plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from experiments.ve_torus_knee import sample_product_torus
from models.training import (
    TrainingResult,
    diffusion_denoiser_output,
    load_checkpoint,
)


FloatArray = npt.NDArray[np.float64]
Denoiser = Callable[[Tensor, Tensor], Tensor]


def _statistics(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "standard_error": float(np.std(values, ddof=1) / math.sqrt(values.size)),
    }


def _batched_denoise(
    denoiser: Denoiser,
    inputs: Tensor,
    conditions: Tensor,
    *,
    batch_size: int,
) -> tuple[Tensor, int]:
    outputs: list[Tensor] = []
    calls = 0
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            outputs.append(denoiser(inputs[start:stop], conditions[start:stop]))
            calls += 1
    output = torch.cat(outputs, dim=0)
    if output.shape != inputs.shape or not torch.isfinite(output).all():
        raise ValueError("denoiser output must be finite and match its input")
    return output, calls


def antithetic_noise_recoverability(
    denoiser: Denoiser,
    queries: Tensor,
    noise: Tensor,
    sigma: float,
    *,
    batch_size: int,
    factor_width: int = 2,
) -> tuple[dict[str, FloatArray], int]:
    """Return per-query recoverability using ordinary denoiser forwards.

    ``noise`` has shape ``(query_count, probe_count, ambient_dim)``.  The same
    probes can be reused at every scale to make scale comparisons paired.
    """

    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("queries must have shape (nonempty, ambient_dim)")
    if noise.ndim != 3 or noise.shape[0] != queries.shape[0]:
        raise ValueError("noise must have shape (query_count, probes, ambient_dim)")
    if noise.shape[2] != queries.shape[1] or noise.shape[1] == 0:
        raise ValueError("noise dimensions must match queries")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if factor_width <= 0 or queries.shape[1] % factor_width:
        raise ValueError("factor_width must divide the ambient dimension")
    if not torch.isfinite(queries).all() or not torch.isfinite(noise).all():
        raise ValueError("queries and noise must be finite")

    plus = queries[:, None, :] + sigma * noise
    minus = queries[:, None, :] - sigma * noise
    paired_inputs = torch.stack((plus, minus), dim=2)
    flattened = paired_inputs.reshape(-1, queries.shape[1])
    conditions = torch.full(
        (flattened.shape[0],),
        sigma,
        device=flattened.device,
        dtype=flattened.dtype,
    )
    denoised, calls = _batched_denoise(
        denoiser, flattened, conditions, batch_size=batch_size
    )
    denoised = denoised.reshape(
        queries.shape[0], noise.shape[1], 2, queries.shape[1]
    )
    predicted_plus = (plus - denoised[:, :, 0, :]) / sigma
    predicted_minus = (minus - denoised[:, :, 1, :]) / sigma
    paired_prediction = 0.5 * (predicted_plus - predicted_minus)

    def normalized_alignment(section: slice) -> FloatArray:
        selected_noise = noise[:, :, section]
        selected_prediction = paired_prediction[:, :, section]
        numerator = torch.sum(selected_prediction * selected_noise, dim=(1, 2))
        denominator = torch.sum(selected_noise.square(), dim=(1, 2))
        if torch.any(denominator <= 0.0):
            raise ValueError("every query must have nonzero probe energy")
        return np.asarray((numerator / denominator).cpu().numpy(), dtype=np.float64)

    values: dict[str, FloatArray] = {
        "total": normalized_alignment(slice(0, queries.shape[1]))
    }
    for start in range(0, queries.shape[1], factor_width):
        factor = start // factor_width
        values[f"factor_{factor}"] = normalized_alignment(
            slice(start, start + factor_width)
        )
    return values, calls


def _circle_population_denoiser(
    points: FloatArray,
    *,
    center: FloatArray,
    radius: float,
    sigma: float,
) -> FloatArray:
    """Exact posterior mean for a uniform circle under Gaussian corruption."""

    from scipy.special import ive

    relative = points - center
    norms = np.linalg.norm(relative, axis=-1)
    concentration = radius * norms / sigma**2
    ratio = ive(1, concentration) / ive(0, concentration)
    direction = np.zeros_like(relative)
    nonzero = norms > np.finfo(np.float64).tiny
    direction[nonzero] = relative[nonzero] / norms[nonzero, None]
    return center + radius * ratio[..., None] * direction


def population_torus_recoverability(
    queries: FloatArray,
    noise: FloatArray,
    sigma: float,
    *,
    center: FloatArray,
    radii: FloatArray,
) -> dict[str, FloatArray]:
    """Population counterpart evaluated with the same queries and probes."""

    plus = queries[:, None, :] + sigma * noise
    minus = queries[:, None, :] - sigma * noise
    denoised_plus = np.empty_like(plus)
    denoised_minus = np.empty_like(minus)
    for factor, radius in enumerate(radii):
        section = slice(2 * factor, 2 * factor + 2)
        denoised_plus[:, :, section] = _circle_population_denoiser(
            plus[:, :, section],
            center=center[section],
            radius=float(radius),
            sigma=sigma,
        )
        denoised_minus[:, :, section] = _circle_population_denoiser(
            minus[:, :, section],
            center=center[section],
            radius=float(radius),
            sigma=sigma,
        )
    paired_prediction = 0.5 * (
        (plus - denoised_plus) / sigma - (minus - denoised_minus) / sigma
    )

    def normalized_alignment(section: slice) -> FloatArray:
        selected_noise = noise[:, :, section]
        numerator = np.sum(
            paired_prediction[:, :, section] * selected_noise, axis=(1, 2)
        )
        denominator = np.sum(np.square(selected_noise), axis=(1, 2))
        return np.asarray(numerator / denominator, dtype=np.float64)

    values = {"total": normalized_alignment(slice(0, queries.shape[1]))}
    for factor in range(len(radii)):
        section = slice(2 * factor, 2 * factor + 2)
        values[f"factor_{factor}"] = normalized_alignment(section)
    return values


def _scale_grid(
    minimum: float, maximum: float, dense_count: int, sparse_factor: float
) -> tuple[FloatArray, set[float]]:
    dense = np.geomspace(minimum, maximum, dense_count)
    sparse: list[float] = []
    value = minimum
    while value <= maximum * (1.0 + 1.0e-12):
        sparse.append(min(value, maximum))
        value *= sparse_factor
    if not math.isclose(sparse[-1], maximum):
        sparse.append(maximum)
    all_scales = np.unique(np.concatenate((dense, np.asarray(sparse))))
    return all_scales, set(sparse)


def _read_curve(path: Path) -> dict[str, FloatArray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty curve file: {path}")
    names = ("sigma", "learned_full_mean", "learned_response_mean", "oracle_full", "oracle_response")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in names
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exploratory_sparse_selector(
    rows: list[dict[str, Any]],
    *,
    fit_rmse_max: float = 0.02,
    next_transition_residual_min: float = 0.05,
    saturation_fraction: float = 0.90,
) -> dict[str, Any]:
    """Fit the first tube transition using sparse, label-free anchors.

    This is deliberately marked exploratory: its thresholds were specified
    while inspecting this pilot and require prospective validation elsewhere.
    """

    from scipy.optimize import least_squares

    sparse_rows = [row for row in rows if bool(row["sparse_anchor"])]
    scales = np.asarray([row["sigma"] for row in sparse_rows], dtype=np.float64)
    values = np.asarray(
        [row["learned_total_mean"] for row in sparse_rows], dtype=np.float64
    )

    def fit_prefix(count: int) -> dict[str, Any]:
        fit_scales = scales[:count]
        fit_values = values[:count]

        def residual(parameters: FloatArray) -> FloatArray:
            amplitude = parameters[0]
            delta = math.exp(parameters[1])
            prediction = amplitude * np.square(fit_scales) / (
                delta**2 + np.square(fit_scales)
            )
            return prediction - fit_values

        initial = np.asarray(
            [min(0.99, max(float(np.max(fit_values)), 0.05)), math.log(scales[0] / 2.0)],
            dtype=np.float64,
        )
        fitted = least_squares(
            residual,
            initial,
            bounds=(
                np.asarray([1.0e-6, math.log(scales[0] / 1_000.0)]),
                np.asarray([1.0, math.log(scales[-1] * 1_000.0)]),
            ),
        )
        amplitude = float(fitted.x[0])
        delta = float(math.exp(fitted.x[1]))
        prediction = amplitude * np.square(fit_scales) / (
            delta**2 + np.square(fit_scales)
        )
        return {
            "prefix_count": count,
            "amplitude": amplitude,
            "delta": delta,
            "fit_rmse": float(
                np.sqrt(np.mean(np.square(prediction - fit_values)))
            ),
        }

    attempts: list[dict[str, Any]] = []
    for prefix_count in range(3, len(scales)):
        fitted = fit_prefix(prefix_count)
        next_scale = float(scales[prefix_count])
        next_observed = float(values[prefix_count])
        amplitude = float(fitted["amplitude"])
        delta = float(fitted["delta"])
        next_predicted = amplitude * next_scale**2 / (delta**2 + next_scale**2)
        next_residual = next_observed - next_predicted
        saturated = scales[:prefix_count][
            values[:prefix_count] >= saturation_fraction * amplitude
        ]
        attempt = {
            **fitted,
            "next_sigma": next_scale,
            "next_observed": next_observed,
            "next_predicted": next_predicted,
            "next_residual": next_residual,
            "saturated_anchor_count": int(saturated.size),
        }
        attempts.append(attempt)
        if (
            float(fitted["fit_rmse"]) <= fit_rmse_max
            and saturated.size >= 2
            and next_residual >= next_transition_residual_min
        ):
            selected_sigma = float(np.exp(np.median(np.log(saturated))))
            return {
                "status": "selected",
                "validation_scope": "exploratory_posthoc_rule_not_prospective",
                "uses_geometry_or_lid_labels": False,
                "fit_rmse_max": fit_rmse_max,
                "next_transition_residual_min": next_transition_residual_min,
                "saturation_fraction": saturation_fraction,
                "scale_evaluations": prefix_count + 1,
                "fit_scales": scales[:prefix_count].tolist(),
                "fit_values": values[:prefix_count].tolist(),
                "selected_sigma": selected_sigma,
                "saturated_scales": saturated.tolist(),
                "fit": attempt,
                "attempts": attempts,
            }
    return {
        "status": "abstained",
        "validation_scope": "exploratory_posthoc_rule_not_prospective",
        "uses_geometry_or_lid_labels": False,
        "fit_rmse_max": fit_rmse_max,
        "next_transition_residual_min": next_transition_residual_min,
        "saturation_fraction": saturation_fraction,
        "scale_evaluations": len(scales),
        "attempts": attempts,
    }


def _make_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    ambient_dim: int,
    radii: FloatArray,
    source_summary: dict[str, Any],
    flipd_curve: dict[str, FloatArray],
    sparse_selector: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
    learned_q = np.asarray([row["learned_total_mean"] for row in rows])
    learned_q05 = np.asarray([row["learned_total_q05"] for row in rows])
    learned_q95 = np.asarray([row["learned_total_q95"] for row in rows])
    oracle_q = np.asarray([row["oracle_total_mean"] for row in rows])
    sparse = np.asarray([bool(row["sparse_anchor"]) for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True)
    axes[0].plot(sigma, learned_q, color="#7c3aed", linewidth=2.2)
    axes[0].fill_between(
        sigma, learned_q05, learned_q95, color="#7c3aed", alpha=0.14, linewidth=0
    )
    axes[0].set_ylabel("recognized-noise fraction Q")
    axes[0].set_title(
        "Model-only antithetic noise recoverability (ordinary forwards)"
    )
    axes[0].grid(alpha=0.22)
    axes[1].plot(
        sigma,
        ambient_dim * (1.0 - learned_q),
        color="#2563eb",
        linewidth=2.2,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("normalized VE noise sigma")
    axes[1].set_ylabel("D (1 - Q)")
    axes[1].grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_dir / "model_only_recoverability.png", dpi=200)
    figure.savefig(output_dir / "model_only_recoverability.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.plot(sigma, learned_q, color="#7c3aed", linewidth=2.2, label="learned Q")
    axis.fill_between(
        sigma, learned_q05, learned_q95, color="#7c3aed", alpha=0.12, linewidth=0
    )
    axis.plot(
        sigma,
        oracle_q,
        color="black",
        linestyle="--",
        linewidth=1.7,
        label="population Q",
    )
    axis.scatter(
        sigma[sparse],
        learned_q[sparse],
        color="#f97316",
        s=28,
        zorder=4,
        label="factor-2 sparse anchors",
    )
    for factor in range(len(radii) + 1):
        level = 0.5 + 0.5 * factor / len(radii)
        axis.axhline(level, color="#64748b", linestyle=":", linewidth=0.9)
    for factor, radius in enumerate(radii):
        axis.axvline(
            radius,
            color="#334155",
            linestyle=":",
            linewidth=1.0,
            label="known circle radii" if factor == 0 else None,
        )
    axis.set_xscale("log")
    axis.set_xlabel("normalized VE noise sigma")
    axis.set_ylabel("recognized-noise fraction Q")
    axis.set_title("Post-hoc population and geometry alignment")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "posthoc_recoverability.png", dpi=200)
    figure.savefig(output_dir / "posthoc_recoverability.svg")
    plt.close(figure)

    source_sigma = flipd_curve["sigma"]
    learned_response = np.interp(
        np.log(sigma), np.log(source_sigma), flipd_curve["learned_response_mean"]
    )
    learned_full = np.interp(
        np.log(sigma), np.log(source_sigma), flipd_curve["learned_full_mean"]
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.plot(
        sigma,
        ambient_dim * (1.0 - learned_q),
        color="#7c3aed",
        linewidth=2.2,
        label="forward-only D(1-Q)",
    )
    axis.plot(
        sigma,
        learned_response,
        color="#0891b2",
        linewidth=1.8,
        label="clean-query FLIPD response R",
    )
    axis.plot(
        sigma,
        learned_full,
        color="#2563eb",
        linewidth=1.6,
        alpha=0.8,
        label="clean-query full FLIPD F",
    )
    for factor, knee in enumerate(source_summary["knees"]):
        axis.axvline(
            float(knee["learned_slope_knee"]),
            color="#64748b",
            linestyle="--",
            linewidth=0.9,
            label="post-hoc FLIPD knees" if factor == 0 else None,
        )
    axis.set_xscale("log")
    axis.set_xlabel("normalized VE noise sigma")
    axis.set_ylabel("dimension units")
    axis.set_title("Post-hoc alignment with the saved FLIPD readout")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "flipd_alignment.png", dpi=200)
    figure.savefig(output_dir / "flipd_alignment.svg")
    plt.close(figure)

    if sparse_selector["status"] == "selected":
        fitted = sparse_selector["fit"]
        fit_scales = np.asarray(sparse_selector["fit_scales"], dtype=np.float64)
        fit_values = np.asarray(sparse_selector["fit_values"], dtype=np.float64)
        next_scale = float(fitted["next_sigma"])
        next_observed = float(fitted["next_observed"])
        plot_scales = np.geomspace(fit_scales[0], next_scale, 256)
        amplitude = float(fitted["amplitude"])
        delta = float(fitted["delta"])
        prediction = amplitude * np.square(plot_scales) / (
            delta**2 + np.square(plot_scales)
        )
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        axis.scatter(
            fit_scales,
            fit_values,
            color="#7c3aed",
            s=52,
            zorder=3,
            label="anchors used for first-transition fit",
        )
        axis.plot(
            plot_scales,
            prediction,
            color="#7c3aed",
            linewidth=1.8,
            linestyle="--",
            label="one-tube saturation fit",
        )
        axis.scatter(
            [next_scale],
            [next_observed],
            marker="x",
            color="#dc2626",
            s=82,
            linewidth=2.0,
            label="next-transition violation",
        )
        axis.axvline(
            float(sparse_selector["selected_sigma"]),
            color="#059669",
            linewidth=1.6,
            label="selected interior sigma",
        )
        axis.set_xscale("log")
        axis.set_xlabel("normalized VE noise sigma")
        axis.set_ylabel("recognized-noise fraction Q")
        axis.set_title("Exploratory five-scale selector (no geometry or LID labels)")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "sparse_selector.png", dpi=200)
        figure.savefig(output_dir / "sparse_selector.svg")
        plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: TrainingResult = load_checkpoint(args.checkpoint, device=args.device)
    if result.family != "gaussian_diffusion":
        raise ValueError("the checkpoint must be a Gaussian diffusion")
    if result.model.config.ambient_dim != 2 * len(args.radii):
        raise ValueError("checkpoint ambient dimension does not match radii")
    if args.query_size % 2:
        raise ValueError("query_size must be even for exact empirical centering")

    parameter = next(result.model.parameters())
    raw_queries = sample_product_torus(
        args.query_size, args.radii, seed=args.query_seed
    )
    normalized_queries = (
        torch.as_tensor(raw_queries, device=parameter.device, dtype=parameter.dtype)
        - result.normalization_mean.to(device=parameter.device, dtype=parameter.dtype)
    ) / result.normalization_scale
    rng = np.random.default_rng(args.noise_seed)
    noise_numpy = rng.standard_normal(
        (args.query_size, args.probe_count, result.model.config.ambient_dim)
    ).astype(np.float32)
    noise = torch.as_tensor(noise_numpy, device=parameter.device, dtype=parameter.dtype)
    scales, sparse_scales = _scale_grid(
        args.sigma_min, args.sigma_max, args.scale_count, args.sparse_factor
    )
    normalized_radii = np.asarray(args.radii, dtype=np.float64) / result.normalization_scale
    center = -np.asarray(result.normalization_mean, dtype=np.float64) / result.normalization_scale
    normalized_queries_numpy = np.asarray(
        normalized_queries.detach().cpu().numpy(), dtype=np.float64
    )
    noise_oracle = np.asarray(noise_numpy, dtype=np.float64)

    def denoiser(inputs: Tensor, conditions: Tensor) -> Tensor:
        return diffusion_denoiser_output(
            result.model,
            inputs,
            conditions,
            preconditioning=result.config.diffusion_preconditioning,
            data_std=result.config.diffusion_data_std,
        )

    rows: list[dict[str, Any]] = []
    model_batch_calls = 0
    started = time.monotonic()
    for index, sigma in enumerate(scales):
        learned, calls = antithetic_noise_recoverability(
            denoiser,
            normalized_queries,
            noise,
            float(sigma),
            batch_size=args.evaluation_batch_size,
        )
        model_batch_calls += calls
        oracle = population_torus_recoverability(
            normalized_queries_numpy,
            noise_oracle,
            float(sigma),
            center=center,
            radii=normalized_radii,
        )
        row: dict[str, Any] = {
            "sigma": float(sigma),
            "sparse_anchor": any(
                math.isclose(float(sigma), value, rel_tol=1.0e-12, abs_tol=0.0)
                for value in sparse_scales
            ),
        }
        for source_name, values in (("learned", learned), ("oracle", oracle)):
            for component, component_values in values.items():
                for statistic, value in _statistics(component_values).items():
                    row[f"{source_name}_{component}_{statistic}"] = value
        rows.append(row)
        if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(scales):
            print(
                f"evaluated {index + 1:03d}/{len(scales)} scales in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    _write_rows(output_dir / "recoverability.csv", rows)
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    flipd_curve = _read_curve(args.flipd_curves)
    sparse_selector = exploratory_sparse_selector(rows)
    if sparse_selector["status"] == "selected":
        selected_sigma = float(sparse_selector["selected_sigma"])
        source_sigma = flipd_curve["sigma"]
        sparse_selector["posthoc_selected_point"] = {
            "learned_full_flipd": float(
                np.interp(
                    math.log(selected_sigma),
                    np.log(source_sigma),
                    flipd_curve["learned_full_mean"],
                )
            ),
            "learned_response": float(
                np.interp(
                    math.log(selected_sigma),
                    np.log(source_sigma),
                    flipd_curve["learned_response_mean"],
                )
            ),
            "population_full_flipd": float(
                np.interp(
                    math.log(selected_sigma),
                    np.log(source_sigma),
                    flipd_curve["oracle_full"],
                )
            ),
            "known_lid": len(args.radii),
        }
    (output_dir / "sparse_selector.json").write_text(
        json.dumps(sparse_selector, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    learned_q = np.asarray([row["learned_total_mean"] for row in rows])
    oracle_q = np.asarray([row["oracle_total_mean"] for row in rows])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ve_torus_antithetic_noise_recoverability",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": result.checkpoint_sha256,
        "query_definition": (
            "independent uniform clean points on S1(R1) x S1(R2), normalized "
            "with the checkpoint preprocessing"
        ),
        "ambient_space": f"R^{result.model.config.ambient_dim}",
        "intrinsic_dimension": len(args.radii),
        "raw_radii": [float(value) for value in args.radii],
        "normalized_radii": normalized_radii.tolist(),
        "query_count": args.query_size,
        "probe_count_per_query": args.probe_count,
        "query_seed": args.query_seed,
        "noise_seed": args.noise_seed,
        "scale_count": int(len(scales)),
        "sigma_range": [float(scales[0]), float(scales[-1])],
        "sparse_anchor_factor": args.sparse_factor,
        "sparse_anchor_count": int(sum(bool(row["sparse_anchor"]) for row in rows)),
        "model_input_derivatives": 0,
        "antithetic_inputs_per_query_probe": 2,
        "logical_batched_forward_passes_per_scale": 1,
        "physical_minibatch_forward_calls": model_batch_calls,
        "learned_vs_population_q_rmse": float(
            np.sqrt(np.mean(np.square(learned_q - oracle_q)))
        ),
        "annotation_scope": (
            "The learned Q curve uses only ordinary model forwards. Population "
            "denoising, factor coordinates, radii, LID, and saved FLIPD knees "
            "are post-hoc validation information."
        ),
        "exploratory_sparse_selector": sparse_selector,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _make_plots(
        output_dir,
        rows,
        ambient_dim=result.model.config.ambient_dim,
        radii=normalized_radii,
        source_summary=source_summary,
        flipd_curve=flipd_curve,
        sparse_selector=sparse_selector,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--flipd-curves", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radii", type=float, nargs="+", default=[1.0, 16.0])
    parser.add_argument("--query-seed", type=int, default=20_260_902 + 303)
    parser.add_argument("--noise-seed", type=int, default=20_260_904)
    parser.add_argument("--query-size", type=int, default=2_048)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=8.0)
    parser.add_argument("--scale-count", type=int, default=129)
    parser.add_argument("--sparse-factor", type=float, default=2.0)
    parser.add_argument("--evaluation-batch-size", type=int, default=32_768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
