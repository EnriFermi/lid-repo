"""Train and audit a VE denoiser on a two-scale product torus.

The product manifold

    S^1(R_1) x ... x S^1(R_k) -> R^(2k)

has one intrinsic coordinate at each prescribed geometric scale.  For a
Gaussian corruption kernel its posterior and FLIPD readout factorize over the
circles, so the population curve is available by one-dimensional quadrature.
This makes the construction a direct test of whether learned FLIPD knees track
loss of embedded-coordinate information rather than an arbitrary time index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from models.readouts import diffusion_flipd
from models.training import (
    TrainingConfig,
    TrainingResult,
    diffusion_denoiser_output,
    load_checkpoint,
    train_model,
)


FloatArray = npt.NDArray[np.float64]


def sample_product_torus(
    sample_count: int,
    radii: tuple[float, ...] | list[float],
    *,
    seed: int,
) -> npt.NDArray[np.float32]:
    """Sample a product of circles with exact antipodal empirical centering."""

    radii_array = np.asarray(radii, dtype=np.float64)
    if isinstance(sample_count, bool) or sample_count <= 0 or sample_count % 2:
        raise ValueError("sample_count must be a positive even integer")
    if radii_array.ndim != 1 or radii_array.size == 0:
        raise ValueError("radii must be a non-empty one-dimensional sequence")
    if not np.isfinite(radii_array).all() or np.any(radii_array <= 0.0):
        raise ValueError("radii must be finite and positive")

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=(sample_count // 2, len(radii)))
    base = np.empty((sample_count // 2, 2 * len(radii)), dtype=np.float64)
    for factor, radius in enumerate(radii_array):
        base[:, 2 * factor] = radius * np.cos(angles[:, factor])
        base[:, 2 * factor + 1] = radius * np.sin(angles[:, factor])
    samples = np.concatenate((base, -base), axis=0).astype(np.float32)
    rng.shuffle(samples, axis=0)
    return samples


def normalized_torus_radii(
    radii: tuple[float, ...] | list[float], normalization_scale: float
) -> FloatArray:
    """Convert raw radii to the checkpoint's global-RMS coordinates."""

    if not math.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError("normalization_scale must be finite and positive")
    values = np.asarray(radii, dtype=np.float64)
    return values / normalization_scale


def circle_oracle_curve(
    scales: npt.ArrayLike,
    radius: float,
    *,
    quadrature_points: int = 65_536,
) -> dict[str, FloatArray]:
    """Population posterior and FLIPD terms for a query on one circle.

    At the query ``(R, 0)``, the posterior angle is von Mises with
    concentration ``kappa=R^2/sigma^2``.  We evaluate its first moment by a
    stable periodic quadrature instead of depending on a special-function
    package.  The returned response and bias sum exactly to the full readout.
    """

    sigma = np.asarray(scales, dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("scales must be finite and positive")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if (
        isinstance(quadrature_points, bool)
        or quadrature_points < 1_024
        or quadrature_points % 2
    ):
        raise ValueError("quadrature_points must be an even integer >= 1024")

    theta = np.linspace(-math.pi, math.pi, quadrature_points, endpoint=False)
    cosine = np.cos(theta)
    kappa = np.square(radius / sigma)
    posterior_cosine = np.empty_like(sigma)
    for index, concentration in enumerate(kappa):
        # Subtracting the maximum exponent (zero at theta=0) prevents overflow.
        weights = np.exp(concentration * (cosine - 1.0))
        posterior_cosine[index] = np.dot(weights, cosine) / weights.sum()

    response = kappa * (1.0 - np.square(posterior_cosine))
    bias = kappa * np.square(1.0 - posterior_cosine)
    return {
        "posterior_cosine": posterior_cosine,
        "response": response,
        "bias": bias,
        "full": response + bias,
    }


def hard_projection_cosine_curve(
    scales: npt.ArrayLike,
    radius: float,
    *,
    sample_count: int = 200_000,
    seed: int = 0,
) -> FloatArray:
    """Mean retained angular cosine after noise and nearest-circle projection."""

    sigma = np.asarray(scales, dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0 or np.any(sigma <= 0.0):
        raise ValueError("scales must be a non-empty positive array")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((sample_count, 2))
    retained = np.empty_like(sigma)
    for index, scale in enumerate(sigma):
        displaced_x = radius + scale * noise[:, 0]
        displaced_y = scale * noise[:, 1]
        retained[index] = np.mean(
            displaced_x / np.hypot(displaced_x, displaced_y)
        )
    return retained


def learned_diffusion_components(
    result: TrainingResult,
    query: npt.ArrayLike,
    sigma: float,
    *,
    batch_size: int,
) -> dict[str, FloatArray]:
    """Exact-trace FLIPD decomposition into the torus' coordinate pairs."""

    if result.family != "gaussian_diffusion":
        raise ValueError("the component audit requires a Gaussian diffusion model")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    query_array = np.asarray(query, dtype=np.float32)
    if query_array.ndim != 2 or query_array.shape[1] % 2:
        raise ValueError("query must have shape (n, 2k)")
    model = result.model
    if query_array.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match checkpoint")
    parameter = next(model.parameters())
    normalized = (
        torch.as_tensor(query_array)
        - result.normalization_mean.reshape(1, -1)
    ) / result.normalization_scale
    model.eval()

    score_batches: list[torch.Tensor] = []
    divergence_diagonal_batches: list[torch.Tensor] = []
    posterior_cosine_batches: list[torch.Tensor] = []
    center = (
        -result.normalization_mean.reshape(1, -1) / result.normalization_scale
    )
    for start in range(0, normalized.shape[0], batch_size):
        inputs = normalized[start : start + batch_size].to(
            device=parameter.device, dtype=parameter.dtype
        )
        inputs = inputs.detach().requires_grad_(True)
        condition = torch.full(
            (inputs.shape[0],), sigma, device=inputs.device, dtype=inputs.dtype
        )
        raw_denoiser = diffusion_denoiser_output(
            model,
            inputs,
            condition,
            preconditioning=result.config.diffusion_preconditioning,
            data_std=result.config.diffusion_data_std,
        )
        diagonal_terms: list[torch.Tensor] = []
        for coordinate in range(inputs.shape[1]):
            gradient = torch.autograd.grad(
                raw_denoiser[:, coordinate].sum(),
                inputs,
                retain_graph=coordinate + 1 < inputs.shape[1],
                create_graph=False,
            )[0]
            diagonal_terms.append(gradient[:, coordinate])
        variance = sigma**2
        score_batches.append(((raw_denoiser - inputs) / variance).detach().cpu())
        raw_diagonal = torch.stack(diagonal_terms, dim=1)
        divergence_diagonal_batches.append(
            ((raw_diagonal - 1.0) / variance).detach().cpu()
        )
        relative_inputs = inputs - center.to(device=inputs.device, dtype=inputs.dtype)
        relative_denoiser = raw_denoiser - center.to(
            device=inputs.device, dtype=inputs.dtype
        )
        posterior_cosines: list[torch.Tensor] = []
        for factor in range(inputs.shape[1] // 2):
            section = slice(2 * factor, 2 * factor + 2)
            denominator = relative_inputs[:, section].square().sum(dim=1)
            posterior_cosines.append(
                (relative_denoiser[:, section] * relative_inputs[:, section])
                .sum(dim=1)
                .div(denominator)
            )
        posterior_cosine_batches.append(
            torch.stack(posterior_cosines, dim=1).detach().cpu()
        )

    score = np.asarray(torch.cat(score_batches).numpy(), dtype=np.float64)
    divergence_diagonal = np.asarray(
        torch.cat(divergence_diagonal_batches).numpy(), dtype=np.float64
    )
    posterior_cosine = np.asarray(
        torch.cat(posterior_cosine_batches).numpy(), dtype=np.float64
    )
    factor_count = score.shape[1] // 2
    response = np.empty((score.shape[0], factor_count), dtype=np.float64)
    bias = np.empty_like(response)
    for factor in range(factor_count):
        section = slice(2 * factor, 2 * factor + 2)
        response[:, factor] = 2.0 + sigma**2 * np.sum(
            divergence_diagonal[:, section], axis=1
        )
        bias[:, factor] = sigma**2 * np.sum(np.square(score[:, section]), axis=1)
    full = response + bias
    total_from_public_readout = diffusion_flipd(
        score,
        np.sum(divergence_diagonal, axis=1),
        sigma=sigma,
        ambient_dim=score.shape[1],
    )
    if not np.allclose(np.sum(full, axis=1), total_from_public_readout, atol=1e-9):
        raise RuntimeError("coordinate decomposition disagrees with FLIPD readout")
    return {
        "response": response,
        "bias": bias,
        "full": full,
        "posterior_cosine": posterior_cosine,
    }


def _smooth_curve(values: FloatArray) -> FloatArray:
    if values.size < 5:
        return values.copy()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(padded, np.asarray([1, 4, 6, 4, 1]) / 16.0, mode="valid")


def log_slope_knee(
    scales: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    lower: float,
    upper: float,
    smooth: bool = False,
) -> tuple[float, float]:
    """Return the strongest downward derivative within a declared scale window."""

    sigma = np.asarray(scales, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if sigma.shape != curve.shape or sigma.ndim != 1:
        raise ValueError("scales and values must be matching one-dimensional arrays")
    if smooth:
        curve = _smooth_curve(curve)
    downward_slope = -np.gradient(curve, np.log(sigma))
    mask = (sigma >= lower) & (sigma <= upper)
    if not np.any(mask):
        raise ValueError("knee window contains no sampled scales")
    eligible = np.flatnonzero(mask)
    index = int(eligible[np.argmax(downward_slope[eligible])])
    return float(sigma[index]), float(downward_slope[index])


def half_retention_crossing(scales: npt.ArrayLike, retention: npt.ArrayLike) -> float:
    """Interpolate the first 0.5 crossing on a log-scale grid."""

    sigma = np.asarray(scales, dtype=np.float64)
    values = np.asarray(retention, dtype=np.float64)
    below = np.flatnonzero(values <= 0.5)
    if below.size == 0 or below[0] == 0:
        return math.nan
    right = int(below[0])
    left = right - 1
    fraction = (0.5 - values[left]) / (values[right] - values[left])
    log_crossing = np.log(sigma[left]) + fraction * (
        np.log(sigma[right]) - np.log(sigma[left])
    )
    return float(np.exp(log_crossing))


def _curve_statistics(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.5)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _write_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_history_csv(path: Path, result: TrainingResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_loss", "learning_rate"],
        )
        writer.writeheader()
        for metric in result.history:
            writer.writerow(metric.to_dict())


def _plot_results(
    path_stem: Path,
    scales: FloatArray,
    rows: list[dict[str, float]],
    radii: FloatArray,
    oracle_knees: list[float],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    total_oracle = np.asarray([row["oracle_full"] for row in rows])
    total_learned = np.asarray([row["learned_full_mean"] for row in rows])
    axes[0].plot(scales, total_oracle, color="black", linewidth=2.2, label="population total")
    axes[0].plot(scales, total_learned, color="#1f77b4", linewidth=2.0, label="learned total")
    q05 = np.asarray([row["learned_full_q05"] for row in rows])
    q95 = np.asarray([row["learned_full_q95"] for row in rows])
    axes[0].fill_between(scales, q05, q95, color="#1f77b4", alpha=0.16, linewidth=0)
    colors = ("#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    for factor, radius in enumerate(radii):
        color = colors[factor % len(colors)]
        axes[0].plot(
            scales,
            [row[f"oracle_factor_{factor}_full"] for row in rows],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"oracle factor {factor + 1} (R={radius:.3g})",
        )
        axes[0].plot(
            scales,
            [row[f"learned_factor_{factor}_full_mean"] for row in rows],
            color=color,
            linewidth=1.3,
            alpha=0.9,
            label=f"learned factor {factor + 1}",
        )
    axes[0].set_ylabel("FLIPD dimension")
    axes[0].set_ylim(-0.35, 2.45)
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)

    for factor, radius in enumerate(radii):
        color = colors[factor % len(colors)]
        axes[1].plot(
            scales,
            [row[f"posterior_factor_{factor}_cosine"] for row in rows],
            color=color,
            linewidth=1.8,
            label=f"posterior cosine, R={radius:.3g}",
        )
        axes[1].plot(
            scales,
            [row[f"learned_posterior_factor_{factor}_cosine_mean"] for row in rows],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"learned posterior, R={radius:.3g}",
        )
        axes[1].plot(
            scales,
            [row[f"projection_factor_{factor}_cosine"] for row in rows],
            color=color,
            linestyle=":",
            linewidth=1.8,
            label=f"hard projection, R={radius:.3g}",
        )
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    for knee in oracle_knees:
        for axis in axes:
            axis.axvline(knee, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("normalized VE noise sigma")
    axes[1].set_ylabel("embedded-coordinate retention")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".png"), dpi=180)
    figure.savefig(path_stem.with_suffix(".svg"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    progress_path = output_dir / "training-progress.pt"
    if args.evaluate_only:
        result = load_checkpoint(checkpoint_path, device=args.device)
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing checkpoint: {checkpoint_path}; "
                "use --evaluate-only or a fresh output directory"
            )
        train = sample_product_torus(args.train_size, args.radii, seed=args.seed + 101)
        validation = sample_product_torus(
            args.validation_size, args.radii, seed=args.seed + 202
        )
        config = TrainingConfig(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            time_embedding_dim=args.time_embedding_dim,
            validation_interval=1,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_norm=1.0,
            deterministic=True,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            diffusion_loss_weighting=args.diffusion_loss_weighting,
            diffusion_data_std=args.diffusion_data_std,
            diffusion_preconditioning=args.diffusion_preconditioning,
            fourier_features=args.fourier_features,
            max_condition_frequency=args.max_condition_frequency,
            dropout=0.0,
            normalize=True,
        )

        def report(metric: dict[str, float | int | bool | str]) -> None:
            epoch = int(metric["epoch"])
            if bool(metric.get("validated")) and (epoch == 1 or epoch % 10 == 0):
                print(
                    f"epoch={epoch:04d} train={float(metric['train_loss']):.6f} "
                    f"validation={float(metric['validation_loss']):.6f} "
                    f"best={float(metric['best_validation_loss']):.6f}",
                    flush=True,
                )

        result = train_model(
            "gaussian_diffusion",
            train,
            validation,
            config,
            checkpoint_path,
            report,
            progress_checkpoint_path=progress_path,
        )

    if result.model.config.ambient_dim != 2 * len(args.radii):
        raise ValueError("checkpoint dimension does not match the requested radii")
    if result.config.sigma_min is None or result.config.sigma_max is None:
        raise ValueError("checkpoint does not declare a VE scale interval")
    if (
        not math.isclose(result.config.sigma_min, args.sigma_min)
        or not math.isclose(result.config.sigma_max, args.sigma_max)
    ):
        raise ValueError("checkpoint VE interval does not match CLI arguments")

    radii = normalized_torus_radii(args.radii, result.normalization_scale)
    scales = np.geomspace(args.sigma_min, args.sigma_max, args.scale_count)
    queries = sample_product_torus(
        args.query_size, args.radii, seed=args.seed + 303
    )
    oracle_factors = [
        circle_oracle_curve(
            scales, float(radius), quadrature_points=args.quadrature_points
        )
        for radius in radii
    ]
    projection_factors = [
        hard_projection_cosine_curve(
            scales,
            float(radius),
            sample_count=args.projection_samples,
            seed=args.seed + 404 + factor,
        )
        for factor, radius in enumerate(radii)
    ]

    rows: list[dict[str, float]] = []
    learned_factor_means: list[list[float]] = [[] for _ in radii]
    learned_retention_means: list[list[float]] = [[] for _ in radii]
    start_time = time.monotonic()
    for scale_index, sigma in enumerate(scales):
        learned = learned_diffusion_components(
            result, queries, float(sigma), batch_size=args.evaluation_batch_size
        )
        total_full = np.sum(learned["full"], axis=1)
        total_response = np.sum(learned["response"], axis=1)
        total_bias = np.sum(learned["bias"], axis=1)
        row: dict[str, float] = {"sigma": float(sigma)}
        oracle_full = 0.0
        oracle_response = 0.0
        oracle_bias = 0.0
        for factor, oracle in enumerate(oracle_factors):
            for name in ("full", "response", "bias"):
                row[f"oracle_factor_{factor}_{name}"] = float(
                    oracle[name][scale_index]
                )
            row[f"posterior_factor_{factor}_cosine"] = float(
                oracle["posterior_cosine"][scale_index]
            )
            row[f"projection_factor_{factor}_cosine"] = float(
                projection_factors[factor][scale_index]
            )
            oracle_full += float(oracle["full"][scale_index])
            oracle_response += float(oracle["response"][scale_index])
            oracle_bias += float(oracle["bias"][scale_index])
            for name in ("full", "response", "bias"):
                statistics = _curve_statistics(learned[name][:, factor])
                for statistic, value in statistics.items():
                    row[f"learned_factor_{factor}_{name}_{statistic}"] = value
            learned_factor_means[factor].append(
                float(np.mean(learned["full"][:, factor]))
            )
            posterior_statistics = _curve_statistics(
                learned["posterior_cosine"][:, factor]
            )
            for statistic, value in posterior_statistics.items():
                row[
                    f"learned_posterior_factor_{factor}_cosine_{statistic}"
                ] = value
            learned_retention_means[factor].append(posterior_statistics["mean"])
        row["oracle_full"] = oracle_full
        row["oracle_response"] = oracle_response
        row["oracle_bias"] = oracle_bias
        for name, values in (
            ("full", total_full),
            ("response", total_response),
            ("bias", total_bias),
        ):
            for statistic, value in _curve_statistics(values).items():
                row[f"learned_{name}_{statistic}"] = value
        rows.append(row)
        if scale_index == 0 or (scale_index + 1) % 12 == 0:
            print(
                f"evaluated {scale_index + 1:03d}/{len(scales)} scales "
                f"in {time.monotonic() - start_time:.1f}s",
                flush=True,
            )

    oracle_knees: list[float] = []
    learned_knees: list[float] = []
    knee_records: list[dict[str, float | None]] = []
    for factor, radius in enumerate(radii):
        lower = max(args.sigma_min, 0.5 * float(radius))
        upper = min(args.sigma_max, 3.0 * float(radius))
        oracle_knee, oracle_slope = log_slope_knee(
            scales,
            oracle_factors[factor]["full"],
            lower=lower,
            upper=upper,
        )
        learned_knee, learned_slope = log_slope_knee(
            scales,
            learned_factor_means[factor],
            lower=lower,
            upper=upper,
            smooth=True,
        )
        posterior_crossing = half_retention_crossing(
            scales, oracle_factors[factor]["posterior_cosine"]
        )
        projection_crossing = half_retention_crossing(
            scales, projection_factors[factor]
        )
        learned_posterior_crossing = half_retention_crossing(
            scales, learned_retention_means[factor]
        )
        oracle_knees.append(oracle_knee)
        learned_knees.append(learned_knee)
        knee_records.append(
            {
                "radius": float(radius),
                "oracle_slope_knee": oracle_knee,
                "oracle_slope_knee_over_radius": oracle_knee / float(radius),
                "oracle_peak_downward_slope": oracle_slope,
                "learned_slope_knee": learned_knee,
                "learned_slope_knee_over_radius": learned_knee / float(radius),
                "learned_peak_downward_slope": learned_slope,
                "learned_to_oracle_knee_ratio": learned_knee / oracle_knee,
                "posterior_cosine_half_crossing": posterior_crossing,
                "posterior_crossing_over_radius": posterior_crossing / float(radius),
                "learned_posterior_cosine_half_crossing": (
                    learned_posterior_crossing
                    if math.isfinite(learned_posterior_crossing)
                    else None
                ),
                "learned_posterior_crossing_over_radius": (
                    learned_posterior_crossing / float(radius)
                    if math.isfinite(learned_posterior_crossing)
                    else None
                ),
                "projection_cosine_half_crossing": projection_crossing,
                "projection_crossing_over_radius": projection_crossing / float(radius),
            }
        )

    learned_total = np.asarray([row["learned_full_mean"] for row in rows])
    oracle_total = np.asarray([row["oracle_full"] for row in rows])
    probe_scales = {
        "low": 0.2 * float(radii[0]),
        "middle": float(math.sqrt(radii[0] * radii[-1])),
        "high": min(args.sigma_max, 4.0 * float(radii[-1])),
    }
    probes: dict[str, dict[str, float]] = {}
    for name, requested_scale in probe_scales.items():
        index = int(np.argmin(np.abs(np.log(scales) - math.log(requested_scale))))
        probes[name] = {
            "sigma": float(scales[index]),
            "oracle_full": float(oracle_total[index]),
            "learned_full_mean": float(learned_total[index]),
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ve_product_torus_geometric_knee",
        "hypothesis": (
            "FLIPD knees occur when Gaussian corruption changes an embedded "
            "coordinate at order-one scale"
        ),
        "raw_radii": [float(value) for value in args.radii],
        "normalized_radii": radii.tolist(),
        "normalization_scale": float(result.normalization_scale),
        "normalization_mean": result.normalization_mean.tolist(),
        "training_config": asdict(result.config),
        "training_metrics": result.metrics,
        "checkpoint_sha256": result.checkpoint_sha256,
        "query_count": args.query_size,
        "scale_count": args.scale_count,
        "quadrature_points": args.quadrature_points,
        "projection_samples": args.projection_samples,
        "knee_definition": (
            "maximum negative derivative with respect to log(sigma) in "
            "[0.5 R, 3 R]; learned curve uses a five-point binomial smoother"
        ),
        "knees": knee_records,
        "curve_rmse": float(np.sqrt(np.mean(np.square(learned_total - oracle_total)))),
        "curve_mae": float(np.mean(np.abs(learned_total - oracle_total))),
        "plateau_probes": probes,
    }
    _write_curve_csv(output_dir / "curves.csv", rows)
    _write_history_csv(output_dir / "training-history.csv", result)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        _plot_results(
            output_dir / "curves",
            scales,
            rows,
            radii,
            oracle_knees,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_902)
    parser.add_argument("--radii", type=float, nargs="+", default=[1.0, 16.0])
    parser.add_argument("--train-size", type=int, default=65_536)
    parser.add_argument("--validation-size", type=int, default=16_384)
    parser.add_argument("--query-size", type=int, default=2_048)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--fourier-features", type=int, default=32)
    parser.add_argument("--max-condition-frequency", type=float, default=100.0)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=16.0)
    parser.add_argument(
        "--diffusion-loss-weighting",
        choices=("sigma_squared_score", "edm_x0"),
        default="sigma_squared_score",
    )
    parser.add_argument("--diffusion-data-std", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-preconditioning",
        choices=("none", "edm"),
        default="none",
    )
    parser.add_argument("--scale-count", type=int, default=257)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--quadrature-points", type=int, default=65_536)
    parser.add_argument("--projection-samples", type=int, default=200_000)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
