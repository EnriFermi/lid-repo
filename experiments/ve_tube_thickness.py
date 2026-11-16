"""Measure whether one batched denoiser-gain probe predicts a lower LID knee.

The controlled distribution is an anisotropic Gaussian tube in ``R^D``:

    X_T ~ N(0, tau^2 I_d),     X_N ~ N(0, delta^2 I_(D-d)).

At the centre, its exact response/full Gaussian-channel readout is

    d_eff(sigma) = d tau^2/(tau^2 + sigma^2)
                 + (D-d) delta^2/(delta^2 + sigma^2).

For a normal direction, the Bayes denoiser gain is
``gamma=delta^2/(delta^2+sigma^2)``.  This script estimates ``delta`` using
centred finite differences whose plus/minus inputs, directions, and scales are
packed into one model call.  It then independently computes a dense exact-trace
curve at an analytic query solely to score the predicted lower knee.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

from models.readouts import diffusion_flipd
from models.training import (
    TrainingConfig,
    TrainingResult,
    diffusion_denoiser_output,
    load_checkpoint,
    predict_primitives,
    train_model,
)


FloatArray = npt.NDArray[np.float64]
Denoiser = Callable[[Tensor, Tensor], Tensor]


def sample_gaussian_tube(
    sample_count: int,
    *,
    ambient_dim: int,
    tangent_dim: int,
    tangent_std: float,
    normal_std: float,
    seed: int,
) -> npt.NDArray[np.float32]:
    """Sample paired antipodes so the empirical centre is exactly zero."""

    if isinstance(sample_count, bool) or sample_count <= 0 or sample_count % 2:
        raise ValueError("sample_count must be a positive even integer")
    if not 0 < tangent_dim < ambient_dim:
        raise ValueError("tangent_dim must lie strictly between zero and ambient_dim")
    if not math.isfinite(tangent_std) or tangent_std <= 0.0:
        raise ValueError("tangent_std must be finite and positive")
    if not math.isfinite(normal_std) or normal_std <= 0.0:
        raise ValueError("normal_std must be finite and positive")

    standard_deviations = np.concatenate(
        (
            np.full(tangent_dim, tangent_std, dtype=np.float64),
            np.full(ambient_dim - tangent_dim, normal_std, dtype=np.float64),
        )
    )
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((sample_count // 2, ambient_dim))
    base *= standard_deviations[None, :]
    samples = np.concatenate((base, -base), axis=0).astype(np.float32)
    rng.shuffle(samples, axis=0)
    return samples


def gaussian_tube_oracle_curve(
    scales: npt.ArrayLike,
    *,
    ambient_dim: int,
    tangent_dim: int,
    tangent_std: float,
    normal_std: float,
) -> dict[str, FloatArray]:
    """Return exact centre-query response contributions for a Gaussian tube."""

    sigma = np.asarray(scales, dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("scales must be finite and positive")
    if not 0 < tangent_dim < ambient_dim:
        raise ValueError("tangent_dim must lie strictly between zero and ambient_dim")
    if tangent_std <= 0.0 or normal_std <= 0.0:
        raise ValueError("tube standard deviations must be positive")
    tangent = tangent_dim * tangent_std**2 / (tangent_std**2 + sigma**2)
    normal = (ambient_dim - tangent_dim) * normal_std**2 / (
        normal_std**2 + sigma**2
    )
    return {"tangent": tangent, "normal": normal, "total": tangent + normal}


def centered_directional_gains(
    denoiser: Denoiser,
    center: Tensor,
    directions: Tensor,
    scales: npt.ArrayLike,
    *,
    amplitude_ratio: float,
) -> FloatArray:
    """Estimate all directional gains with exactly one batched model call."""

    sigma_numpy = np.asarray(scales, dtype=np.float64)
    if sigma_numpy.ndim != 1 or sigma_numpy.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.isfinite(sigma_numpy).all() or np.any(sigma_numpy <= 0.0):
        raise ValueError("scales must be finite and positive")
    if center.ndim != 1 or center.numel() == 0:
        raise ValueError("center must be a non-empty vector")
    if directions.ndim != 2 or directions.shape[1] != center.numel():
        raise ValueError("directions must have shape (count, ambient_dim)")
    if directions.shape[0] == 0 or not torch.isfinite(directions).all():
        raise ValueError("directions must be finite and non-empty")
    norms = torch.linalg.vector_norm(directions, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("directions must have unit norm")
    if not math.isfinite(amplitude_ratio) or amplitude_ratio <= 0.0:
        raise ValueError("amplitude_ratio must be finite and positive")

    sigma = torch.as_tensor(
        sigma_numpy, device=center.device, dtype=center.dtype
    )
    amplitudes = amplitude_ratio * sigma
    offsets = amplitudes[:, None, None] * directions[None, :, :]
    plus = center[None, None, :] + offsets
    minus = center[None, None, :] - offsets
    inputs = torch.stack((plus, minus), dim=2).reshape(-1, center.numel())
    conditions = sigma[:, None, None].expand(-1, directions.shape[0], 2).reshape(-1)
    with torch.no_grad():
        outputs = denoiser(inputs, conditions)
    if outputs.shape != inputs.shape or not torch.isfinite(outputs).all():
        raise ValueError("denoiser outputs must be finite and match its inputs")
    paired = outputs.reshape(sigma.shape[0], directions.shape[0], 2, center.numel())
    differences = paired[:, :, 0, :] - paired[:, :, 1, :]
    gains = torch.einsum("skd,kd->sk", differences, directions)
    gains = gains / (2.0 * amplitudes[:, None])
    return np.asarray(gains.detach().cpu().numpy(), dtype=np.float64)


def estimate_thickness_from_gains(
    scales: npt.ArrayLike,
    gains: npt.ArrayLike,
    *,
    minimum_gain: float = 0.05,
    maximum_gain: float = 0.95,
) -> dict[str, float | int | None]:
    """Fit one isotropic tube width from directional denoiser gains."""

    sigma = np.asarray(scales, dtype=np.float64)
    gamma = np.asarray(gains, dtype=np.float64)
    if sigma.ndim != 1 or gamma.ndim != 2 or gamma.shape[0] != sigma.size:
        raise ValueError("gains must have shape (len(scales), direction_count)")
    if not 0.0 < minimum_gain < maximum_gain < 1.0:
        raise ValueError("gain bounds must lie strictly inside (0, 1)")
    valid = (
        np.isfinite(gamma)
        & (gamma >= minimum_gain)
        & (gamma <= maximum_gain)
    )
    sigma_squared = np.square(sigma)[:, None]
    delta_squared = np.full_like(gamma, np.nan)
    delta_squared[valid] = (
        sigma_squared.repeat(gamma.shape[1], axis=1)[valid]
        * gamma[valid]
        / (1.0 - gamma[valid])
    )
    valid_values = delta_squared[np.isfinite(delta_squared) & (delta_squared > 0.0)]
    valid_count = int(valid_values.size)
    total_count = int(gamma.size)
    if valid_count == 0:
        return {
            "delta_hat": None,
            "delta_q10": None,
            "delta_q90": None,
            "valid_count": 0,
            "total_count": total_count,
            "valid_fraction": 0.0,
            "gain_fit_rmse": None,
            "inverse_curvature_slope": None,
            "inverse_curvature_intercept": None,
        }

    delta_hat = float(np.sqrt(np.median(valid_values)))
    predicted = delta_hat**2 / (delta_hat**2 + sigma_squared)
    fit_rmse = float(np.sqrt(np.mean(np.square(gamma[valid] - predicted.repeat(
        gamma.shape[1], axis=1
    )[valid]))))
    inverse_curvature = np.full_like(gamma, np.nan)
    inverse_curvature[valid] = sigma_squared.repeat(gamma.shape[1], axis=1)[valid] / (
        1.0 - gamma[valid]
    )
    regression_x = sigma_squared.repeat(gamma.shape[1], axis=1)[valid]
    regression_y = inverse_curvature[valid]
    if regression_x.size >= 2 and np.ptp(regression_x) > 0.0:
        design = np.column_stack((regression_x, np.ones_like(regression_x)))
        slope, intercept = np.linalg.lstsq(design, regression_y, rcond=None)[0]
        slope_value: float | None = float(slope)
        intercept_value: float | None = float(intercept)
    else:
        slope_value = None
        intercept_value = None
    delta_values = np.sqrt(valid_values)
    return {
        "delta_hat": delta_hat,
        "delta_q10": float(np.quantile(delta_values, 0.1)),
        "delta_q90": float(np.quantile(delta_values, 0.9)),
        "valid_count": valid_count,
        "total_count": total_count,
        "valid_fraction": valid_count / total_count,
        "gain_fit_rmse": fit_rmse,
        "inverse_curvature_slope": slope_value,
        "inverse_curvature_intercept": intercept_value,
    }


def _smooth_curve(values: FloatArray) -> FloatArray:
    if values.size < 5:
        return values.copy()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(
        padded, np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0, mode="valid"
    )


def strongest_downward_knee(
    scales: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    lower: float,
    upper: float,
    smooth: bool = True,
) -> dict[str, float | bool]:
    """Locate the strongest downward log-scale derivative in a fixed window."""

    sigma = np.asarray(scales, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if sigma.ndim != 1 or sigma.shape != curve.shape:
        raise ValueError("scales and values must be matching vectors")
    evaluated = _smooth_curve(curve) if smooth else curve
    slope = -np.gradient(evaluated, np.log(sigma))
    eligible = np.flatnonzero((sigma >= lower) & (sigma <= upper))
    if eligible.size < 3:
        raise ValueError("knee window must contain at least three scales")
    local = int(np.argmax(slope[eligible]))
    index = int(eligible[local])
    return {
        "sigma": float(sigma[index]),
        "slope": float(slope[index]),
        "boundary_hit": bool(local in {0, eligible.size - 1}),
    }


def downward_level_crossing(
    scales: npt.ArrayLike, values: npt.ArrayLike, *, level: float
) -> float | None:
    """Interpolate the first downward crossing on a log-spaced axis."""

    sigma = np.asarray(scales, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if sigma.ndim != 1 or sigma.shape != curve.shape:
        raise ValueError("scales and values must be matching vectors")
    below = np.flatnonzero(curve <= level)
    if below.size == 0 or below[0] == 0:
        return None
    right = int(below[0])
    left = right - 1
    fraction = (level - curve[left]) / (curve[right] - curve[left])
    return float(
        np.exp(
            np.log(sigma[left])
            + fraction * (np.log(sigma[right]) - np.log(sigma[left]))
        )
    )


def multiplicative_error(prediction: float | None, target: float) -> float | None:
    if prediction is None or prediction <= 0.0 or target <= 0.0:
        return None
    return float(math.exp(abs(math.log(prediction / target))))


def _width_token(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_or_load(
    args: argparse.Namespace, *, normal_std: float, output_dir: Path
) -> TrainingResult:
    checkpoint = output_dir / "model.pt"
    progress = output_dir / "training-progress.pt"
    if args.evaluate_only:
        return load_checkpoint(checkpoint, device=args.device)
    if checkpoint.exists():
        raise FileExistsError(
            f"refusing to overwrite {checkpoint}; use --evaluate-only or a fresh root"
        )
    train = sample_gaussian_tube(
        args.train_size,
        ambient_dim=args.ambient_dim,
        tangent_dim=args.tangent_dim,
        tangent_std=args.tangent_std,
        normal_std=normal_std,
        seed=args.seed + 101,
    )
    validation = sample_gaussian_tube(
        args.validation_size,
        ambient_dim=args.ambient_dim,
        tangent_dim=args.tangent_dim,
        tangent_std=args.tangent_std,
        normal_std=normal_std,
        seed=args.seed + 202,
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
        diffusion_loss_weighting="edm_x0",
        diffusion_data_std=1.0,
        diffusion_preconditioning="edm",
        fourier_features=args.fourier_features,
        max_condition_frequency=args.max_condition_frequency,
        dropout=0.0,
        normalize=True,
    )

    def report(metric: dict[str, float | int | bool | str]) -> None:
        epoch = int(metric["epoch"])
        if bool(metric.get("validated")) and (epoch == 1 or epoch % 20 == 0):
            print(
                f"normal_std={normal_std:g} epoch={epoch:04d} "
                f"train={float(metric['train_loss']):.6f} "
                f"validation={float(metric['validation_loss']):.6f} "
                f"best={float(metric['best_validation_loss']):.6f}",
                flush=True,
            )

    return train_model(
        "gaussian_diffusion",
        train,
        validation,
        config,
        checkpoint,
        report,
        progress_checkpoint_path=progress,
    )


def _evaluate_width(
    args: argparse.Namespace, *, normal_std: float, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _train_or_load(args, normal_std=normal_std, output_dir=output_dir)
    if result.model.config.ambient_dim != args.ambient_dim:
        raise ValueError("checkpoint ambient dimension does not match the experiment")

    normalized_tangent_std = args.tangent_std / result.normalization_scale
    normalized_normal_std = normal_std / result.normalization_scale

    parameter = next(result.model.parameters())
    center = (
        -result.normalization_mean.to(device=parameter.device, dtype=parameter.dtype)
        / result.normalization_scale
    )
    tangent_count = min(args.gain_tangent_directions, args.tangent_dim)
    normal_count = min(
        args.gain_normal_directions, args.ambient_dim - args.tangent_dim
    )
    direction_indices = list(range(tangent_count)) + list(
        range(args.tangent_dim, args.tangent_dim + normal_count)
    )
    directions = torch.eye(
        args.ambient_dim, device=parameter.device, dtype=parameter.dtype
    )[direction_indices]
    gain_scales = np.geomspace(
        args.gain_scale_min, args.gain_scale_max, args.gain_scale_count
    )

    def denoiser(inputs: Tensor, conditions: Tensor) -> Tensor:
        return diffusion_denoiser_output(
            result.model,
            inputs,
            conditions,
            preconditioning=result.config.diffusion_preconditioning,
            data_std=result.config.diffusion_data_std,
        )

    all_gains = centered_directional_gains(
        denoiser,
        center,
        directions,
        gain_scales,
        amplitude_ratio=args.gain_amplitude_ratio,
    )
    tangent_gains = all_gains[:, :tangent_count]
    normal_gains = all_gains[:, tangent_count:]
    thickness = estimate_thickness_from_gains(gain_scales, normal_gains)
    delta_hat_raw = thickness["delta_hat"]
    delta_hat = None if delta_hat_raw is None else float(delta_hat_raw)

    gain_rows: list[dict[str, Any]] = []
    for scale_index, sigma in enumerate(gain_scales):
        for local_index, coordinate in enumerate(direction_indices):
            kind = "tangent" if local_index < tangent_count else "normal"
            gamma = float(all_gains[scale_index, local_index])
            delta_directional: float | None = None
            if kind == "normal" and 0.0 < gamma < 1.0:
                delta_directional = float(
                    sigma * math.sqrt(gamma / (1.0 - gamma))
                )
            gain_rows.append(
                {
                    "sigma": float(sigma),
                    "kind": kind,
                    "coordinate": coordinate,
                    "gain": gamma,
                    "directional_delta": delta_directional,
                }
            )

    curve_scales = np.geomspace(
        args.curve_scale_min, args.curve_scale_max, args.curve_count
    )
    oracle = gaussian_tube_oracle_curve(
        curve_scales,
        ambient_dim=args.ambient_dim,
        tangent_dim=args.tangent_dim,
        tangent_std=normalized_tangent_std,
        normal_std=normalized_normal_std,
    )
    raw_query = np.zeros((1, args.ambient_dim), dtype=np.float32)
    curve_rows: list[dict[str, float]] = []
    learned_response: list[float] = []
    learned_full: list[float] = []
    evaluation_start = time.monotonic()
    for index, sigma in enumerate(curve_scales):
        prediction = predict_primitives(
            result,
            raw_query,
            float(sigma),
            divergence_backend="exact",
            trace_probes=0,
            batch_size=1,
        )
        response = args.ambient_dim + sigma**2 * float(prediction.divergence[0])
        full = float(
            diffusion_flipd(
                prediction.field,
                prediction.divergence,
                sigma=float(sigma),
                ambient_dim=args.ambient_dim,
            )[0]
        )
        learned_response.append(response)
        learned_full.append(full)
        curve_rows.append(
            {
                "sigma": float(sigma),
                "oracle_tangent": float(oracle["tangent"][index]),
                "oracle_normal": float(oracle["normal"][index]),
                "oracle_total": float(oracle["total"][index]),
                "learned_response": response,
                "learned_full": full,
            }
        )
    print(
        f"normal_std={normal_std:g} exact-trace curve evaluated in "
        f"{time.monotonic() - evaluation_start:.1f}s",
        flush=True,
    )

    learned_response_array = np.asarray(learned_response, dtype=np.float64)
    learned_full_array = np.asarray(learned_full, dtype=np.float64)
    oracle_total_knee = strongest_downward_knee(
        curve_scales,
        oracle["total"],
        lower=args.knee_scale_min,
        upper=args.knee_scale_max,
        smooth=False,
    )
    oracle_normal_knee = strongest_downward_knee(
        curve_scales,
        oracle["normal"],
        lower=args.knee_scale_min,
        upper=args.knee_scale_max,
        smooth=False,
    )
    learned_response_knee = strongest_downward_knee(
        curve_scales,
        learned_response_array,
        lower=args.knee_scale_min,
        upper=args.knee_scale_max,
        smooth=True,
    )
    learned_full_knee = strongest_downward_knee(
        curve_scales,
        learned_full_array,
        lower=args.knee_scale_min,
        upper=args.knee_scale_max,
        smooth=True,
    )
    response_half = downward_level_crossing(
        curve_scales,
        learned_response_array,
        level=0.5 * (args.ambient_dim + args.tangent_dim),
    )
    tangent_oracle_at_gain = args.tangent_dim * normalized_tangent_std**2 / (
        normalized_tangent_std**2 + np.square(gain_scales)
    )
    tangent_gain_oracle = tangent_oracle_at_gain / args.tangent_dim
    tangent_gain_rmse = float(
        np.sqrt(np.mean(np.square(tangent_gains - tangent_gain_oracle[:, None])))
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ve_gaussian_tube_one_forward_thickness",
        "raw_normal_std": normal_std,
        "normalized_normal_std": normalized_normal_std,
        "raw_tangent_std": args.tangent_std,
        "normalized_tangent_std": normalized_tangent_std,
        "ambient_dim": args.ambient_dim,
        "tangent_dim": args.tangent_dim,
        "normal_dim": args.ambient_dim - args.tangent_dim,
        "normalization_scale": result.normalization_scale,
        "checkpoint_sha256": result.checkpoint_sha256,
        "training_config": asdict(result.config),
        "training_metrics": result.metrics,
        "gain_probe": {
            "model_forward_calls": 1,
            "batched_examples": int(2 * len(direction_indices) * len(gain_scales)),
            "normal_directions": normal_count,
            "tangent_control_directions": tangent_count,
            "scale_min": float(gain_scales[0]),
            "scale_max": float(gain_scales[-1]),
            "scale_count": int(gain_scales.size),
            "amplitude_ratio": args.gain_amplitude_ratio,
            "thickness_fit": thickness,
            "tangent_gain_rmse": tangent_gain_rmse,
        },
        "fixed_knee_window": [args.knee_scale_min, args.knee_scale_max],
        "oracle_total_knee": oracle_total_knee,
        "oracle_normal_knee": oracle_normal_knee,
        "learned_response_knee": learned_response_knee,
        "learned_full_knee": learned_full_knee,
        "learned_response_half_crossing": response_half,
        "prediction_errors": {
            "vs_configured_width_factor": multiplicative_error(
                delta_hat, normalized_normal_std
            ),
            "vs_oracle_normal_knee_factor": multiplicative_error(
                delta_hat, float(oracle_normal_knee["sigma"])
            ),
            "vs_oracle_total_knee_factor": multiplicative_error(
                delta_hat, float(oracle_total_knee["sigma"])
            ),
            "vs_learned_response_knee_factor": multiplicative_error(
                delta_hat, float(learned_response_knee["sigma"])
            ),
            "vs_learned_full_knee_factor": multiplicative_error(
                delta_hat, float(learned_full_knee["sigma"])
            ),
            "vs_learned_response_half_factor": multiplicative_error(
                delta_hat, response_half if response_half is not None else -1.0
            ),
        },
        "curve_rmse": {
            "response_vs_oracle": float(
                np.sqrt(np.mean(np.square(learned_response_array - oracle["total"])))
            ),
            "full_vs_oracle": float(
                np.sqrt(np.mean(np.square(learned_full_array - oracle["total"])))
            ),
        },
    }
    _write_rows(output_dir / "curves.csv", curve_rows)
    _write_rows(output_dir / "gains.csv", gain_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _plot_aggregate(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(records)))
    configured: list[float] = []
    predicted: list[float] = []
    learned_knees: list[float] = []
    for color, record in zip(colors, records, strict=True):
        run_dir = output_dir / f"normal-std-{_width_token(record['raw_normal_std'])}"
        rows = np.genfromtxt(run_dir / "curves.csv", delimiter=",", names=True)
        label = f"delta={record['normalized_normal_std']:.3g}"
        axes[0].plot(
            rows["sigma"], rows["learned_response"], color=color, label=label
        )
        axes[0].plot(
            rows["sigma"], rows["oracle_total"], color=color, linestyle="--", alpha=0.7
        )
        delta_hat = record["gain_probe"]["thickness_fit"]["delta_hat"]
        if delta_hat is not None:
            axes[0].axvline(delta_hat, color=color, linewidth=0.9, alpha=0.7)
            configured.append(float(record["normalized_normal_std"]))
            predicted.append(float(delta_hat))
            learned_knees.append(float(record["learned_response_knee"]["sigma"]))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("normalized VE sigma")
    axes[0].set_ylabel("response LID at centre")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    if configured:
        values = np.asarray(configured + predicted + learned_knees)
        lower = float(np.min(values) / 1.25)
        upper = float(np.max(values) * 1.25)
        axes[1].plot([lower, upper], [lower, upper], color="black", linestyle=":")
        axes[1].scatter(configured, predicted, label="gain prediction", marker="o")
        axes[1].scatter(configured, learned_knees, label="learned curve knee", marker="x")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlim(lower, upper)
        axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("configured normalized thickness")
    axes[1].set_ylabel("predicted / measured scale")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "thickness_prediction.png", dpi=180)
    figure.savefig(output_dir / "thickness_prediction.svg")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "evaluate_only", "no_plot"}
    }
    records: list[dict[str, Any]] = []
    for normal_std in args.normal_stds:
        run_dir = output_dir / f"normal-std-{_width_token(normal_std)}"
        record = _evaluate_width(args, normal_std=normal_std, output_dir=run_dir)
        records.append(record)
        fit = record["gain_probe"]["thickness_fit"]
        print(
            f"normal_std={normal_std:g} normalized={record['normalized_normal_std']:.6f} "
            f"delta_hat={fit['delta_hat']} "
            f"response_knee={record['learned_response_knee']['sigma']:.6f} "
            f"factor={record['prediction_errors']['vs_learned_response_knee_factor']}",
            flush=True,
        )

    learned_errors = [
        record["prediction_errors"]["vs_learned_response_knee_factor"]
        for record in records
        if record["prediction_errors"]["vs_learned_response_knee_factor"] is not None
    ]
    configured_errors = [
        record["prediction_errors"]["vs_configured_width_factor"]
        for record in records
        if record["prediction_errors"]["vs_configured_width_factor"] is not None
    ]
    oracle_errors = [
        record["prediction_errors"]["vs_oracle_normal_knee_factor"]
        for record in records
        if record["prediction_errors"]["vs_oracle_normal_knee_factor"] is not None
    ]
    half_transition_errors = [
        record["prediction_errors"]["vs_learned_response_half_factor"]
        for record in records
        if record["prediction_errors"]["vs_learned_response_half_factor"] is not None
    ]
    successful_triplets = [
        (
            float(record["normalized_normal_std"]),
            float(record["gain_probe"]["thickness_fit"]["delta_hat"]),
            float(record["learned_response_knee"]["sigma"]),
        )
        for record in records
        if record["gain_probe"]["thickness_fit"]["delta_hat"] is not None
    ]
    configured = np.asarray(
        [triplet[0] for triplet in successful_triplets], dtype=np.float64
    )
    predicted = np.asarray(
        [triplet[1] for triplet in successful_triplets], dtype=np.float64
    )
    learned = np.asarray(
        [triplet[2] for triplet in successful_triplets], dtype=np.float64
    )
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ve_gaussian_tube_one_forward_thickness",
        "preregistered_primary_target": "learned_response_max_downward_log_slope",
        "preregistered_primary_error": "symmetric_multiplicative_factor",
        "resolved_experiment_arguments": resolved_arguments,
        "width_count": len(records),
        "records": records,
        "aggregate_metrics": {
            "prediction_success_count": len(learned_errors),
            "median_factor_vs_learned_response_knee": (
                float(np.median(learned_errors)) if learned_errors else None
            ),
            "maximum_factor_vs_learned_response_knee": (
                float(np.max(learned_errors)) if learned_errors else None
            ),
            "median_factor_vs_configured_width": (
                float(np.median(configured_errors)) if configured_errors else None
            ),
            "maximum_factor_vs_configured_width": (
                float(np.max(configured_errors)) if configured_errors else None
            ),
            "median_factor_vs_oracle_normal_knee": (
                float(np.median(oracle_errors)) if oracle_errors else None
            ),
            "maximum_factor_vs_oracle_normal_knee": (
                float(np.max(oracle_errors)) if oracle_errors else None
            ),
            "median_factor_vs_learned_response_half": (
                float(np.median(half_transition_errors))
                if half_transition_errors
                else None
            ),
            "maximum_factor_vs_learned_response_half": (
                float(np.max(half_transition_errors))
                if half_transition_errors
                else None
            ),
            "log_correlation_prediction_vs_configured_width": (
                float(np.corrcoef(np.log(configured), np.log(predicted))[0, 1])
                if predicted.size >= 2 and np.isfinite(predicted).all()
                else None
            ),
            "log_correlation_prediction_vs_learned_knee": (
                float(np.corrcoef(np.log(learned), np.log(predicted))[0, 1])
                if predicted.size >= 2 and np.isfinite(predicted).all()
                else None
            ),
        },
    }
    aggregate_rows: list[dict[str, Any]] = []
    for record in records:
        aggregate_rows.append(
            {
                "raw_normal_std": record["raw_normal_std"],
                "normalized_normal_std": record["normalized_normal_std"],
                "delta_hat": record["gain_probe"]["thickness_fit"]["delta_hat"],
                "gain_fit_rmse": record["gain_probe"]["thickness_fit"]["gain_fit_rmse"],
                "valid_gain_fraction": record["gain_probe"]["thickness_fit"][
                    "valid_fraction"
                ],
                "oracle_normal_knee": record["oracle_normal_knee"]["sigma"],
                "oracle_total_knee": record["oracle_total_knee"]["sigma"],
                "learned_response_knee": record["learned_response_knee"]["sigma"],
                "learned_full_knee": record["learned_full_knee"]["sigma"],
                "learned_response_half_crossing": record[
                    "learned_response_half_crossing"
                ],
                "factor_vs_configured_width": record["prediction_errors"][
                    "vs_configured_width_factor"
                ],
                "factor_vs_learned_response_knee": record["prediction_errors"][
                    "vs_learned_response_knee_factor"
                ],
            }
        )
    _write_rows(output_dir / "records.csv", aggregate_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        try:
            _plot_aggregate(output_dir, records)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            print(
                "matplotlib is unavailable; numerical artifacts were written "
                "without plots",
                flush=True,
            )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_903)
    parser.add_argument("--ambient-dim", type=int, default=16)
    parser.add_argument("--tangent-dim", type=int, default=2)
    parser.add_argument("--tangent-std", type=float, default=1.0)
    parser.add_argument(
        "--normal-stds", type=float, nargs="+", default=[0.03, 0.06, 0.12, 0.24]
    )
    parser.add_argument("--train-size", type=int, default=32_768)
    parser.add_argument("--validation-size", type=int, default=8_192)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--fourier-features", type=int, default=32)
    parser.add_argument("--max-condition-frequency", type=float, default=100.0)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=8.0)
    parser.add_argument("--curve-scale-min", type=float, default=0.01)
    parser.add_argument("--curve-scale-max", type=float, default=4.0)
    parser.add_argument("--curve-count", type=int, default=161)
    parser.add_argument("--knee-scale-min", type=float, default=0.02)
    parser.add_argument("--knee-scale-max", type=float, default=1.2)
    parser.add_argument("--gain-scale-min", type=float, default=0.025)
    parser.add_argument("--gain-scale-max", type=float, default=1.0)
    parser.add_argument("--gain-scale-count", type=int, default=9)
    parser.add_argument("--gain-amplitude-ratio", type=float, default=0.1)
    parser.add_argument("--gain-normal-directions", type=int, default=4)
    parser.add_argument("--gain-tangent-directions", type=int, default=2)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary["aggregate_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
