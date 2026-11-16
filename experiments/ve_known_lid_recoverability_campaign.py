"""Prospective five-manifold VE noise-recoverability pilot.

The pilot regenerates the coefficient-space geometries from the pinned
LID-Benchmarks source, trains one VE x0 denoiser per geometry, verifies held-out
denoising quality, and only then evaluates a frozen forward-only scale selector.

The automatic selector receives no LID labels, tangent/normal directions, or
input derivatives.  At each of ten fixed factor-two scales it evaluates the
antithetic statistic

    Q(sigma) = E <(z_hat(x+sigma z)-z_hat(x-sigma z))/2, z> / E ||z||^2,

where z_hat(y,sigma)=(y-m_theta(y,sigma))/sigma.  Its torus-pilot thresholds
are frozen in dimension units before any dataset in this campaign is scored.
Dense Q and exact-trace FLIPD curves are generated only as post-hoc diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import numpy.typing as npt

# PyTorch's deterministic CUDA contract requires this to be present before the
# first cuBLAS handle is created.  ``setdefault`` preserves an explicit launcher
# choice while making direct campaign invocations reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scipy.optimize import least_squares
from torch import Tensor

from experiments.ve_torus_noise_recoverability import (
    antithetic_noise_recoverability,
)
from models.training import (
    TrainingConfig,
    TrainingResult,
    diffusion_denoiser_output,
    load_checkpoint,
    predict_primitives,
    train_model,
)


FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

PINNED_UPSTREAM_REVISION = "2dcb8e41015f53413ff1ddd049bb006c81a5df52"
DEFAULT_DATASETS = (
    "e1_spiral_pca",
    "e6_exp_pca",
    "e7_crescent_moon_radius3.0",
    "e8_spaghetti_pca",
    "e8_sphere4_pca",
)
SELECTOR_ANCHORS = tuple(0.01 * 2.0**index for index in range(10))
BENCHMARK_VALIDATION_SCALES = tuple(0.01 * 2.0**index for index in range(7))
SELECTOR_FIT_RMSE_DIM_MAX = 0.08
SELECTOR_NEXT_RESIDUAL_DIM_MIN = 0.20
SELECTOR_SATURATION_FRACTION = 0.90


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    true_lid: int
    active_dim: int
    description: str
    upstream_function: str
    registry_target_note: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "e1_spiral_pca": DatasetSpec(
        name="e1_spiral_pca",
        true_lid=1,
        active_dim=2,
        description="rapidly coiling one-dimensional spiral in a two-coordinate plane",
        upstream_function="spiral_pca",
        registry_target_note="expected_lid=1",
    ),
    "e6_exp_pca": DatasetSpec(
        name="e6_exp_pca",
        true_lid=2,
        active_dim=3,
        description="two-dimensional exponentially narrowing funnel in three coordinates",
        upstream_function="exp_pca",
        registry_target_note="theoretical override 2; generated splits store 2",
    ),
    "e7_crescent_moon_radius3.0": DatasetSpec(
        name="e7_crescent_moon_radius3.0",
        true_lid=3,
        active_dim=3,
        description="three-dimensional bounded crescent volume with a curved boundary",
        upstream_function="crescent_moon_pca",
        registry_target_note="theoretical override 3; generated splits store 3",
    ),
    "e8_spaghetti_pca": DatasetSpec(
        name="e8_spaghetti_pca",
        true_lid=1,
        active_dim=20,
        description="one-dimensional trigonometric curve spanning twenty coordinates",
        upstream_function="spaghetti_pca",
        registry_target_note="theoretical override 1; upstream generator incorrectly stores 20",
    ),
    "e8_sphere4_pca": DatasetSpec(
        name="e8_sphere4_pca",
        true_lid=5,
        active_dim=6,
        description="four separated S5 components with radii 3, 1, 1/3, and 1/9",
        upstream_function="sphere4_pca",
        registry_target_note="artifact geometry uses dim=6 and expected_lid=5",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: npt.NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_benchmark_coefficients(
    name: str,
    sample_count: int,
    *,
    seed: int = 0,
    ambient_dim: int = 30,
) -> Float32Array:
    """Regenerate one pinned-upstream coefficient geometry.

    This deliberately returns only the official coefficient representation.
    The PCA image embedding is affine-isometric on these coordinates and is
    unnecessary for the present mechanism pilot.  ``RandomState`` matches the
    legacy ``np.random.seed`` generator used by the vendored source.
    """

    if name not in DATASET_SPECS:
        raise ValueError(f"unknown pilot dataset: {name}")
    if isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if ambient_dim != 30:
        raise ValueError("the pinned coefficient representation has dimension 30")
    rng = np.random.RandomState(seed)
    coefficients = np.zeros((sample_count, ambient_dim), dtype=np.float64)

    if name == "e1_spiral_pca":
        parameter = rng.uniform(1.0, 100.0, sample_count)
        radius = 1.0 / parameter
        phase = parameter**2
        coefficients[:, 0] = radius * np.sin(phase)
        coefficients[:, 1] = radius * np.cos(phase)
    elif name == "e6_exp_pca":
        active = rng.uniform(0.0, 8.0, size=(sample_count, 3))
        radius = 3.0 * np.exp(-active[:, 0])
        theta = rng.uniform(0.0, 2.0 * math.pi, sample_count)
        active[:, 0] -= 4.0
        active[:, 1] = radius * np.sin(theta)
        active[:, 2] = radius * np.cos(theta)
        coefficients[:, :3] = active
    elif name == "e7_crescent_moon_radius3.0":
        candidates = rng.uniform(-1.0, 1.0, size=(20 * sample_count, 3))
        mask = (
            np.square(candidates[:, 0]) + np.square(candidates[:, 1]) <= 1.0
        ) & (
            np.sqrt(
                np.square(candidates[:, 0])
                + np.square(candidates[:, 1] - 0.1)
            )
            >= 0.899
        )
        accepted = candidates[mask]
        if accepted.shape[0] < sample_count:
            raise RuntimeError("pinned crescent rejection sampler produced too few points")
        coefficients[:, :3] = 3.0 * accepted[:sample_count]
    elif name == "e8_spaghetti_pca":
        theta = rng.uniform(0.0, 2.0 * math.pi, sample_count)
        for coordinate in range(20):
            coefficients[:, coordinate] = np.sin((coordinate + 2) * theta)
    elif name == "e8_sphere4_pca":
        centers = np.asarray(
            ((3.0, 3.0), (-3.0, -3.0), (3.0, -3.0), (-3.0, 3.0)),
            dtype=np.float64,
        )
        for index in range(sample_count):
            component = int(rng.randint(4))
            radius = 3.0 / 3.0**component
            direction = rng.normal(size=6)
            direction /= np.linalg.norm(direction)
            direction *= radius
            direction[:2] += centers[component]
            coefficients[index, :6] = direction
    else:  # pragma: no cover - guarded above and exhaustive for type checkers.
        raise AssertionError(name)

    result = coefficients.astype(np.float32)
    if result.shape != (sample_count, ambient_dim) or not np.isfinite(result).all():
        raise RuntimeError("invalid generated coefficient array")
    return result


def make_splits(
    name: str,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    data_seed: int,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    all_samples = sample_benchmark_coefficients(
        name,
        train_size + validation_size + test_size,
        seed=data_seed,
    )
    validation_stop = train_size + validation_size
    return (
        np.ascontiguousarray(all_samples[:train_size]),
        np.ascontiguousarray(all_samples[train_size:validation_stop]),
        np.ascontiguousarray(all_samples[validation_stop:]),
    )


def selector_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "frozen_before_campaign_results": True,
        "frozen_at_utc_date": "2026-09-04",
        "source_pilot": "ve_torus_noise_recoverability/seed-20260904",
        "anchors": list(SELECTOR_ANCHORS),
        "anchor_policy": "ten fixed factor-two normalized VE scales",
        "input": "dataset-mean antithetic Q on held-out validation queries",
        "fit_model_dimension_units": "U(sigma)=A*sigma^2/(delta^2+sigma^2)",
        "fit_rmse_dimension_units_max": SELECTOR_FIT_RMSE_DIM_MAX,
        "next_transition_residual_dimension_units_min": (
            SELECTOR_NEXT_RESIDUAL_DIM_MIN
        ),
        "saturation_fraction": SELECTOR_SATURATION_FRACTION,
        "selection": "log-median of saturated anchors in first accepted prefix",
        "failure_policy": "abstain",
        "uses_lid_labels": False,
        "uses_geometry": False,
        "uses_input_derivatives": False,
        "logical_model_calls": len(SELECTOR_ANCHORS),
        "antithetic_inputs_per_query_probe_per_call": 2,
        "dimension_unit_conversion": (
            "The torus Q thresholds 0.02 and 0.05 at D=4 become 0.08 and "
            "0.20 after multiplication by D."
        ),
    }


def frozen_sparse_selector(
    scales: npt.ArrayLike,
    q_values: npt.ArrayLike,
    *,
    ambient_dim: int,
) -> dict[str, Any]:
    """Apply the predeclared first-transition rule in dimension units."""

    sigma = np.asarray(scales, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    if sigma.shape != (len(SELECTOR_ANCHORS),) or q.shape != sigma.shape:
        raise ValueError("selector requires exactly the ten frozen anchors")
    if not np.allclose(sigma, SELECTOR_ANCHORS, rtol=1.0e-12, atol=0.0):
        raise ValueError("selector scales differ from the frozen contract")
    if ambient_dim <= 0 or not np.isfinite(q).all():
        raise ValueError("invalid selector inputs")
    values = ambient_dim * q

    def fit_prefix(count: int) -> dict[str, Any]:
        fit_scales = sigma[:count]
        fit_values = values[:count]

        def residual(parameters: FloatArray) -> FloatArray:
            amplitude = parameters[0]
            delta = math.exp(parameters[1])
            prediction = amplitude * np.square(fit_scales) / (
                delta**2 + np.square(fit_scales)
            )
            return prediction - fit_values

        initial = np.asarray(
            [
                min(float(ambient_dim), max(float(np.max(fit_values)), 0.05)),
                math.log(float(fit_scales[0]) / 2.0),
            ],
            dtype=np.float64,
        )
        fitted = least_squares(
            residual,
            initial,
            bounds=(
                np.asarray(
                    [1.0e-6, math.log(float(fit_scales[0]) / 1_000.0)]
                ),
                np.asarray(
                    [float(ambient_dim), math.log(float(fit_scales[-1]) * 1_000.0)]
                ),
            ),
        )
        amplitude = float(fitted.x[0])
        delta = float(math.exp(fitted.x[1]))
        prediction = amplitude * np.square(fit_scales) / (
            delta**2 + np.square(fit_scales)
        )
        return {
            "prefix_count": count,
            "amplitude_dimension_units": amplitude,
            "delta": delta,
            "fit_rmse_dimension_units": float(
                np.sqrt(np.mean(np.square(prediction - fit_values)))
            ),
        }

    attempts: list[dict[str, Any]] = []
    for prefix_count in range(3, len(sigma)):
        fitted = fit_prefix(prefix_count)
        next_scale = float(sigma[prefix_count])
        next_observed = float(values[prefix_count])
        amplitude = float(fitted["amplitude_dimension_units"])
        delta = float(fitted["delta"])
        next_predicted = amplitude * next_scale**2 / (delta**2 + next_scale**2)
        next_residual = next_observed - next_predicted
        saturated_mask = (
            values[:prefix_count]
            >= SELECTOR_SATURATION_FRACTION * amplitude
        )
        saturated = sigma[:prefix_count][saturated_mask]
        attempt = {
            **fitted,
            "next_sigma": next_scale,
            "next_observed_dimension_units": next_observed,
            "next_predicted_dimension_units": next_predicted,
            "next_residual_dimension_units": next_residual,
            "saturated_anchor_count": int(saturated.size),
        }
        attempts.append(attempt)
        if (
            float(fitted["fit_rmse_dimension_units"])
            <= SELECTOR_FIT_RMSE_DIM_MAX
            and saturated.size >= 2
            and next_residual >= SELECTOR_NEXT_RESIDUAL_DIM_MIN
        ):
            return {
                "status": "selected",
                "selected_sigma": float(np.exp(np.median(np.log(saturated)))),
                "saturated_scales": saturated.tolist(),
                "decision_prefix_scale_count": prefix_count + 1,
                "fixed_scale_budget": len(sigma),
                "all_fixed_scales_evaluated": True,
                "fit": attempt,
                "attempts": attempts,
                "contract": selector_contract(),
            }
    return {
        "status": "abstained",
        "fixed_scale_budget": len(sigma),
        "all_fixed_scales_evaluated": True,
        "attempts": attempts,
        "contract": selector_contract(),
    }


def _normalise(result: TrainingResult, value: Float32Array) -> Tensor:
    parameter = next(result.model.parameters())
    raw = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
    mean = result.normalization_mean.to(
        device=parameter.device, dtype=parameter.dtype
    )
    return (raw - mean.reshape(1, -1)) / result.normalization_scale


def _denoise_batches(
    result: TrainingResult,
    noisy: Tensor,
    sigma: float,
    *,
    batch_size: int,
) -> Tensor:
    outputs: list[Tensor] = []
    result.model.eval()
    with torch.inference_mode():
        for start in range(0, noisy.shape[0], batch_size):
            batch = noisy[start : start + batch_size]
            condition = torch.full(
                (batch.shape[0],), sigma, device=batch.device, dtype=batch.dtype
            )
            outputs.append(
                diffusion_denoiser_output(
                    result.model,
                    batch,
                    condition,
                    preconditioning=result.config.diffusion_preconditioning,
                    data_std=result.config.diffusion_data_std,
                )
            )
    return torch.cat(outputs)


def evaluate_learning_quality(
    result: TrainingResult,
    train: Float32Array,
    test: Float32Array,
    spec: DatasetSpec,
    *,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Float32Array]]:
    """Held-out denoising gate, including the nonzero active coordinates."""

    scales = np.asarray(SELECTOR_ANCHORS, dtype=np.float64)
    clean = _normalise(result, test)
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(seed)
    rows: list[dict[str, float]] = []
    active = slice(0, spec.active_dim)
    inactive = slice(spec.active_dim, clean.shape[1])
    plot_payload: dict[str, Float32Array] = {}
    representative_sigma = 0.16

    for sigma in scales:
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        noisy = clean + float(sigma) * noise
        denoised = _denoise_batches(
            result, noisy, float(sigma), batch_size=batch_size
        )
        coefficient_skip = 1.0 / (1.0 + float(sigma) ** 2)
        baseline = coefficient_skip * noisy

        def mse(value: Tensor, section: slice) -> float:
            selected = value[:, section] - clean[:, section]
            if selected.shape[1] == 0:
                return 0.0
            return float(selected.square().mean().item())

        model_all = mse(denoised, slice(0, clean.shape[1]))
        baseline_all = mse(baseline, slice(0, clean.shape[1]))
        model_active = mse(denoised, active)
        baseline_active = mse(baseline, active)
        model_inactive = mse(denoised, inactive)
        baseline_inactive = mse(baseline, inactive)
        rows.append(
            {
                "sigma": float(sigma),
                "model_mse": model_all,
                "edm_zero_residual_mse": baseline_all,
                "model_to_baseline_ratio": model_all / baseline_all,
                "active_model_mse": model_active,
                "active_baseline_mse": baseline_active,
                "active_model_to_baseline_ratio": model_active / baseline_active,
                "inactive_model_mse": model_inactive,
                "inactive_baseline_mse": baseline_inactive,
                "inactive_model_to_baseline_ratio": (
                    model_inactive / baseline_inactive
                    if baseline_inactive > 0.0
                    else 0.0
                ),
            }
        )
        if math.isclose(float(sigma), representative_sigma):
            plot_payload = {
                "clean": np.asarray(clean.detach().cpu().numpy(), dtype=np.float32),
                "noisy": np.asarray(noisy.detach().cpu().numpy(), dtype=np.float32),
                "denoised": np.asarray(
                    denoised.detach().cpu().numpy(), dtype=np.float32
                ),
            }

    ratios = np.asarray([row["model_to_baseline_ratio"] for row in rows])
    active_ratios = np.asarray(
        [row["active_model_to_baseline_ratio"] for row in rows]
    )
    core = (scales >= 0.02) & (scales <= 1.28)
    weights = (np.square(scales) + 1.0) / np.square(scales)
    model_weighted = float(
        np.mean(weights * np.asarray([row["model_mse"] for row in rows]))
    )
    baseline_weighted = float(
        np.mean(
            weights
            * np.asarray([row["edm_zero_residual_mse"] for row in rows])
        )
    )
    active_model_weighted = float(
        np.mean(weights * np.asarray([row["active_model_mse"] for row in rows]))
    )
    active_baseline_weighted = float(
        np.mean(weights * np.asarray([row["active_baseline_mse"] for row in rows]))
    )
    train_validation_ratio = (
        float(result.metrics["final_validation_loss"])
        / float(result.metrics["final_train_loss"])
    )
    conditions = {
        "finite_training_metrics": all(
            math.isfinite(float(value)) for value in result.metrics.values()
        ),
        "heldout_weighted_ratio_at_most_0_90": (
            model_weighted / baseline_weighted <= 0.90
        ),
        "active_weighted_ratio_at_most_0_95": (
            active_model_weighted / active_baseline_weighted <= 0.95
        ),
        "active_geometry_improves_at_some_scale_by_10pct": (
            float(np.min(active_ratios[core])) <= 0.90
        ),
        "core_overall_improves_on_at_least_four_scales": (
            int(np.sum(ratios[core] <= 0.95)) >= 4
        ),
        "final_validation_to_train_ratio_at_most_1_50": (
            train_validation_ratio <= 1.50
        ),
    }
    report: dict[str, Any] = {
        "status": "passed" if all(conditions.values()) else "failed",
        "conditions": conditions,
        "training_metrics": result.metrics,
        "normalization_scale": float(result.normalization_scale),
        "normalization_mean_l2": float(
            torch.linalg.vector_norm(result.normalization_mean).item()
        ),
        "heldout_weighted_model_loss": model_weighted,
        "heldout_weighted_edm_zero_residual_loss": baseline_weighted,
        "heldout_weighted_ratio": model_weighted / baseline_weighted,
        "active_weighted_model_loss": active_model_weighted,
        "active_weighted_edm_zero_residual_loss": active_baseline_weighted,
        "active_weighted_ratio": active_model_weighted / active_baseline_weighted,
        "minimum_active_scale_ratio": float(np.min(active_ratios[core])),
        "core_overall_improved_scale_count": int(np.sum(ratios[core] <= 0.95)),
        "core_scale_count": int(np.sum(core)),
        "final_validation_to_train_ratio": train_validation_ratio,
        "representative_projection_sigma": representative_sigma,
        "test_examples": int(test.shape[0]),
        "active_dim_for_quality_gate": spec.active_dim,
        "note": (
            "Known active coordinates are used only to reject a model that learned "
            "the trivial zero padding but not the benchmark geometry. They are never "
            "provided to the scale selector."
        ),
        "train_array_sha256": _array_sha256(train),
        "test_array_sha256": _array_sha256(test),
    }
    return report, rows, plot_payload


def _curve_scales() -> FloatArray:
    dense = np.geomspace(0.01, 8.0, 49)
    return np.unique(
        np.concatenate((dense, np.asarray(SELECTOR_ANCHORS, dtype=np.float64)))
    )


def _summary_statistics(values: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mae_to_true_lid": math.nan,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "standard_error": float(np.std(values, ddof=1) / math.sqrt(values.size)),
    }


def _exact_flipd(
    result: TrainingResult,
    raw_queries: Float32Array,
    sigma: float,
    *,
    batch_size: int,
) -> dict[str, FloatArray]:
    primitive = predict_primitives(
        result,
        raw_queries,
        sigma,
        family="gaussian_diffusion",
        divergence_backend="exact",
        trace_probes=0,
        trace_seed=0,
        batch_size=batch_size,
    )
    ambient_dim = result.model.config.ambient_dim
    response = ambient_dim + sigma**2 * primitive.divergence
    correction = sigma**2 * np.einsum(
        "ij,ij->i", primitive.field, primitive.field
    )
    return {
        "response": np.asarray(response, dtype=np.float64),
        "correction": np.asarray(correction, dtype=np.float64),
        "full": np.asarray(response + correction, dtype=np.float64),
    }


def evaluate_recoverability_and_lid(
    result: TrainingResult,
    validation: Float32Array,
    test: Float32Array,
    spec: DatasetSpec,
    *,
    query_count: int,
    probe_count: int,
    query_seed: int,
    noise_seed: int,
    forward_batch_size: int,
    trace_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if query_count > len(validation) or query_count > len(test):
        raise ValueError("query_count exceeds a held-out split")
    query_rng = np.random.default_rng(query_seed)
    validation_indices = query_rng.choice(len(validation), query_count, replace=False)
    test_indices = query_rng.choice(len(test), query_count, replace=False)
    validation_query = np.ascontiguousarray(validation[validation_indices])
    test_query = np.ascontiguousarray(test[test_indices])
    normalized_validation = _normalise(result, validation_query)
    parameter = next(result.model.parameters())
    noise_numpy = np.random.default_rng(noise_seed).standard_normal(
        (query_count, probe_count, result.model.config.ambient_dim)
    ).astype(np.float32)
    noise = torch.as_tensor(
        noise_numpy, device=parameter.device, dtype=parameter.dtype
    )

    def denoiser(inputs: Tensor, conditions: Tensor) -> Tensor:
        return diffusion_denoiser_output(
            result.model,
            inputs,
            conditions,
            preconditioning=result.config.diffusion_preconditioning,
            data_std=result.config.diffusion_data_std,
        )

    scales = _curve_scales()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    total_forward_minibatches = 0
    for index, sigma in enumerate(scales):
        q_values, calls = antithetic_noise_recoverability(
            denoiser,
            normalized_validation,
            noise,
            float(sigma),
            batch_size=forward_batch_size,
            factor_width=result.model.config.ambient_dim,
        )
        total_forward_minibatches += calls
        validation_flipd = _exact_flipd(
            result,
            validation_query,
            float(sigma),
            batch_size=trace_batch_size,
        )
        test_flipd = _exact_flipd(
            result,
            test_query,
            float(sigma),
            batch_size=trace_batch_size,
        )
        q = q_values["total"]
        row: dict[str, Any] = {
            "sigma": float(sigma),
            "selector_anchor": any(
                math.isclose(float(sigma), value, rel_tol=1.0e-12, abs_tol=0.0)
                for value in SELECTOR_ANCHORS
            ),
            "q_mean": float(np.mean(q)),
            "q_median": float(np.median(q)),
            "q_q05": float(np.quantile(q, 0.05)),
            "q_q95": float(np.quantile(q, 0.95)),
            "q_standard_error": float(np.std(q, ddof=1) / math.sqrt(q.size)),
            "q_effective_dimension": float(
                result.model.config.ambient_dim * (1.0 - np.mean(q))
            ),
        }
        for split_name, components in (
            ("validation", validation_flipd),
            ("test", test_flipd),
        ):
            for component_name, component_values in components.items():
                statistics = _summary_statistics(component_values)
                statistics["mae_to_true_lid"] = float(
                    np.mean(np.abs(component_values - spec.true_lid))
                )
                for statistic_name, value in statistics.items():
                    row[
                        f"{split_name}_{component_name}_{statistic_name}"
                    ] = value
        rows.append(row)
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(scales):
            print(
                f"{spec.name}: evaluated {index + 1:02d}/{len(scales)} scales "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )

    anchor_rows = [row for row in rows if bool(row["selector_anchor"])]
    selector = frozen_sparse_selector(
        [float(row["sigma"]) for row in anchor_rows],
        [float(row["q_mean"]) for row in anchor_rows],
        ambient_dim=result.model.config.ambient_dim,
    )
    # Match the pre-existing learned benchmark grid in
    # configs/experiment/paper_learned_matrix.yaml.  The rest of the dense
    # curve is mechanism diagnostics, not an enlarged oracle search space.
    validation_candidates = [
        row
        for row in rows
        if any(
            math.isclose(
                float(row["sigma"]), candidate, rel_tol=1.0e-12, abs_tol=0.0
            )
            for candidate in BENCHMARK_VALIDATION_SCALES
        )
    ]
    if len(validation_candidates) != len(BENCHMARK_VALIDATION_SCALES):
        raise RuntimeError("diagnostic grid is missing benchmark validation scales")
    validation_best_row = min(
        validation_candidates,
        key=lambda row: float(row["validation_full_mae_to_true_lid"]),
    )
    validation_sigma = float(validation_best_row["sigma"])
    validation_test_row = min(
        rows,
        key=lambda row: abs(math.log(float(row["sigma"]) / validation_sigma)),
    )
    automatic: dict[str, Any]
    if selector["status"] == "selected":
        selected_sigma = float(selector["selected_sigma"])
        automatic_row = min(
            rows,
            key=lambda row: abs(math.log(float(row["sigma"]) / selected_sigma)),
        )
        if not math.isclose(
            float(automatic_row["sigma"]), selected_sigma, rel_tol=1.0e-12
        ):
            raise RuntimeError("selected sigma was absent from diagnostic grid")
        automatic = {
            "status": "selected",
            "sigma": selected_sigma,
            "test_full_mean": float(automatic_row["test_full_mean"]),
            "test_full_median": float(automatic_row["test_full_median"]),
            "test_full_mae": float(automatic_row["test_full_mae_to_true_lid"]),
            "test_response_mean": float(automatic_row["test_response_mean"]),
            "test_correction_mean": float(automatic_row["test_correction_mean"]),
        }
    else:
        automatic = {"status": "abstained"}

    effective = np.asarray(
        [float(row["q_effective_dimension"]) for row in rows], dtype=np.float64
    )
    response = np.asarray(
        [float(row["validation_response_mean"]) for row in rows], dtype=np.float64
    )
    diagnostic = {
        "schema_version": 1,
        "dataset": spec.name,
        "known_lid": spec.true_lid,
        "ambient_dim": result.model.config.ambient_dim,
        "validation_query_count": query_count,
        "test_query_count": query_count,
        "probe_count_per_validation_query": probe_count,
        "query_seed": query_seed,
        "noise_seed": noise_seed,
        "selector": selector,
        "automatic_selection": automatic,
        "labelled_validation_selection": {
            "protocol": "supervised validation MAE on frozen benchmark grid",
            "candidate_scales": list(BENCHMARK_VALIDATION_SCALES),
            "sigma": validation_sigma,
            "validation_full_mae": float(
                validation_best_row["validation_full_mae_to_true_lid"]
            ),
            "test_full_mean": float(validation_test_row["test_full_mean"]),
            "test_full_median": float(validation_test_row["test_full_median"]),
            "test_full_mae": float(
                validation_test_row["test_full_mae_to_true_lid"]
            ),
        },
        "dense_diagnostics": {
            "scale_count": len(rows),
            "uses_selector": False,
            "q_effective_vs_clean_response_pearson": float(
                np.corrcoef(effective, response)[0, 1]
            ),
            "q_effective_vs_clean_response_mae": float(
                np.mean(np.abs(effective - response))
            ),
            "q_effective_vs_clean_response_rmse": float(
                np.sqrt(np.mean(np.square(effective - response)))
            ),
        },
        "cost": {
            "automatic_selector_fixed_logical_scale_calls": len(
                SELECTOR_ANCHORS
            ),
            "automatic_selector_inputs_per_query_probe_per_call": 2,
            "all_dense_q_physical_minibatch_calls": total_forward_minibatches,
            "exact_trace_dense_curves_are_posthoc_only": True,
        },
    }
    return diagnostic, rows


def _selector_q_at_anchors(
    result: TrainingResult,
    raw_queries: Float32Array,
    *,
    probe_count: int,
    noise_seed: int,
    forward_batch_size: int,
) -> tuple[list[float], int]:
    normalized = _normalise(result, raw_queries)
    parameter = next(result.model.parameters())
    noise_numpy = np.random.default_rng(noise_seed).standard_normal(
        (len(raw_queries), probe_count, result.model.config.ambient_dim)
    ).astype(np.float32)
    noise = torch.as_tensor(
        noise_numpy, device=parameter.device, dtype=parameter.dtype
    )

    def denoiser(inputs: Tensor, conditions: Tensor) -> Tensor:
        return diffusion_denoiser_output(
            result.model,
            inputs,
            conditions,
            preconditioning=result.config.diffusion_preconditioning,
            data_std=result.config.diffusion_data_std,
        )

    means: list[float] = []
    calls = 0
    for sigma in SELECTOR_ANCHORS:
        values, scale_calls = antithetic_noise_recoverability(
            denoiser,
            normalized,
            noise,
            sigma,
            batch_size=forward_batch_size,
            factor_width=result.model.config.ambient_dim,
        )
        means.append(float(np.mean(values["total"])))
        calls += scale_calls
    return means, calls


def run_selector_sampling_audit(
    args: argparse.Namespace,
    name: str,
) -> dict[str, Any]:
    """Post-hoc query/probe sensitivity check without changing primary status."""

    output = args.output_dir.expanduser().resolve() / name
    checkpoint = output / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"missing trained checkpoint for sampling audit: {checkpoint}"
        )
    _, validation, _ = make_splits(
        name,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        data_seed=args.data_seed,
    )
    result = load_checkpoint(checkpoint, device=args.device)
    rng = np.random.default_rng(args.query_seed + 10_000)
    repeats: list[dict[str, Any]] = []
    for repeat in range(args.audit_repeats):
        indices = rng.choice(len(validation), args.query_count, replace=False)
        q, physical_calls = _selector_q_at_anchors(
            result,
            np.ascontiguousarray(validation[indices]),
            probe_count=args.probe_count,
            noise_seed=args.noise_seed + 10_000 + repeat,
            forward_batch_size=args.audit_forward_batch_size,
        )
        selection = frozen_sparse_selector(
            SELECTOR_ANCHORS, q, ambient_dim=result.model.config.ambient_dim
        )
        repeats.append(
            {
                "repeat": repeat,
                "query_count": args.query_count,
                "probe_count": args.probe_count,
                "q_means": q,
                "physical_minibatch_calls": physical_calls,
                "selection": selection,
            }
        )

    expanded_query_count = min(args.audit_query_count, len(validation))
    expanded_indices = rng.choice(len(validation), expanded_query_count, replace=False)
    expanded_q, expanded_calls = _selector_q_at_anchors(
        result,
        np.ascontiguousarray(validation[expanded_indices]),
        probe_count=args.audit_probe_count,
        noise_seed=args.noise_seed + 20_000,
        forward_batch_size=args.audit_forward_batch_size,
    )
    expanded_selection = frozen_sparse_selector(
        SELECTOR_ANCHORS,
        expanded_q,
        ambient_dim=result.model.config.ambient_dim,
    )
    audit = {
        "schema_version": 1,
        "dataset": name,
        "scope": (
            "posthoc sampling audit; does not alter the prospective primary "
            "selector result"
        ),
        "selector_contract": selector_contract(),
        "same_size_repeats": repeats,
        "same_size_selected_count": sum(
            item["selection"]["status"] == "selected" for item in repeats
        ),
        "same_size_abstained_count": sum(
            item["selection"]["status"] == "abstained" for item in repeats
        ),
        "expanded": {
            "query_count": expanded_query_count,
            "probe_count": args.audit_probe_count,
            "q_means": expanded_q,
            "physical_minibatch_calls": expanded_calls,
            "selection": expanded_selection,
        },
    }
    _write_json(output / "selector_sampling_audit.json", audit)
    print(
        f"{name}: sampling audit repeats selected="
        f"{audit['same_size_selected_count']}/{args.audit_repeats}; "
        f"expanded={expanded_selection['status']}",
        flush=True,
    )
    return audit


def _plot_training_history(path: Path, result: TrainingResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [item.epoch for item in result.history]
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    axis.plot(epochs, [item.train_loss for item in result.history], label="train")
    axis.plot(
        epochs,
        [item.validation_loss for item in result.history],
        label="validation",
    )
    axis.axvline(result.best_epoch, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("epoch")
    axis.set_ylabel("VE training objective")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_quality(path: Path, rows: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma = np.asarray([row["sigma"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    axis.plot(
        sigma,
        [row["model_to_baseline_ratio"] for row in rows],
        marker="o",
        label="all 30 coordinates",
    )
    axis.plot(
        sigma,
        [row["active_model_to_baseline_ratio"] for row in rows],
        marker="s",
        label="nonzero active coordinates",
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel("normalized VE sigma")
    axis.set_ylabel("held-out MSE / zero-residual EDM baseline")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_projection(
    path: Path,
    train: Float32Array,
    payload: dict[str, Float32Array],
    *,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fit_indices = rng.choice(len(train), min(len(train), 10_000), replace=False)
    fit = np.asarray(train[fit_indices], dtype=np.float64)
    mean = fit.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(fit - mean, full_matrices=False)
    basis = right[:2].T
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    for axis, (name, values) in zip(axes, payload.items(), strict=True):
        projected = (np.asarray(values, dtype=np.float64) - mean) @ basis
        axis.scatter(projected[:, 0], projected[:, 1], s=5, alpha=0.35)
        axis.set_title(name)
        axis.grid(alpha=0.15)
    figure.suptitle("Fixed-sigma held-out PCA projection")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_curves(
    path: Path,
    rows: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    *,
    true_lid: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma = np.asarray([float(row["sigma"]) for row in rows])
    effective = np.asarray([float(row["q_effective_dimension"]) for row in rows])
    response = np.asarray([float(row["validation_response_mean"]) for row in rows])
    full = np.asarray([float(row["validation_full_mean"]) for row in rows])
    anchors = np.asarray([bool(row["selector_anchor"]) for row in rows])
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    axis.plot(sigma, effective, linewidth=2.2, label="forward-only D(1-Q)")
    axis.scatter(sigma[anchors], effective[anchors], s=28, zorder=4, label="10 selector anchors")
    axis.plot(sigma, response, linewidth=1.7, label="exact FLIPD response R (diagnostic)")
    axis.plot(sigma, full, linewidth=1.5, alpha=0.8, label="exact full FLIPD F (diagnostic)")
    axis.axhline(true_lid, color="black", linestyle="--", linewidth=1.0, label="known LID")
    automatic = diagnostic["automatic_selection"]
    if automatic["status"] == "selected":
        axis.axvline(
            float(automatic["sigma"]), color="#059669", linewidth=1.6, label="automatic sigma"
        )
    axis.axvline(
        float(diagnostic["labelled_validation_selection"]["sigma"]),
        color="#dc2626",
        linestyle=":",
        linewidth=1.5,
        label="labelled-validation sigma",
    )
    axis.set_xscale("log")
    axis.set_xlabel("normalized VE sigma")
    axis.set_ylabel("dimension units")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        seed=args.model_seed,
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
        num_workers=0,
        deterministic=True,
        sigma_min=0.01,
        sigma_max=16.0,
        diffusion_loss_weighting="edm_x0",
        diffusion_data_std=1.0,
        diffusion_preconditioning="edm",
        fourier_features=args.fourier_features,
        max_condition_frequency=100.0,
        dropout=0.0,
        normalize=True,
    )


def _dataset_manifest(
    spec: DatasetSpec,
    train: Float32Array,
    validation: Float32Array,
    test: Float32Array,
    *,
    data_seed: int,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    generator_source = repository / "lid_benchmarks/generators/utils/pca.py"
    registry = repository / "configs/datasets/registry/paper_benchmarks.yaml"
    return {
        "schema_version": 1,
        "dataset": asdict(spec),
        "representation": "coefficients",
        "ambient_dim": 30,
        "identity": "regenerated-coefficients-from-pinned-upstream",
        "is_exact_archive_artifact": False,
        "pinned_upstream_revision": PINNED_UPSTREAM_REVISION,
        "data_seed": data_seed,
        "source_files": {
            "lid_benchmarks/generators/utils/pca.py": _sha256_file(generator_source),
            "configs/datasets/registry/paper_benchmarks.yaml": _sha256_file(registry),
        },
        "splits": {
            "train": {
                "shape": list(train.shape),
                "dtype": train.dtype.str,
                "sha256": _array_sha256(train),
            },
            "validation": {
                "shape": list(validation.shape),
                "dtype": validation.dtype.str,
                "sha256": _array_sha256(validation),
            },
            "test": {
                "shape": list(test.shape),
                "dtype": test.dtype.str,
                "sha256": _array_sha256(test),
            },
        },
        "nonzero_coordinate_count": int(
            np.sum(np.max(np.abs(train), axis=0) > 0.0)
        ),
        "note": (
            "The coefficient equations and split sizes match the pinned generator. "
            "The sealed 4.7 GB benchmark archive is not mounted, so this pilot must "
            "not be called exact-archive evidence."
        ),
    }


def run_dataset(args: argparse.Namespace, name: str) -> dict[str, Any]:
    spec = DATASET_SPECS[name]
    output = args.output_dir.expanduser().resolve() / name
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = make_splits(
        name,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        data_seed=args.data_seed,
    )
    manifest = _dataset_manifest(
        spec, train, validation, test, data_seed=args.data_seed
    )
    _write_json(output / "dataset_manifest.json", manifest)
    config = _training_config(args)
    _write_json(output / "training_config.json", asdict(config))
    _write_json(output / "selector_contract.json", selector_contract())
    checkpoint = output / "model.pt"
    progress = output / "training-progress.pt"

    if checkpoint.exists():
        result = load_checkpoint(checkpoint, device=args.device)
        if asdict(result.config) != asdict(config):
            raise ValueError(
                f"existing checkpoint config mismatch for {name}; use a fresh output root"
            )
    else:
        print(f"{name}: training on {args.device}", flush=True)

        def report(metric: dict[str, float | int | bool | str]) -> None:
            epoch = int(metric["epoch"])
            if bool(metric.get("validated")) and (epoch == 1 or epoch % 10 == 0):
                print(
                    f"{name}: epoch={epoch:04d} "
                    f"train={float(metric['train_loss']):.6f} "
                    f"validation={float(metric['validation_loss']):.6f} "
                    f"best={float(metric['best_validation_loss']):.6f}",
                    flush=True,
                )

        result = train_model(
            "gaussian_diffusion",
            train,
            validation,
            config,
            checkpoint,
            report,
            progress_checkpoint_path=progress,
        )
    _plot_training_history(output / "training_history.png", result)

    quality, quality_rows, projection = evaluate_learning_quality(
        result,
        train,
        test,
        spec,
        seed=args.quality_seed,
        batch_size=args.evaluation_batch_size,
    )
    _write_json(output / "learning_quality.json", quality)
    _write_csv(output / "learning_quality.csv", quality_rows)
    _plot_quality(output / "learning_quality.png", quality_rows)
    _plot_projection(
        output / "denoising_projection.png",
        train,
        projection,
        seed=args.quality_seed,
    )

    diagnostic: dict[str, Any] | None = None
    if quality["status"] == "passed" or args.evaluate_failed_models:
        diagnostic, curve_rows = evaluate_recoverability_and_lid(
            result,
            validation,
            test,
            spec,
            query_count=args.query_count,
            probe_count=args.probe_count,
            query_seed=args.query_seed,
            noise_seed=args.noise_seed,
            forward_batch_size=args.forward_batch_size,
            trace_batch_size=args.trace_batch_size,
        )
        diagnostic["interpretation_valid"] = quality["status"] == "passed"
        _write_json(output / "evaluation.json", diagnostic)
        _write_csv(output / "curves.csv", curve_rows)
        _plot_curves(
            output / "recoverability_and_flipd.png",
            curve_rows,
            diagnostic,
            true_lid=spec.true_lid,
        )

    summary = {
        "schema_version": 1,
        "dataset": name,
        "description": spec.description,
        "known_lid": spec.true_lid,
        "ambient_dim": 30,
        "active_dim": spec.active_dim,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": result.checkpoint_sha256,
        "dataset_manifest_sha256": _sha256_file(output / "dataset_manifest.json"),
        "training_config": asdict(result.config),
        "training_metrics": result.metrics,
        "learning_quality": quality,
        "evaluation": diagnostic,
        "status": (
            "complete" if diagnostic is not None and quality["status"] == "passed"
            else "invalid_failed_learning_gate"
        ),
    }
    _write_json(output / "summary.json", summary)
    print(
        f"{name}: {summary['status']} quality={quality['status']} "
        f"checkpoint={result.checkpoint_sha256[:12]}",
        flush=True,
    )
    return summary


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def aggregate_campaign(output_dir: Path, dataset_names: Iterable[str]) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in dataset_names:
        path = root / name / "summary.json"
        if not path.exists():
            missing.append(name)
            continue
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        evaluation = summary.get("evaluation")
        automatic = evaluation.get("automatic_selection", {}) if evaluation else {}
        labelled = (
            evaluation.get("labelled_validation_selection", {}) if evaluation else {}
        )
        sampling_audit_path = (
            root / summary["dataset"] / "selector_sampling_audit.json"
        )
        sampling_audit = (
            json.loads(sampling_audit_path.read_text(encoding="utf-8"))
            if sampling_audit_path.exists()
            else None
        )
        local_probe_path = root / summary["dataset"] / "local_probe_diagnostic.json"
        local_probe = (
            json.loads(local_probe_path.read_text(encoding="utf-8"))
            if local_probe_path.exists()
            else None
        )
        local_probe_record = None
        if local_probe:
            local_probe_record = min(
                local_probe["records"],
                key=lambda item: abs(float(item["probe_ratio"]) - 0.05),
            )
        rows.append(
            {
                "dataset": summary["dataset"],
                "known_lid": summary["known_lid"],
                "learning_gate": summary["learning_quality"]["status"],
                "heldout_baseline_ratio": summary["learning_quality"][
                    "heldout_weighted_ratio"
                ],
                "active_baseline_ratio": summary["learning_quality"][
                    "active_weighted_ratio"
                ],
                "selector_status": automatic.get("status", "not_evaluated"),
                "automatic_sigma": automatic.get("sigma"),
                "automatic_test_mae": automatic.get("test_full_mae"),
                "validation_sigma": labelled.get("sigma"),
                "validation_selected_test_mae": labelled.get("test_full_mae"),
                "q_response_pearson": (
                    evaluation["dense_diagnostics"][
                        "q_effective_vs_clean_response_pearson"
                    ]
                    if evaluation
                    else None
                ),
                "q_response_mae": (
                    evaluation["dense_diagnostics"][
                        "q_effective_vs_clean_response_mae"
                    ]
                    if evaluation
                    else None
                ),
                "sampling_repeat_selected_count": (
                    sampling_audit["same_size_selected_count"]
                    if sampling_audit
                    else None
                ),
                "sampling_repeat_count": (
                    len(sampling_audit["same_size_repeats"])
                    if sampling_audit
                    else None
                ),
                "expanded_selector_status": (
                    sampling_audit["expanded"]["selection"]["status"]
                    if sampling_audit
                    else None
                ),
                "local_probe_0_05_pearson": (
                    local_probe_record["pearson_to_exact_clean_response"]
                    if local_probe_record
                    else None
                ),
                "local_probe_0_05_mae": (
                    local_probe_record["mae_to_exact_clean_response"]
                    if local_probe_record
                    else None
                ),
                "local_probe_0_05_selector_status_posthoc": (
                    local_probe_record["selector_if_substituted_posthoc"]["status"]
                    if local_probe_record
                    else None
                ),
            }
        )
    if rows:
        _write_csv(root / "campaign_results.csv", rows)

    selected_rows = [
        row
        for row in rows
        if row["learning_gate"] == "passed"
        and row["selector_status"] == "selected"
    ]
    valid_rows = [row for row in rows if row["learning_gate"] == "passed"]
    aggregate = {
        "schema_version": 1,
        "experiment": "five-manifold-prospective-ve-recoverability-pilot",
        "datasets_requested": list(dataset_names),
        "datasets_complete": [summary["dataset"] for summary in summaries],
        "datasets_missing": missing,
        "selector_contract": selector_contract(),
        "learning_gate_pass_count": sum(
            row["learning_gate"] == "passed" for row in rows
        ),
        "selector_selected_count": len(selected_rows),
        "selector_abstained_count": sum(
            row["selector_status"] == "abstained" for row in valid_rows
        ),
        "sampling_audit_all_abstained": all(
            row["sampling_repeat_selected_count"] == 0
            and row["expanded_selector_status"] == "abstained"
            for row in valid_rows
        ),
        "posthoc_local_probe_selected_count": sum(
            row["local_probe_0_05_selector_status_posthoc"] == "selected"
            for row in valid_rows
        ),
        "automatic_test_mae_mean_on_selected": (
            float(np.mean([row["automatic_test_mae"] for row in selected_rows]))
            if selected_rows
            else None
        ),
        "validation_selected_test_mae_mean_on_same_selected": (
            float(
                np.mean(
                    [row["validation_selected_test_mae"] for row in selected_rows]
                )
            )
            if selected_rows
            else None
        ),
        "validation_selected_test_mae_mean_all_gate_passing": (
            float(
                np.mean(
                    [row["validation_selected_test_mae"] for row in valid_rows]
                )
            )
            if valid_rows
            else None
        ),
        "rows": rows,
        "claim_boundary": (
            "Single deterministic checkpoint per regenerated coefficient geometry; "
            "this tests prospective cross-geometry behavior and query/probe sampling, "
            "not retraining-seed uncertainty or sealed-archive parity."
        ),
    }
    _write_json(root / "campaign_summary.json", aggregate)

    if rows:
        _plot_campaign_overview(root / "campaign_overview.png", rows)

    lines = [
        "# Five-manifold VE recoverability pilot",
        "",
        "The scale selector was frozen before inspecting these five runs. Dense Q and",
        "exact-trace FLIPD curves are post-hoc diagnostics and are not selector inputs.",
        "The data are regenerated pinned-upstream coefficient geometries, not the absent",
        "sealed benchmark archive.",
        "",
        "| dataset | LID | train gate | selector | auto sigma | auto test MAE | labelled-val sigma | labelled-val test MAE | Q/R corr |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {known_lid} | {learning_gate} | {selector_status} | "
            "{automatic_sigma} | {automatic_test_mae} | {validation_sigma} | "
            "{validation_selected_test_mae} | {q_response_pearson} |".format(
                **{
                    **row,
                    "automatic_sigma": _format_number(row["automatic_sigma"]),
                    "automatic_test_mae": _format_number(
                        row["automatic_test_mae"]
                    ),
                    "validation_sigma": _format_number(row["validation_sigma"]),
                    "validation_selected_test_mae": _format_number(
                        row["validation_selected_test_mae"]
                    ),
                    "q_response_pearson": _format_number(
                        row["q_response_pearson"]
                    ),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Learning gates passed: {aggregate['learning_gate_pass_count']}/{len(rows)}.",
            f"- Automatic selections: {aggregate['selector_selected_count']}/{len(valid_rows)} gate-passing runs.",
            f"- Abstentions: {aggregate['selector_abstained_count']}/{len(valid_rows)} gate-passing runs.",
            "- Post-hoc sampling audit: all four same-size repeats and the "
            "1000-query/32-probe run abstained on every dataset."
            if aggregate["sampling_audit_all_abstained"]
            else "- Post-hoc sampling audit: at least one outcome changed.",
            f"- Post-hoc local probe (epsilon=0.05 sigma) would select "
            f"{aggregate['posthoc_local_probe_selected_count']}/{len(valid_rows)} runs; "
            "this is mechanism evidence, not a replacement primary result.",
            "- Automatic mean test MAE on selected runs: "
            + _format_number(aggregate["automatic_test_mae_mean_on_selected"])
            + ".",
            "- Labelled-validation mean test MAE on those same runs: "
            + _format_number(
                aggregate["validation_selected_test_mae_mean_on_same_selected"]
            )
            + ".",
            "- Labelled-validation mean test MAE across all gate-passing runs: "
            + _format_number(
                aggregate["validation_selected_test_mae_mean_all_gate_passing"]
            )
            + ".",
            "",
            "See each dataset directory for the learning gate, training history,",
            "PCA denoising check, exact diagnostic curves, checkpoint, and manifests.",
        ]
    )
    (root / "campaign_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _plot_campaign_overview(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        str(row["dataset"])
        .replace("_pca", "")
        .replace("e7_crescent_moon_radius3.0", "e7_crescent")
        for row in rows
    ]
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        [float(row["heldout_baseline_ratio"]) for row in rows],
        width,
        label="all coordinates",
    )
    axes[0].bar(
        positions + width / 2,
        [float(row["active_baseline_ratio"]) for row in rows],
        width,
        label="active coordinates",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("held-out MSE / baseline")
    axes[0].set_title("Diffusion learning gate")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        positions - width / 2,
        [float(row["q_response_pearson"]) for row in rows],
        width,
        color="#7c3aed",
        label="shell probe, epsilon=sigma",
    )
    axes[1].bar(
        positions + width / 2,
        [
            float(row["local_probe_0_05_pearson"])
            if row["local_probe_0_05_pearson"] is not None
            else math.nan
            for row in rows
        ],
        width,
        color="#059669",
        label="local probe, epsilon=0.05 sigma",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_title("Dense diagnostic: D(1-Q) vs R")
    axes[1].legend(fontsize=7)

    axes[2].bar(
        positions,
        [float(row["validation_selected_test_mae"]) for row in rows],
        color="#2563eb",
    )
    for position, row in zip(positions, rows, strict=True):
        axes[2].text(
            position,
            float(row["validation_selected_test_mae"]) + 0.02,
            "ABSTAIN" if row["selector_status"] == "abstained" else "SELECT",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[2].set_ylabel("test MAE")
    axes[2].set_title("Labelled-validation reference; auto status above")
    axes[2].set_ylim(
        0.0,
        max(float(row["validation_selected_test_mae"]) for row in rows) + 0.30,
    )

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_SPECS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=20_260_904)
    parser.add_argument("--quality-seed", type=int, default=20_260_905)
    parser.add_argument("--query-seed", type=int, default=20_260_906)
    parser.add_argument("--noise-seed", type=int, default=20_260_907)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--fourier-features", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8_192)
    parser.add_argument("--forward-batch-size", type=int, default=32_768)
    parser.add_argument("--trace-batch-size", type=int, default=128)
    parser.add_argument("--evaluate-failed-models", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--sampling-audit-only", action="store_true")
    parser.add_argument("--audit-repeats", type=int, default=4)
    parser.add_argument("--audit-query-count", type=int, default=1_000)
    parser.add_argument("--audit-probe-count", type=int, default=32)
    parser.add_argument("--audit-forward-batch-size", type=int, default=65_536)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    if args.sampling_audit_only:
        for name in args.datasets:
            run_selector_sampling_audit(args, name)
        return
    if not args.aggregate_only:
        for name in args.datasets:
            run_dataset(args, name)
    if args.aggregate_only or tuple(args.datasets) == DEFAULT_DATASETS:
        aggregate = aggregate_campaign(args.output_dir, args.datasets)
        print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
