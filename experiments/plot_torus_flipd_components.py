"""Plot the response--correction decomposition of a saved VE torus run.

This is a read-only post-processing diagnostic.  It consumes the per-scale CSV
written by ``ve_torus_knee.py`` and never loads or evaluates the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty curve file: {path}")
    return rows


def _column(rows: list[dict[str, str]], name: str) -> FloatArray:
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"missing column {name!r}") from error
    if not np.isfinite(values).all():
        raise ValueError(f"column {name!r} contains non-finite values")
    return values


def _smooth(values: FloatArray) -> FloatArray:
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    return np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")


def _window_peak(
    scales: FloatArray, values: FloatArray, *, radius: float
) -> tuple[int, float, float]:
    mask = (scales >= 0.5 * radius) & (scales <= 2.0 * radius)
    if not np.any(mask):
        raise ValueError(f"no scales in peak window for radius {radius}")
    candidates = np.flatnonzero(mask)
    index = int(candidates[np.argmax(values[candidates])])
    return index, float(scales[index]), float(values[index])


def _nearest_index(scales: FloatArray, target: float) -> int:
    return int(np.argmin(np.abs(np.log(scales / target))))


def run(curves_path: Path, summary_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_rows(curves_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    scales = _column(rows, "sigma")
    learned_full = _column(rows, "learned_full_mean")
    learned_response = _column(rows, "learned_response_mean")
    learned_correction = _column(rows, "learned_bias_mean")
    oracle_full = _column(rows, "oracle_full")
    oracle_response = _column(rows, "oracle_response")
    oracle_correction = _column(rows, "oracle_bias")

    learned_identity_error = float(
        np.max(np.abs(learned_full - learned_response - learned_correction))
    )
    oracle_identity_error = float(
        np.max(np.abs(oracle_full - oracle_response - oracle_correction))
    )
    if learned_identity_error > 1.0e-9 or oracle_identity_error > 1.0e-9:
        raise RuntimeError("saved curves violate full = response + correction")

    learned_full_smooth = _smooth(learned_full)
    learned_response_smooth = _smooth(learned_response)
    learned_correction_smooth = _smooth(learned_correction)
    log_scales = np.log(scales)
    downward_full = -np.gradient(learned_full_smooth, log_scales)
    downward_response = -np.gradient(learned_response_smooth, log_scales)
    downward_correction = -np.gradient(learned_correction_smooth, log_scales)
    slope_identity_error = float(
        np.max(np.abs(downward_full - downward_response - downward_correction))
    )

    radii = [float(value) for value in summary["normalized_radii"]]
    knee_records = summary["knees"]
    diagnostics: list[dict[str, float]] = []
    learned_correction_peaks: list[float] = []
    oracle_correction_peaks: list[float] = []
    learned_knees: list[float] = []
    for factor, radius in enumerate(radii):
        learned_factor_correction = _smooth(
            _column(rows, f"learned_factor_{factor}_bias_mean")
        )
        oracle_factor_correction = _column(rows, f"oracle_factor_{factor}_bias")
        learned_peak_index, learned_peak, learned_peak_value = _window_peak(
            scales, learned_factor_correction, radius=radius
        )
        _, oracle_peak, oracle_peak_value = _window_peak(
            scales, oracle_factor_correction, radius=radius
        )
        learned_knee = float(knee_records[factor]["learned_slope_knee"])
        knee_index = _nearest_index(scales, learned_knee)
        learned_correction_peaks.append(learned_peak)
        oracle_correction_peaks.append(oracle_peak)
        learned_knees.append(learned_knee)
        diagnostics.append(
            {
                "factor": factor + 1,
                "radius": radius,
                "learned_correction_peak_sigma": learned_peak,
                "learned_correction_peak_over_radius": learned_peak / radius,
                "learned_correction_peak_value": learned_peak_value,
                "oracle_correction_peak_sigma": oracle_peak,
                "oracle_correction_peak_over_radius": oracle_peak / radius,
                "oracle_correction_peak_value": oracle_peak_value,
                "learned_full_knee_sigma": learned_knee,
                "learned_full_knee_over_radius": learned_knee / radius,
                "correction_peak_to_full_knee_ratio": learned_peak / learned_knee,
                "learned_full_at_knee": float(learned_full[knee_index]),
                "learned_response_at_knee": float(learned_response[knee_index]),
                "learned_correction_at_knee": float(learned_correction[knee_index]),
                "downward_full_slope_at_knee": float(downward_full[knee_index]),
                "downward_response_slope_at_knee": float(
                    downward_response[knee_index]
                ),
                "downward_correction_slope_at_knee": float(
                    downward_correction[knee_index]
                ),
                "learned_correction_peak_grid_index": learned_peak_index,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary = {
        "schema_version": 1,
        "experiment": "ve_torus_flipd_response_correction_decomposition",
        "source_curves": str(curves_path),
        "source_summary": str(summary_path),
        "learned_identity_max_abs_error": learned_identity_error,
        "oracle_identity_max_abs_error": oracle_identity_error,
        "downward_slope_identity_max_abs_error": slope_identity_error,
        "annotation_scope": (
            "Curves are model-only. Reported correction peaks and FLIPD knees "
            "are post-hoc diagnostics using the known factor radii."
        ),
        "factors": diagnostics,
    }
    (output_dir / "component_diagnostics.json").write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True)
    full_q05 = _column(rows, "learned_full_q05")
    full_q95 = _column(rows, "learned_full_q95")
    response_q05 = _column(rows, "learned_response_q05")
    response_q95 = _column(rows, "learned_response_q95")
    correction_q05 = _column(rows, "learned_bias_q05")
    correction_q95 = _column(rows, "learned_bias_q95")

    model_only_figure, model_only_axes = plt.subplots(
        2, 1, figsize=(9.2, 7.4), sharex=True
    )
    model_only_axes[0].plot(
        scales,
        learned_full,
        color="#2563eb",
        linewidth=2.2,
        label="learned FLIPD F",
    )
    model_only_axes[0].fill_between(
        scales,
        full_q05,
        full_q95,
        color="#2563eb",
        alpha=0.14,
        linewidth=0,
        label="query 5--95%",
    )
    model_only_axes[0].set_ylabel("dimension contribution")
    model_only_axes[0].set_title(
        "Model-only curves: no manifold directions, radii, or LID labels"
    )
    model_only_axes[0].grid(alpha=0.22)
    model_only_axes[0].legend(fontsize=8)

    model_only_axes[1].plot(
        scales,
        learned_response,
        color="#0891b2",
        linewidth=2.0,
        label="response R",
    )
    model_only_axes[1].fill_between(
        scales,
        response_q05,
        response_q95,
        color="#0891b2",
        alpha=0.10,
        linewidth=0,
    )
    model_only_axes[1].plot(
        scales,
        learned_correction,
        color="#ea580c",
        linewidth=2.2,
        label="correction C",
    )
    model_only_axes[1].fill_between(
        scales,
        correction_q05,
        correction_q95,
        color="#ea580c",
        alpha=0.12,
        linewidth=0,
    )
    model_only_axes[1].set_xscale("log")
    model_only_axes[1].set_xlabel("normalized VE noise sigma")
    model_only_axes[1].set_ylabel("dimension contribution")
    model_only_axes[1].grid(alpha=0.22)
    model_only_axes[1].legend(fontsize=8)
    model_only_figure.tight_layout()
    model_only_figure.savefig(output_dir / "model_only_components.png", dpi=200)
    model_only_figure.savefig(output_dir / "model_only_components.svg")
    plt.close(model_only_figure)

    axes[0].plot(scales, learned_full, color="#2563eb", linewidth=2.2, label="learned FLIPD F")
    axes[0].fill_between(scales, full_q05, full_q95, color="#2563eb", alpha=0.14, linewidth=0)
    axes[0].plot(scales, oracle_full, color="black", linestyle="--", linewidth=1.7, label="population FLIPD F")
    axes[0].axhline(2.0, color="#64748b", linestyle=":", linewidth=1.0)
    axes[0].axhline(1.0, color="#64748b", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("dimension contribution")
    axes[0].set_title("Post-hoc population alignment of the model-only curves")
    axes[0].grid(alpha=0.22)

    axes[1].plot(scales, learned_response, color="#0891b2", linewidth=2.0, label="learned response R")
    axes[1].fill_between(scales, response_q05, response_q95, color="#0891b2", alpha=0.10, linewidth=0)
    axes[1].plot(scales, learned_correction, color="#ea580c", linewidth=2.2, label="learned correction C")
    axes[1].fill_between(scales, correction_q05, correction_q95, color="#ea580c", alpha=0.12, linewidth=0)
    axes[1].plot(scales, oracle_response, color="#0891b2", linestyle="--", linewidth=1.3, alpha=0.8, label="population R")
    axes[1].plot(scales, oracle_correction, color="#ea580c", linestyle="--", linewidth=1.3, alpha=0.8, label="population C")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("normalized VE noise sigma")
    axes[1].set_ylabel("dimension contribution")
    axes[1].grid(alpha=0.22)

    for factor, (correction_peak, knee) in enumerate(
        zip(learned_correction_peaks, learned_knees)
    ):
        for axis in axes:
            axis.axvline(
                correction_peak,
                color="#ea580c",
                linestyle=":",
                linewidth=1.1,
                alpha=0.9,
                label="learned C peak" if factor == 0 else None,
            )
            axis.axvline(
                knee,
                color="#334155",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="radius-windowed FLIPD knee" if factor == 0 else None,
            )

    axes[0].legend(ncol=2, fontsize=8)
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "flipd_components.png", dpi=200)
    figure.savefig(output_dir / "flipd_components.svg")
    plt.close(figure)

    slope_figure, slope_axis = plt.subplots(figsize=(9.2, 4.8))
    slope_axis.plot(scales, downward_full, color="#2563eb", linewidth=2.2, label="-dF / d log sigma")
    slope_axis.plot(scales, downward_response, color="#0891b2", linewidth=1.8, label="-dR / d log sigma")
    slope_axis.plot(scales, downward_correction, color="#ea580c", linewidth=1.8, label="-dC / d log sigma")
    slope_axis.axhline(0.0, color="#64748b", linewidth=0.8)
    for factor, (correction_peak, knee) in enumerate(
        zip(learned_correction_peaks, learned_knees)
    ):
        slope_axis.axvline(
            correction_peak,
            color="#ea580c",
            linestyle=":",
            linewidth=1.1,
            label="learned C peak" if factor == 0 else None,
        )
        slope_axis.axvline(
            knee,
            color="#334155",
            linestyle="--",
            linewidth=1.0,
            label="radius-windowed FLIPD knee" if factor == 0 else None,
        )
    slope_axis.set_xscale("log")
    slope_axis.set_xlabel("normalized VE noise sigma")
    slope_axis.set_ylabel("downward contribution per log sigma")
    slope_axis.set_title("Post-hoc slope decomposition at the two geometric transitions")
    slope_axis.grid(alpha=0.22)
    slope_axis.legend(ncol=2, fontsize=8)
    slope_figure.tight_layout()
    slope_figure.savefig(output_dir / "component_slopes.png", dpi=200)
    slope_figure.savefig(output_dir / "component_slopes.svg")
    plt.close(slope_figure)

    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.curves, args.summary, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
