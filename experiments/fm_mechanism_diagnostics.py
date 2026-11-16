"""Exact fixed-checkpoint geometry diagnostics for the time-boxed FM pilot.

Known-support interventions are diagnostic controls, not deployable estimators.
All spatial derivatives below are with respect to canonical r = Y / alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn


def alpha_at(lam: Tensor, schedule: str) -> Tensor:
    if schedule in {"log_noise", "lognoise", "log_noise_affine"}:
        return torch.ones_like(lam)
    if schedule in {"rectified", "rectified_linear"}:
        return 1 / (1 + lam)
    if schedule in {"vp", "vp_trigonometric"}:
        return torch.rsqrt(1 + lam.square())
    raise ValueError(schedule)


def canonical(field: nn.Module, r: Tensor, lam: float, schedule: str) -> Tensor:
    scale = r.new_full((len(r),), lam)
    return field(alpha_at(scale, schedule)[:, None] * r, torch.log(scale))


def value_and_jacobian(fn: Callable[[Tensor], Tensor], points: Tensor,
                       chunk: int = 32) -> tuple[Tensor, Tensor]:
    """Exact per-query Jacobians of a batch-independent field, no trace probes."""
    values, jacobians = [], []
    for batch in points.split(chunk):
        inputs = batch.detach().requires_grad_(True)
        outputs = fn(inputs)
        # Each vector-Jacobian product sums one output coordinate over queries;
        # model batch-separability makes these the individual query derivatives.
        n, dimension = outputs.shape
        jac = []
        for index in range(dimension):
            grad = torch.autograd.grad(outputs[:, index].sum(), inputs,
                                       retain_graph=index + 1 < dimension)[0]
            jac.append(grad.detach())
        values.append(outputs.detach())
        jacobians.append(torch.stack(jac, dim=1))
    return torch.cat(values), torch.cat(jacobians)


def components(value: Tensor, jacobian: Tensor, points: Tensor,
               basis: Tensor, intrinsic: int, lam: float) -> dict[str, Tensor]:
    """Express the complete Jacobian in an orthogonal tangent/normal basis."""
    rotated_jac = torch.einsum("ia,bij,jc->bac", basis, jacobian, basis)
    displacement = (value - points) @ basis
    tang = rotated_jac[:, :intrinsic, :intrinsic]
    normal = rotated_jac[:, intrinsic:, intrinsic:]
    tn = rotated_jac[:, :intrinsic, intrinsic:]
    nt = rotated_jac[:, intrinsic:, :intrinsic]
    tangent_trace = tang.diagonal(dim1=-2, dim2=-1).sum(-1)
    normal_trace = normal.diagonal(dim1=-2, dim2=-1).sum(-1)
    ct = displacement[:, :intrinsic].square().sum(-1) / lam**2
    cn = displacement[:, intrinsic:].square().sum(-1) / lam**2
    response = tangent_trace + normal_trace
    full = response + ct + cn
    result = {
        "response": response, "full": full,
        "tangent_trace": tangent_trace, "normal_trace": normal_trace,
        "correction_tangent": ct, "correction_normal": cn,
        "normal_jacobian_frobenius_sq": normal.square().sum((-2, -1)),
        "tangent_from_normal_frobenius_sq": tn.square().sum((-2, -1)),
        "normal_from_tangent_frobenius_sq": nt.square().sum((-2, -1)),
        "normal_asymmetry_sq": (normal - normal.transpose(-2, -1)).square().sum((-2, -1)),
        # At a clean affine-support query input projection leaves value fixed.
        "input_projected_response": tangent_trace,
        "input_projected_full": tangent_trace + ct + cn,
        "output_projected_response": tangent_trace,
        "output_projected_full": tangent_trace + ct,
        "centered_normal_full": response + ct,
    }
    if points.shape[-1] <= 256:
        result["normal_symmetric_eigenvalues"] = torch.linalg.eigvalsh(
            (normal + normal.transpose(-2, -1)) / 2
        )
    result["tangent_symmetric_eigenvalues"] = torch.linalg.eigvalsh(
        (tang + tang.transpose(-2, -1)) / 2
    )
    result["normal_displacement"] = displacement[:, intrinsic:]
    result["tangent_displacement"] = displacement[:, :intrinsic]
    return result


def gaussian_oracle(points: Tensor, basis: Tensor, intrinsic: int,
                    variance: float, lam: float) -> tuple[Tensor, Tensor]:
    tangent = basis[:, :intrinsic]
    projection = tangent @ tangent.T
    gain = variance / (variance + lam**2)
    return gain * (points @ projection), (gain * projection).expand(len(points), -1, -1)


def scalar_summary(values: dict[str, Tensor], truth: float | None) -> dict[str, float]:
    out = {}
    for key, arr in values.items():
        if arr.ndim == 1:
            out[key + "_mean"] = float(arr.mean())
            out[key + "_rms"] = float(arr.square().mean().sqrt())
            if truth is not None and (key.endswith("full") or key.endswith("response") or key in {"full", "response"}):
                out[key + "_mae_lid"] = float((arr - truth).abs().mean())
    return out


def verify_linear_control(device: str = "cpu") -> dict[str, float]:
    torch.manual_seed(1937)
    D, d, n = 8, 2, 5
    q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64, device=device))
    gains = torch.tensor([.9, .8, .01, .02, .03, -.01, .02, .01],
                         dtype=torch.float64, device=device)
    matrix = (q * gains) @ q.T
    offset = q[:, -1] * .07
    points = torch.randn(n, d, dtype=torch.float64, device=device) @ q[:, :d].T
    val, jac = value_and_jacobian(lambda r: r @ matrix.T + offset, points)
    comp = components(val, jac, points, q, d, .2)
    assert torch.allclose(jac, matrix.expand(n, -1, -1), atol=1e-12)
    assert torch.allclose(comp["normal_trace"], points.new_full((n,), .08), atol=1e-12)
    assert torch.allclose(comp["correction_normal"], points.new_full((n,), .07**2/.2**2), atol=1e-12)
    pt = q[:, :d] @ q[:, :d].T
    vin, jin = value_and_jacobian(lambda r: (r @ pt) @ matrix.T + offset, points)
    vout, jout = value_and_jacobian(lambda r: (r @ matrix.T + offset) @ pt, points)
    actual_in = jin.diagonal(dim1=-2, dim2=-1).sum(-1) + (vin-points).square().sum(-1)/.2**2
    actual_out = jout.diagonal(dim1=-2, dim2=-1).sum(-1) + (vout-points).square().sum(-1)/.2**2
    assert torch.allclose(actual_in, comp["input_projected_full"], atol=1e-12)
    assert torch.allclose(actual_out, comp["output_projected_full"], atol=1e-12)
    return {"max_jacobian_error": float((jac-matrix).abs().max()),
            "input_projection_identity_error": float((actual_in-comp["input_projected_full"]).abs().max()),
            "output_projection_identity_error": float((actual_out-comp["output_projected_full"]).abs().max())}


def audit_checkpoint(checkpoint: str | Path, *, device: str = "cuda:1",
                     query_count: int = 32, scales: tuple[float, ...] = (
                         .001, .00316227766, .01, .0316227766, .1, .177827941,
                         .316227766, .562341325, 1.0),
                     locations: tuple[str, ...] = ("clean", "noisy"),
                     output: str | Path | None = None) -> list[dict]:
    from experiments.fm_mechanism_pilot import load_pilot_checkpoint, canonical_posterior

    torch.set_num_threads(4)
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fixture = torch.load(checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    output = Path(output) if output else checkpoint.parent / f"diagnostics_{payload['step']:06d}"
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    clean = fixture["clean_queries"][:query_count].to(device=device, dtype=dtype)
    noise = fixture["query_noise"][:query_count].to(device=device, dtype=dtype)
    basis = torch.cat([fixture["tangent_basis"], fixture["normal_basis"]], dim=1).to(device=device, dtype=dtype)
    # Restore orthogonality at analysis precision, keeping the same tangent span.
    basis, _ = torch.linalg.qr(basis)
    intrinsic = int(payload["config"]["intrinsic_dim"])
    variance = float(fixture["latent_std"])**2
    if fixture.get("data_law", "gaussian") == "uniform":
        from experiments.fm_mechanism_oracles import AffineFixture
        population = AffineFixture("uniform", basis[:, :intrinsic],
                                   clean.new_zeros(clean.shape[1]),
                                   clean.new_full((intrinsic,), fixture["latent_half_width"]))
        def oracle_at(r, lam):
            moments = population.posterior(r, lam)
            return moments.mean, moments.jacobian(population.basis)
    else:
        def oracle_at(r, lam):
            return gaussian_oracle(r, basis, intrinsic, variance, lam)
    pt = basis[:, :intrinsic] @ basis[:, :intrinsic].T
    rows, arrays, head_rows = [], {}, []
    start = time.monotonic()
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(checkpoint, schedule, device)
        field = field.to(dtype=dtype)
        coordinate = payload["config"].get("input_coordinate", "native")
        output_coordinate = payload["config"].get("output_coordinate", "data")
        head = field.output[-1]
        normal_weight = basis[:, intrinsic:].T @ head.weight
        normal_bias = basis[:, intrinsic:].T @ head.bias
        head_rows.append({"schedule": schedule,
                          "normal_weight_frobenius": float(normal_weight.norm()),
                          "normal_bias_norm": float(normal_bias.norm()),
                          "all_weight_frobenius": float(head.weight.norm()),
                          "all_bias_norm": float(head.bias.norm())})
        for lam in scales:
            def fn(r):
                return canonical_posterior(field, r, lam, schedule,
                                           input_coordinate=coordinate,
                                           output_coordinate=output_coordinate)
            for location in locations:
                points = clean if location == "clean" else clean + lam * noise
                value, jac = value_and_jacobian(fn, points)
                comp = components(value, jac, points, basis, intrinsic, lam)
                oracle_value, oracle_jac = oracle_at(points, lam)
                oracle_comp = components(oracle_value, oracle_jac, points, basis, intrinsic, lam)
                row = {"step": payload["step"], "schedule": schedule, "lambda": lam,
                       "location": location, "query_count": query_count,
                       "scale_status": "in_training_range" if payload["config"]["lambda_min"] <= lam <= payload["config"]["lambda_max"] else "extrapolation"}
                row.update(scalar_summary(comp, intrinsic if location == "clean" else None))
                for name in ["response", "full", "tangent_trace", "normal_trace",
                             "correction_tangent", "correction_normal"]:
                    row["oracle_" + name + "_mean"] = float(oracle_comp[name].mean())
                    row[name + "_oracle_mae"] = float((comp[name] - oracle_comp[name]).abs().mean())
                row["posterior_oracle_weighted_mse"] = float((value-oracle_value).square().mean()/lam**2)
                row["jacobian_oracle_frobenius_sq"] = float((jac-oracle_jac).square().sum((-2,-1)).mean())
                if "normal_symmetric_eigenvalues" in comp:
                    eig = comp["normal_symmetric_eigenvalues"]
                    row["normal_eigenvalue_min"] = float(eig.min())
                    row["normal_eigenvalue_max"] = float(eig.max())
                    row["normal_eigenvalue_positive_sum_mean"] = float(eig.clamp_min(0).sum(-1).mean())
                    row["normal_eigenvalue_negative_sum_mean"] = float(eig.clamp_max(0).sum(-1).mean())
                # A scale-dependent constant correction, fit on separate clean
                # queries, preserves every derivative and tests global centering.
                if location == "clean":
                    with torch.no_grad():
                        calibration = fixture["clean_queries"][query_count:query_count+512].to(device=device, dtype=dtype)
                        cal_normal = (fn(calibration)-calibration) @ basis[:, intrinsic:]
                        offset = cal_normal.mean(0)
                        shifted_cn = (comp["normal_displacement"]-offset).square().sum(-1)/lam**2
                        shifted_full = comp["response"] + comp["correction_tangent"] + shifted_cn
                        row["constant_centered_full_mae_lid"] = float((shifted_full-intrinsic).abs().mean())
                        row["constant_centered_normal_correction_mean"] = float(shifted_cn.mean())
                        calibration_pred = fn(calibration)
                        calibration_oracle, cal_oracle_jac = oracle_at(calibration, lam)
                        pred_t = calibration_pred @ basis[:, :intrinsic]
                        oracle_t = calibration_oracle @ basis[:, :intrinsic]
                        clean_t = calibration @ basis[:, :intrinsic]
                        fitted_gain = (pred_t * clean_t).sum() / clean_t.square().sum()
                        correction_gain = (pred_t * oracle_t).sum() / pred_t.square().sum()
                        query_pred_t = value @ basis[:, :intrinsic]
                        query_clean_t = points @ basis[:, :intrinsic]
                        calibrated_ct = (correction_gain*query_pred_t-query_clean_t).square().sum(-1)/lam**2
                        calibrated_response = correction_gain*comp["tangent_trace"]+comp["normal_trace"]
                        calibrated_full = calibrated_response+calibrated_ct+comp["correction_normal"]
                        row["fitted_clean_tangent_gain"] = float(fitted_gain)
                        row["oracle_tangent_gain"] = float(cal_oracle_jac.diagonal(dim1=-2,dim2=-1).sum(-1).mean()/intrinsic)
                        row["tangent_scalar_calibration_factor"] = float(correction_gain)
                        row["scalar_calibrated_full_mae_lid"] = float((calibrated_full-intrinsic).abs().mean())
                        row["scalar_calibrated_response_mae_lid"] = float((calibrated_response-intrinsic).abs().mean())
                        row["scalar_calibrated_tangent_correction_mean"] = float(calibrated_ct.mean())
                        residual_t = pred_t-fitted_gain*clean_t
                        row["clean_tangent_radial_fit_residual_weighted_energy"] = float(residual_t.square().sum(-1).mean()/lam**2)
                rows.append(row)
                key = f"{schedule}__{lam:.9g}__{location}"
                for name, tensor in comp.items():
                    arrays[key + "__" + name] = tensor.detach().cpu().numpy()
                arrays[key + "__posterior"] = value.detach().cpu().numpy()
                if clean.shape[1] <= 256:
                    arrays[key + "__jacobian"] = jac.detach().cpu().numpy()
                else:
                    tangent_jac = torch.einsum("ia,bij,jc->bac", basis[:,:intrinsic], jac, basis[:,:intrinsic])
                    arrays[key + "__tangent_jacobian"] = tangent_jac.detach().cpu().numpy()
                arrays[key + "__oracle_posterior"] = oracle_value.detach().cpu().numpy()
                print(json.dumps({k: row[k] for k in ["step","schedule","lambda","location","response_mean","normal_trace_mean","correction_normal_mean","full_oracle_mae"]}), flush=True)
            # Incremental summary stays readable during the longer exact audit.
            keys = sorted({k for row in rows for k in row})
            with (output / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        del field
    arrays["basis"] = basis.cpu().numpy()
    arrays["clean"] = clean.cpu().numpy()
    arrays["noise"] = noise.cpu().numpy()
    np.savez_compressed(output / "query_arrays.npz", **arrays)
    (output / "head_geometry.json").write_text(json.dumps(head_rows, indent=2)+"\n")
    (output / "manifest.json").write_text(json.dumps({
        "checkpoint": checkpoint.name, "source_run": checkpoint.parent.name,
        "step": payload["step"], "query_count": query_count,
        "scales": scales, "locations": locations, "dtype": str(dtype),
        "elapsed_seconds": time.monotonic()-start,
        "derivatives": "exact autograd per output coordinate; no trace sampling",
        "large_dimension_storage": "For D>256: omit normal eigenspectrum and full Jacobian storage; exact traces and projected blocks retained",
        "interventions": "known affine support input/output projection; diagnostic only",
        "linear_control": verify_linear_control(),
    }, indent=2)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--scales", nargs="+", type=float)
    parser.add_argument("--locations", nargs="+", choices=["clean","noisy"], default=["clean","noisy"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        print(json.dumps(verify_linear_control(), indent=2))
    else:
        kwargs = {"scales": tuple(args.scales)} if args.scales else {}
        audit_checkpoint(args.checkpoint, device=args.device, query_count=args.queries,
                         locations=tuple(args.locations), output=args.output, **kwargs)


if __name__ == "__main__":
    main()
