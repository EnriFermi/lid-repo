"""Exact radial chain decomposition for the anchored-path posterior models.

No optimization, synthetic output residual, derivative hook, or changed model
is used. The spatial JVP through final LayerNorm is split algebraically into
numerator and denominator terms. Opposite-ray pairing removes even offsets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.fm_mechanism_diagnostics import alpha_at
from experiments.fm_mechanism_pilot import canonical_posterior
from experiments.fm_scheduler_cause_training import (
    ARM_SCHEDULE, evaluation_copy, load_cause_checkpoint, write_csv,
)


def pre_final_state(field, points, scale, schedule):
    lam = points.new_full((len(points),), scale)
    embedding = field.condition_embedding(lam.log())
    state = field.input(alpha_at(lam, schedule)[:, None] * points) + field.input_condition(embedding)
    for block in field.blocks:
        state = block(state, embedding)
    return state


def radial_chain(field, points, scale, schedule):
    """Return exact per-point vectors and scalar final-normalizer quantities."""
    if points.dtype != torch.float64 or field.input.weight.dtype != torch.float64:
        raise ValueError("Radial diagnostic requires float64")
    h, dh = torch.autograd.functional.jvp(
        lambda x: pre_final_state(field, x, scale, schedule), points, points)
    z, dz = h - h.mean(-1, keepdim=True), dh - dh.mean(-1, keepdim=True)
    layer = field.output[0]
    if isinstance(layer, torch.nn.LayerNorm):
        std = (z.square().mean(-1, keepdim=True) + layer.eps).sqrt()
        chi = (z * dz).mean(-1, keepdim=True) / std.square()
        normalized = layer(h)
        gamma = layer.weight if layer.weight is not None else torch.ones_like(h[0])
        numerator_pre = gamma * dz / std
        denominator_pre = -chi * gamma * z / std
    elif isinstance(layer, torch.nn.Identity):
        std = z.square().mean(-1, keepdim=True).sqrt()
        chi = (z * dz).mean(-1, keepdim=True) / std.square().clamp_min(1e-100)
        normalized, numerator_pre, denominator_pre = h, dh, torch.zeros_like(dh)
    else:
        raise TypeError("Expected final LayerNorm or Identity")
    sigmoid = normalized.sigmoid()
    slope = sigmoid * (1 + normalized * (1 - sigmoid))
    numerator = F.linear(slope * numerator_pre, field.output[-1].weight)
    denominator = F.linear(slope * denominator_pre, field.output[-1].weight)
    value = field.output(h)
    direct_value, direct_jvp = torch.autograd.functional.jvp(
        lambda x: canonical_posterior(field, x, scale, schedule), points, points)
    origin = canonical_posterior(field, torch.zeros_like(points[:1]), scale, schedule).detach()
    value_error = float((value - direct_value).abs().max())
    chain_error = float((numerator + denominator - direct_jvp).abs().max())
    if value_error > 1e-10 or chain_error > 1e-9:
        raise AssertionError((value_error, chain_error))
    return {"points": points.detach(), "value": value.detach(), "jvp": direct_jvp.detach(),
            "numerator": numerator.detach(), "denominator": denominator.detach(),
            "euler_raw": (direct_jvp - value).detach(),
            "euler_centered": (direct_jvp - value + origin).detach(),
            "numerator_euler_centered": (numerator - value + origin).detach(),
            "origin_value": origin, "pre_final_std": std[:, 0].detach(),
            "dlogstd_dlogradius": chi[:, 0].detach(),
            "value_max_error": value_error, "chain_max_error": chain_error}


def evaluate_points(field, points, scale, schedule, batch_size=32):
    parts = [radial_chain(field, part, scale, schedule) for part in points.split(batch_size)]
    result = {}
    for key in parts[0]:
        if key.endswith("max_error"):
            result[key] = max(p[key] for p in parts)
        elif key == "origin_value":
            result[key] = parts[0][key]
        else:
            result[key] = torch.cat([p[key] for p in parts])
    return result


def odd_components(positive, negative):
    """Negative-ray JVP uses direction -r; subtraction therefore gives J_odd*r."""
    return {key: (positive[key] - negative[key]) / 2 for key in
            ("value", "jvp", "numerator", "denominator", "euler_centered", "numerator_euler_centered")}


def squared_relative(vector, points):
    return float(vector.square().sum() / points.square().sum())


def summarize_components(components, points, basis):
    e = components["euler_centered"]
    numerator = components["numerator_euler_centered"]
    denominator = components["denominator"]
    tangent = (e @ basis) @ basis.T
    radial = ((e * points).sum(-1) / points.square().sum(-1))[:, None] * points
    e_energy = float(e.square().sum())
    result = {"euler_relative_energy": squared_relative(e, points),
              "radial_euler_relative_energy": squared_relative(radial, points),
              "sideways_tangent_euler_relative_energy": squared_relative(tangent - radial, points),
              "normal_euler_relative_energy": squared_relative(e - tangent, points),
              "numerator_relative_energy": squared_relative(numerator, points),
              "denominator_relative_energy": squared_relative(denominator, points),
              "twice_cross_relative_energy": float(2 * (numerator * denominator).sum() / points.square().sum()),
              "denominator_signed_projection_on_euler": float((e * denominator).sum()) / max(e_energy, 1e-100),
              "radial_gain": float((components["value"] * points).sum() / points.square().sum()),
              "radial_euler_coefficient": float((e * points).sum() / points.square().sum()),
              "numerator_radial_coefficient": float((numerator * points).sum() / points.square().sum()),
              "denominator_radial_coefficient": float((denominator * points).sum() / points.square().sum())}
    result["energy_identity_absolute_error"] = abs(result["euler_relative_energy"] -
        result["numerator_relative_energy"] - result["denominator_relative_energy"] - result["twice_cross_relative_energy"])
    return result


def run(source_root, output, *, device="cpu", directions=12, radius_count=5, heldout_count=128,
        scales=(.316227766, 1.), seed=20260906, step=9000, tailavg=True):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_root, output = Path(source_root), Path(output)
    fixture = torch.load(source_root / "data.pt", map_location="cpu", weights_only=False)
    if fixture["data_law"] != "gaussian":
        raise ValueError("First-pass radial attribution requires the affine Gaussian oracle")
    evaluation_data = torch.load(source_root / "evaluation_data.pt", map_location="cpu", weights_only=False)
    empirical = evaluation_data["queries"].double()
    probabilities = torch.linspace(.05, .95, radius_count, dtype=torch.float64)
    radii = torch.quantile(empirical.norm(dim=-1), probabilities)
    basis = torch.linalg.qr(fixture["tangent_basis"].double()).Q.to(device)
    generator = torch.Generator().manual_seed(seed)
    tangent_directions = torch.randn(directions, basis.shape[1], generator=generator, dtype=torch.float64)
    tangent_directions /= tangent_directions.norm(dim=-1, keepdim=True)
    ray_directions = tangent_directions.to(device) @ basis.T
    rays = (ray_directions[:, None] * radii.to(device)[None, :, None]).flatten(0, 1)
    heldout = empirical[:heldout_count].to(device)
    query_groups = {"rays": rays, "heldout": heldout}
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"radii": radii, "radius_quantiles": probabilities, "directions": ray_directions.cpu(),
              "rays": rays.cpu(), "heldout": heldout.cpu(), "basis": basis.cpu()}
    summary, ray_rows, pair_rows, checkpoints = [], [], [], {}
    maximum_error = 0.
    started = time.monotonic()
    odd_results = {}
    for arm in ARM_SCHEDULE:
        directory = arm + ("_tailavg" if tailavg else "")
        checkpoint = source_root / directory / f"checkpoint_step_{step:06d}.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        field, payload = load_cause_checkpoint(checkpoint, device)
        field = evaluation_copy(field, arm, payload["anchor"])
        for parameter in field.parameters():
            parameter.requires_grad_(False)
        checkpoints[arm] = {"path": str(checkpoint), "sha256": digest, "final_layernorm": isinstance(field.output[0], torch.nn.LayerNorm)}
        for scale in scales:
            for group, points in query_groups.items():
                positive = evaluate_points(field, points, scale, ARM_SCHEDULE[arm])
                negative = evaluate_points(field, -points, scale, ARM_SCHEDULE[arm])
                odd = odd_components(positive, negative)
                odd_results[arm, scale, group] = odd
                maximum_error = max(maximum_error, positive["chain_max_error"], negative["chain_max_error"])
                for parity, values in (("centered", positive), ("odd", odd)):
                    summary.append({"arm": arm, "lambda": scale, "group": group, "parity": parity,
                                    "count": len(points), **summarize_components(values, points, basis),
                                    "mean_pre_final_std": float(positive["pre_final_std"].mean()),
                                    "mean_dlogstd_dlogradius": float(positive["dlogstd_dlogradius"].mean())})
                prefix = f"{arm}/{scale}/{group}"
                for label, values in (("positive", positive), ("negative", negative), ("odd", odd)):
                    for name, value in values.items():
                        if torch.is_tensor(value):
                            arrays[f"{prefix}/{label}/{name}"] = value.cpu()
                if group == "rays":
                    for index in range(len(points)):
                        point = points[index]
                        norm2 = point.square().sum()
                        ray_rows.append({"arm": arm, "lambda": scale, "direction": index // radius_count,
                            "radius_index": index % radius_count, "radius": float(point.norm()),
                            "gain": float((positive["value"][index] * point).sum() / norm2),
                            "odd_gain": float((odd["value"][index] * point).sum() / norm2),
                            "odd_radial_euler": float((odd["euler_centered"][index] * point).sum() / norm2),
                            "pre_final_std": float(positive["pre_final_std"][index]),
                            "dlogstd_dlogradius": float(positive["dlogstd_dlogradius"][index])})
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        print(json.dumps({"arm": arm, "elapsed_seconds": time.monotonic() - started}), flush=True)
    for scale in scales:
        for group, points in query_groups.items():
            for left, right in (("rect", "vp_low"), ("rect_high", "vp")):
                delta = {name: odd_results[left, scale, group][name] - odd_results[right, scale, group][name]
                         for name in odd_results[left, scale, group]}
                pair_rows.append({"left": left, "right": right, "lambda": scale, "group": group,
                                  **summarize_components(delta, points, basis)})
    write_csv(output / "metrics.csv", summary)
    write_csv(output / "rays.csv", ray_rows)
    write_csv(output / "paired_euler.csv", pair_rows)
    torch.save(arrays, output / "arrays.pt")
    result = {"source_root": str(source_root), "checkpoints": checkpoints, "scales": list(scales),
              "directions": directions, "radii": radii.tolist(), "heldout_count": len(heldout),
              "maximum_chain_absolute_error": maximum_error,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - started,
              "scope": "Exact spatial chain decomposition, not training or a claim that either signed summand explains a percentage of historical error.",
              "identity": "Jb*r = numerator + denominator; Euler=Jb*r-b+b(0). Odd pairing removes all even offsets. Denominator signed projection may be negative or exceed one due to cancellation.",
              "no_final_LN": "When output normalization is Identity, denominator is exactly zero; pre-final std remains a descriptive statistic, not a normalizing operation."}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--directions", type=int, default=12)
    parser.add_argument("--radius-count", type=int, default=5)
    parser.add_argument("--heldout-count", type=int, default=128)
    parser.add_argument("--scales", nargs="+", type=float, default=[.316227766, 1.])
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
