"""Typical-support check of the pre-LN residual network's leading-ray limit.

This is inference only. Exact hidden-state accounting is distinguished from
its asymptotic approximation, and additive conditioning tests are labeled
fixed-input forward comparisons, not explanations of training history.
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
    ARM_SCHEDULE, arm_multiplier, evaluation_copy, load_cause_checkpoint, write_csv,
)


def center(x):
    return x - x.mean(-1, keepdim=True)


def hidden_parts(field, points, scale, schedule, *, condition_scale=None):
    scales = points.new_full((len(points),), scale)
    conditioned = scale if condition_scale is None else condition_scale
    embedding = field.condition_embedding(points.new_full((len(points),), conditioned).log())
    native = alpha_at(scales, schedule)[:, None] * points
    linear = F.linear(native, field.input.weight)
    constant = field.input.bias[None, :] + field.input_condition(embedding)
    state = linear + constant
    residual = torch.zeros_like(state)
    kappa = 2.**-.5
    for block in field.blocks:
        next_state = block(state, embedding)
        update = next_state / kappa - state
        linear, constant, residual = (kappa * linear, kappa * constant,
                                      kappa * (residual + update))
        state = next_state
    return {"state": state, "linear": linear, "input_constant": constant,
            "residual_updates": residual}


def energy_accounting(parts):
    h = center(parts["state"])
    energy = h.square().sum()
    result = {"centered_state_rms": float(h.square().mean().sqrt())}
    names = ("linear", "input_constant", "residual_updates")
    centered = {name: center(parts[name]) for name in names}
    for name in names:
        value = centered[name]
        result[f"{name}_energy_ratio"] = float(value.square().sum()/energy)
        result[f"{name}_signed_projection"] = float((h*value).sum()/energy)
    result["twice_all_cross_ratio"] = sum(
        float(2*(centered[a]*centered[b]).sum()/energy)
        for i, a in enumerate(names) for b in names[i+1:])
    result["signed_projection_sum"] = sum(result[f"{name}_signed_projection"] for name in names)
    result["linear_approximation_relative_mse"] = float((h-centered["linear"]).square().sum()/energy)
    result["accounting_max_error"] = float((parts["state"]-sum(parts[name] for name in names)).abs().max())
    return result


def gain(value, points):
    return float((value * points).sum() / points.square().sum())


@torch.no_grad()
def run(source_roots, output, *, device="cpu", count=128,
        scales=(.01, .0316227766, .1, .316227766, 1.), frozen_condition=.01):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rows, arrays, sources = [], {}, {}
    for root in map(Path, source_roots):
        fixture = torch.load(root/"data.pt", map_location="cpu", weights_only=False)
        if fixture["data_law"] != "gaussian":
            raise ValueError("Only the affine-Gaussian oracle is supported")
        evaluation = torch.load(root/"evaluation_data.pt", map_location="cpu", weights_only=False)
        queries, clean, noise = (evaluation[k][:count].to(device=device, dtype=torch.float64)
                                 for k in ("queries", "clean", "noise"))
        tau2 = float(fixture["latent_std"])**2
        arrays[f"{root.name}/queries"] = queries.cpu()
        arrays[f"{root.name}/clean"] = clean.cpu()
        arrays[f"{root.name}/noise"] = noise.cpu()
        for arm, schedule in ARM_SCHEDULE.items():
            checkpoint = root/f"{arm}_tailavg"/"checkpoint_step_009000.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            field, payload = load_cause_checkpoint(checkpoint, device)
            field = evaluation_copy(field, arm, payload["anchor"])
            sources[f"{root.name}/{arm}"] = {"path": str(checkpoint), "sha256": digest}
            for scale in scales:
                alpha = float(alpha_at(queries.new_tensor([scale]), schedule)[0])
                total_alpha = alpha * arm_multiplier(arm, payload["anchor"])
                oracle_gain = tau2/(tau2+scale**2)
                for group, points in (("queries", queries), ("noisy_population", clean+scale*noise)):
                    parts = hidden_parts(field, points, scale, schedule)
                    actual = field.output(parts["state"])
                    direct = canonical_posterior(field, points, scale, schedule)
                    direct_error = float((actual-direct).abs().max())
                    if direct_error > 1e-10:
                        raise AssertionError(direct_error)
                    row = {"source": root.name, "arm": arm, "lambda": scale,
                           "group": group, "count": len(points), "total_native_alpha": total_alpha,
                           "oracle_canonical_gain": oracle_gain,
                           "oracle_native_gain": oracle_gain/total_alpha,
                           **energy_accounting(parts), "forward_max_error": direct_error}
                    if group == "queries":
                        negative = hidden_parts(field, -points, scale, schedule)
                        odd_actual = (actual-field.output(negative["state"]))/2
                        fixed_positive = hidden_parts(field, points, scale, schedule, condition_scale=frozen_condition)
                        fixed_negative = hidden_parts(field, -points, scale, schedule, condition_scale=frozen_condition)
                        odd_fixed = (field.output(fixed_positive["state"])-field.output(fixed_negative["state"]))/2
                        row.update(learned_canonical_odd_gain=gain(odd_actual, points),
                                   learned_native_odd_gain=gain(odd_actual, points)/total_alpha,
                                   frozen_condition_canonical_odd_gain=gain(odd_fixed, points),
                                   conditioning_delta_canonical_gain=gain(odd_actual-odd_fixed, points),
                                   conditioning_delta_native_gain=gain(odd_actual-odd_fixed, points)/total_alpha)
                        if isinstance(field.output[0], torch.nn.Identity):
                            leading = F.linear(F.relu(parts["linear"]), field.output[-1].weight)
                            leading_negative = F.linear(F.relu(negative["linear"]), field.output[-1].weight)
                            odd_leading = (leading-leading_negative)/2
                            row["leading_canonical_odd_gain"] = gain(odd_leading, points)
                            row["leading_vs_actual_odd_relative_mse"] = float((odd_leading-odd_actual).square().sum()/odd_actual.square().sum())
                        else:
                            row["leading_canonical_odd_gain"] = 0.
                            row["leading_vs_actual_odd_relative_mse"] = 1.
                    else:
                        for key in ("learned_canonical_odd_gain", "learned_native_odd_gain", "frozen_condition_canonical_odd_gain",
                                    "conditioning_delta_canonical_gain", "conditioning_delta_native_gain",
                                    "leading_canonical_odd_gain", "leading_vs_actual_odd_relative_mse"):
                            row[key] = None
                    rows.append(row)
                    for name, value in parts.items():
                        arrays[f"{root.name}/{arm}/{scale}/{group}/{name}"] = value.cpu()
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
                raise AssertionError("Source checkpoint changed")
            print(json.dumps({"source": root.name, "arm": arm, "elapsed_seconds": time.monotonic()-start}), flush=True)
    write_csv(output/"metrics.csv", rows)
    torch.save(arrays, output/"arrays.pt")
    summary = {"sources": sources, "scales": list(scales), "count": count,
               "frozen_condition": frozen_condition, "elapsed_seconds": time.monotonic()-start,
               "maximum_accounting_error": max(r["accounting_max_error"] for r in rows),
               "maximum_forward_error": max(r["forward_max_error"] for r in rows),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Inference-only exact hidden decomposition and fixed-native-input conditioning comparison. Component energies need cross terms; signed projections sum to one. Input constant excludes conditioning inside nonlinear residual updates. No training causal claim."}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
