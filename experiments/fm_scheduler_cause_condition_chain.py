"""Exact log-scale chain on held-fixed observations; no model intervention.

The native-input and explicit-condition contributions sum to the ordinary
derivative of the deployed posterior. Gaussian oracle derivatives are exact.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import torch

from experiments.fm_mechanism_diagnostics import alpha_at
from experiments.fm_scheduler_cause_training import (
    ARM_SCHEDULE, arm_multiplier, load_cause_checkpoint)


def chain(field, observations, scale, arm, anchor):
    t = torch.full((len(observations),), math.log(scale), dtype=observations.dtype)
    alpha = alpha_at(t.exp(), ARM_SCHEDULE[arm]) * arm_multiplier(arm, anchor)
    native = alpha[:, None] * observations
    value, native_radial = torch.autograd.functional.jvp(
        lambda y: field(y, t), native, native)
    _, explicit = torch.autograd.functional.jvp(
        lambda condition: field(native, condition), t, torch.ones_like(t))
    q = -scale / (1 + scale) if arm.startswith("rect") else -scale**2 / (1 + scale**2)
    input_term = q * native_radial
    total = explicit + input_term
    eps = 1e-5

    def shifted(dt):
        shifted_t = t + dt
        amplitude = alpha_at(shifted_t.exp(), ARM_SCHEDULE[arm]) * arm_multiplier(arm, anchor)
        return field(amplitude[:, None] * observations, shifted_t)

    difference = (shifted(eps) - shifted(-eps)) / (2 * eps)
    max_error = float((total - difference).abs().max())
    assert max_error < 1e-6, (arm, scale, max_error)
    return value, explicit, input_term, total, max_error


def run(args):
    torch.set_num_threads(4)
    root = args.artifacts
    rows, hashes = [], {}
    for suffix in ("anchored", "no_final_ln"):
        directory = root / ("gaussian_" + suffix)
        fixture = torch.load(directory / "data.pt", weights_only=False)
        evaluation = torch.load(directory / "evaluation_data.pt", weights_only=False)
        basis = torch.linalg.qr(fixture["tangent_basis"].double()).Q
        variance = float(fixture["latent_std"])**2
        for arm in ("rect", "vp_low", "rect_high", "vp"):
            path = directory / (arm + "_tailavg") / "checkpoint_step_009000.pt"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            field, payload = load_cause_checkpoint(path)
            field = field.double().eval()
            for scale in (.03, .1, .2, .316227766, .5, 1.):
                for kind in ("clean", "noisy"):
                    observations = (evaluation["queries"][:128] if kind == "clean" else
                        evaluation["clean"][:128] + scale * evaluation["noise"][:128]).double()
                    value, explicit, input_term, total, check = chain(
                        field, observations, scale, arm, payload["anchor"])
                    projected = (observations @ basis) @ basis.T
                    gain = variance / (variance + scale**2)
                    truth = gain * projected
                    truth_derivative = -2 * variance * scale**2 / (variance + scale**2)**2 * projected
                    reference = truth.square().sum(-1).mean()
                    energies = lambda z: float(z.square().sum(-1).mean() / reference)
                    row = {"architecture": suffix, "arm": arm, "lambda": scale,
                        "observations": kind, "count": len(observations),
                        "input_contribution_energy": energies(input_term),
                        "explicit_condition_energy": energies(explicit),
                        "total_derivative_energy": energies(total),
                        "oracle_derivative_energy": energies(truth_derivative),
                        "derivative_error_energy": energies(total-truth_derivative),
                        "posterior_error_energy": energies(value-truth),
                        "condition_input_cross": float(2*(explicit*input_term).sum(-1).mean()/reference),
                        "finite_difference_max_error": check,
                        "dlog_alpha_dlog_lambda": (-scale/(1+scale) if arm.startswith("rect") else -scale**2/(1+scale**2))}
                    residual = row["total_derivative_energy"] - row["input_contribution_energy"] - row["explicit_condition_energy"] - row["condition_input_cross"]
                    assert abs(residual) < 1e-12
                    rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "condition_chain.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "condition_chain_manifest.json").write_text(json.dumps({
        "source_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "semantics": "Exact ordinary log-scale derivative; observations held fixed, including noisy ones. Descriptive fixed-checkpoint chain, not attribution of training history.",
        "energy_units": "Mean squared derivative divided by mean squared true posterior norm",
        "rows": len(rows), "max_finite_difference_error": max(r["finite_difference_max_error"] for r in rows)
    }, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/fm_scheduler_cause_20260906"))
    parser.add_argument("--output", type=Path, default=Path("../notes/fm_scheduler_cause_20260906"))
    run(parser.parse_args())
