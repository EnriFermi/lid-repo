"""Actual full-LID consequences of a fixed-metric conditioning-block step."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_mechanism_gradient_transfer import sample, adam_metric, inner_product, masked_direction
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, population_posterior


@torch.no_grad()
def readout(field, points, scale, schedule, fixture, oracle):
    pairs = [fast_value_and_trace(field, part, scale, schedule) for part in points.split(32)]
    value, response = (torch.cat([p[i] for p in pairs]) for i in range(2))
    correction = (value - points).square().sum(-1) / scale**2
    truth = oracle(points, scale)
    if fixture.get("data_law") == "uniform":
        q, _ = torch.linalg.qr(fixture["tangent_basis"].to(points))
        distance = (float(fixture["latent_half_width"]) - (points @ q).abs()).min(-1).values
        masks = {"all": torch.ones(len(points), device=points.device, dtype=torch.bool),
                 "interior": distance > 3 * scale, "boundary": distance < scale}
    else:
        masks = {"all": torch.ones(len(points), device=points.device, dtype=torch.bool)}
    result = {}
    for name, mask in masks.items():
        if int(mask.sum()) < 8:
            continue
        full = response[mask] + correction[mask]
        radius_sq = points[mask].square().sum(-1).clamp_min(1e-30)
        displacement = value[mask] - points[mask]
        oracle_error = value[mask] - truth.mean[mask]
        result[name] = {"count": int(mask.sum()), "full_oracle_mae": float((full - truth.full[mask]).abs().mean()),
                       "full_oracle_mse": float((full - truth.full[mask]).square().mean()),
                       "full_mae_lid": float((full - field.pilot_intrinsic).abs().mean()),
                       "response_mean": float(response[mask].mean()),
                       "response_oracle_mae": float((response[mask] - truth.response[mask]).abs().mean()),
                       "correction_mean": float(correction[mask].mean()),
                       "oracle_correction_mean": float(truth.correction[mask].mean()),
                       "origin_radial_gain_error": float((oracle_error * points[mask]).sum() / radius_sq.sum()),
                       "origin_radial_displacement_gain": float((displacement * points[mask]).sum() / radius_sq.sum()),
                       "origin_radial_correction_mean": float(((displacement * points[mask]).sum(-1).square() / radius_sq / scale**2).mean()),
                       "posterior_oracle_energy_over_lambda2": float((value[mask] - truth.mean[mask]).square().sum(-1).mean() / scale**2)}
    return result


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fixture = torch.load(args.checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    oracle, _ = make_oracle(payload["config"], fixture, args.device)
    train_x, train_z = sample(fixture, 2048, 20260905 + 7101, args.device)
    queries, _ = sample(fixture, args.query_count, 20260905 + 7102, args.device)
    source_points = train_x + args.train_scale * train_z
    target = population_posterior(source_points, args.train_scale, fixture).detach()
    rows, start = [], time.monotonic()
    for schedule in args.schedules:
        field, _ = load_pilot_checkpoint(args.checkpoint, schedule, args.device)
        field = field.double().eval()
        field.pilot_intrinsic = payload["config"]["intrinsic_dim"]
        names, params, metric = adam_metric(field, payload["optimizers"][schedule])
        backup = [p.detach().clone() for p in params]
        field.zero_grad(set_to_none=True)
        before_source = 0.
        for begin in range(0, len(source_points), 256):
            prediction = canonical_posterior(field, source_points[begin:begin + 256], args.train_scale, schedule)
            loss = (prediction - target[begin:begin + 256]).square().mean(-1).sum() / (len(source_points) * args.train_scale**2)
            loss.backward()
            before_source += float(loss.detach())
        direction = [m * p.grad.detach() for m, p in zip(metric, params)]
        eta = args.relative_step * math.sqrt(inner_product(backup, backup) / inner_product(direction, direction))
        before = readout(field, queries, args.eval_scale, schedule, fixture, oracle)
        for mask in ("all", "without_conditioning", "conditioning_only"):
            delta = masked_direction(names, direction, mask)
            for fraction in (1., .5):
                with torch.no_grad():
                    for p, saved, change in zip(params, backup, delta):
                        p.copy_(saved - fraction * eta * change)
                    prediction = canonical_posterior(field, source_points, args.train_scale, schedule)
                    after_source = float((prediction - target).square().mean() / args.train_scale**2)
                after = readout(field, queries, args.eval_scale, schedule, fixture, oracle)
                for stratum in before:
                    rows.append({"schedule": schedule, "train_lambda": args.train_scale, "eval_lambda": args.eval_scale,
                                 "block_mask": mask, "step_fraction": fraction, "stratum": stratum,
                                 "query_count": before[stratum]["count"], "step_multiplier": eta,
                                 "source_risk_before": before_source, "source_risk_after": after_source,
                                 **{f"before_{k}": v for k, v in before[stratum].items() if k != "count"},
                                 **{f"after_{k}": v for k, v in after[stratum].items() if k != "count"},
                                 **{f"change_{k}": after[stratum][k] - before[stratum][k] for k in before[stratum] if k != "count"}})
                with torch.no_grad():
                    for p, saved in zip(params, backup):
                        p.copy_(saved)
        assert all(torch.equal(p, original) for p, original in zip(params, backup))
        print(json.dumps({"schedule": schedule, "status": "complete", "elapsed": time.monotonic() - start}), flush=True)
    with (args.output / "readout_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"source": str(args.checkpoint), "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_risk": "Noisy analytic-posterior MSE /(D lambda^2); original unclipped population gradient",
                "step": "Frozen Adam-v metric, same eta for full versus block masks; no momentum/clipping/decay",
                "readout": "Exact float64 spatial trace plus squared bias; compared with geometry-specific finite-scale oracle and d separately",
                "heldout": "Fresh same-query seeds as gradient spatial probe; strata fixed before outcomes",
                "training_count": 2048, "query_count": args.query_count, "relative_step": args.relative_step,
                "train_scale": args.train_scale, "eval_scale": args.eval_scale,
                "restoration": "All model parameters bitwise restored, source file unchanged",
                "scope": "Local causal readout effect, not a resumed Adam training trajectory",
                "elapsed": time.monotonic() - start}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-count", type=int, default=512)
    parser.add_argument("--relative-step", type=float, default=1e-6)
    parser.add_argument("--train-scale", type=float, required=True)
    parser.add_argument("--eval-scale", type=float, required=True)
    parser.add_argument("--schedules", nargs="+", default=["rectified_linear", "vp_trigonometric"])
    args = parser.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
