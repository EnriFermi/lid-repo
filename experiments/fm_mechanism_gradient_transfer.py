"""Local causal transfer between scale-specific risks in frozen FM fields.

Uses a fixed checkpoint Adam-v metric, not a full Adam/momentum update.
Small parameter steps validate the signed differential. No training loop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, population_posterior


def sample(fixture, count, seed, device):
    gen = torch.Generator().manual_seed(seed)
    basis = fixture["tangent_basis"].double()
    d, D = basis.shape[1], basis.shape[0]
    law = fixture.get("data_law", "gaussian")
    if law == "gaussian":
        z = torch.randn(count, d, generator=gen, dtype=torch.float64)
        clean = float(fixture["latent_std"]) * z @ basis.T
    elif law == "uniform":
        z = 2 * torch.rand(count, d, generator=gen, dtype=torch.float64) - 1
        clean = float(fixture["latent_half_width"]) * z @ basis.T
    elif law == "circle":
        angle = 2 * math.pi * torch.rand(count, 1, generator=gen, dtype=torch.float64)
        clean = float(fixture["circle_radius"]) * torch.cat((angle.cos(), angle.sin()), 1)
    else:
        raise ValueError(law)
    noise = torch.randn(count, D, generator=gen, dtype=torch.float64)
    return clean.to(device), noise.to(device)


def block_name(name):
    if name.startswith("condition_embedding.") or name.startswith("input_condition.") or ".condition." in name:
        return "conditioning"
    if name.startswith("input."):
        return "input"
    if name.startswith("output."):
        return "output"
    return "residual_trunk"


def adam_metric(field, optimizer_state):
    names = [name for name, _ in field.named_parameters()]
    params = list(field.parameters())
    ids = [i for group in optimizer_state["param_groups"] for i in group["params"]]
    if len(ids) != len(params) or len(optimizer_state["param_groups"]) != 1:
        raise ValueError("Probe requires the pilot's single-group standard AdamW")
    group = optimizer_state["param_groups"][0]
    beta2, eps = group["betas"][1], group["eps"]
    diagonal = []
    for p, i in zip(params, ids):
        state = optimizer_state["state"][i]
        v = state["exp_avg_sq"].to(p)
        if v.shape != p.shape:
            raise ValueError("Adam parameter order/shape mismatch")
        diagonal.append((v / (1 - beta2 ** float(state["step"]))).sqrt().add(eps).reciprocal())
    return names, params, diagonal


def tangent_part(error, clean, fixture):
    if fixture.get("data_law") == "circle":
        radial = clean / clean.norm(dim=-1, keepdim=True)
        tangent = torch.stack((-radial[:, 1], radial[:, 0]), 1)
        return (error * tangent).sum(-1, keepdim=True)
    q, _ = torch.linalg.qr(fixture["tangent_basis"].to(error))
    return error @ q


def inner_product(left, right):
    return sum(float((a * b).sum()) for a, b in zip(left, right))


def masked_direction(names, direction, mask):
    if mask not in {"all", "without_conditioning", "conditioning_only"}:
        raise ValueError(mask)
    return [d if mask == "all" or ((block_name(n) == "conditioning") == (mask == "conditioning_only"))
            else torch.zeros_like(d) for n, d in zip(names, direction)]


def gradient(field, fn):
    field.zero_grad(set_to_none=True)
    loss = fn()
    grads = torch.autograd.grad(loss, tuple(field.parameters()))
    return float(loss.detach()), [g.detach().clone() for g in grads]


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fixture = torch.load(args.checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    train_x, train_z = sample(fixture, args.train_count, 20260905 + 7101, args.device)
    eval_x, eval_z = sample(fixture, args.query_count, 20260905 + 7102, args.device)
    coordinate = payload["config"].get("input_coordinate", "native")
    rows, derivatives, validations = [], [], []
    start = time.monotonic()
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(args.checkpoint, schedule, args.device)
        field = field.double().eval()
        names, params, metric = adam_metric(field, payload["optimizers"][schedule])
        backup = [p.detach().clone() for p in params]
        parameter_norm = math.sqrt(inner_product(backup, backup))
        g_train = {}
        for scale in args.train_scales:
            field.zero_grad(set_to_none=True)
            total = 0.0
            for begin in range(0, len(train_x), args.batch_size):
                x = train_x[begin:begin + args.batch_size]
                z = train_z[begin:begin + args.batch_size]
                r = x + scale * z
                with torch.no_grad():
                    target = population_posterior(r, scale, fixture)
                pred = canonical_posterior(field, r, scale, schedule, input_coordinate=coordinate)
                loss = (pred - target).square().mean(-1).sum() / (len(train_x) * scale**2)
                loss.backward()
                total += float(loss.detach())
            g = [p.grad.detach().clone() for p in params]
            direction = [m * a for m, a in zip(metric, g)]
            norm = math.sqrt(inner_product(direction, direction))
            eta = args.relative_step * parameter_norm / max(norm, 1e-30)
            g_train[scale] = (g, direction, eta)
            rows.append({"schedule": schedule, "lambda": scale, "population_excess_risk": total,
                         "gradient_norm": math.sqrt(inner_product(g, g)), "metric_direction_norm": norm,
                         "step_multiplier": eta, "relative_parameter_step": args.relative_step})
        evaluation = {}
        for scale in args.eval_scales:
            if fixture.get("data_law") == "circle":
                radial = eval_x / eval_x.norm(dim=-1, keepdim=True)
                normal_noise = (eval_z * radial).sum(-1, keepdim=True) * radial
            else:
                q, _ = torch.linalg.qr(fixture["tangent_basis"].to(eval_x))
                normal_noise = eval_z - (eval_z @ q) @ q.T
            locations = {"clean": eval_x, "noisy": eval_x + scale * eval_z}
            if args.spatial_strata:
                locations["normal_only"] = eval_x + scale * normal_noise
            strata = {"all": torch.ones(len(eval_x), device=eval_x.device, dtype=torch.bool)}
            if args.spatial_strata and fixture.get("data_law") == "uniform":
                distance = (float(fixture["latent_half_width"]) - (eval_x @ q).abs()).min(-1).values
                strata.update(interior=distance > 3 * scale, boundary=distance < scale)
            selected = [(loc if stratum == "all" else loc + "_" + stratum, p[mask], eval_x[mask])
                        for loc, p in locations.items() for stratum, mask in strata.items() if int(mask.sum()) >= 8]
            for location, points, clean_base in selected:
                with torch.no_grad():
                    target = population_posterior(points, scale, fixture)
                def loss_fn(points=points, target=target, scale=scale, clean_base=clean_base):
                    pred = canonical_posterior(field, points, scale, schedule, input_coordinate=coordinate)
                    error = tangent_part(pred - target, clean_base, fixture)
                    return error.square().sum(-1).mean() / scale**2
                value, h = gradient(field, loss_fn)
                evaluation[(scale, location)] = (loss_fn, value, h)
                for train_scale, (g, direction, eta) in g_train.items():
                    contributions = {}
                    for name, hi, di in zip(names, h, direction):
                        label = block_name(name)
                        contributions[label] = contributions.get(label, 0.0) + float((hi * di).sum())
                    alignment = inner_product(h, direction)
                    euclidean = inner_product(h, g)
                    direction_cosine = alignment / max(math.sqrt(inner_product(h, h) * inner_product(direction, direction)), 1e-30)
                    metric_h = [m * hi for m, hi in zip(metric, h)]
                    metric_cosine = alignment / max(math.sqrt(inner_product(h, metric_h) * inner_product(g, direction)), 1e-30)
                    derivatives.append({"schedule": schedule, "eval_lambda": scale, "location": location,
                                        "eval_query_count": len(points),
                                        "train_lambda": train_scale, "eval_error": value,
                                        "metric_alignment": alignment, "euclidean_alignment": euclidean,
                                        "direction_cosine": direction_cosine, "metric_cosine": metric_cosine,
                                        "predicted_change": -eta * alignment,
                                        **{f"alignment_{k}": v for k, v in contributions.items()}})
        for train_scale, (_, direction, eta) in g_train.items():
            for mask in ("all", "without_conditioning", "conditioning_only"):
                actual_direction = masked_direction(names, direction, mask)
                for fraction in (1.0, .5):
                    with torch.no_grad():
                        for p, original, delta in zip(params, backup, actual_direction):
                            p.copy_(original - fraction * eta * delta)
                    for (scale, location), (loss_fn, value, h) in evaluation.items():
                        with torch.no_grad():
                            after = float(loss_fn())
                        expected = -fraction * eta * inner_product(h, actual_direction)
                        actual = after - value
                        validations.append({"schedule": schedule, "eval_lambda": scale, "location": location,
                                            "train_lambda": train_scale, "step_fraction": fraction,
                                            "block_mask": mask, "step_multiplier": eta,
                                            "before": value, "after": after, "actual_change": actual,
                                            "predicted_change": expected,
                                            "relative_linearization_error": abs(actual - expected) / max(abs(expected), 1e-20)})
                    with torch.no_grad():
                        for p, original in zip(params, backup):
                            p.copy_(original)
        assert all(torch.equal(p, b) for p, b in zip(params, backup))
        print(json.dumps({"schedule": schedule, "status": "complete", "elapsed": time.monotonic() - start}), flush=True)
    for name, values in (("training_gradients", rows), ("transfer", derivatives), ("step_validation", validations)):
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_count": args.train_count, "query_count": args.query_count,
              "train_scales": args.train_scales, "eval_scales": args.eval_scales,
              "training_data": "Fresh fixed-seed population draws, paired across schedules and scales",
              "target": "Analytic conditional posterior, preserving expected unclipped physical-risk gradient",
              "metric": "Frozen checkpoint Adam second-moment diagonal only; no momentum, weight decay or clipping",
              "evaluation": "Posterior tangent value error; local tangent at base clean query for curved geometry",
              "spatial_strata": args.spatial_strata,
              "uniform_strata": "Interior face-distance >3lambda; boundary <lambda; skip masks with fewer than8 queries",
              "intervention": "Small fixed-metric gradient steps in memory, then bitwise restoration; no saved model mutation",
              "block_mask_control": "Same eta from full direction, no renormalization after dropping/retaining conditioning block",
              "scope": "Local signed interference diagnostic, not a trajectory-level or convergence proof",
              "max_relative_step": args.relative_step, "elapsed": time.monotonic() - start}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--relative-step", type=float, default=1e-6)
    parser.add_argument("--spatial-strata", action="store_true")
    parser.add_argument("--train-scales", nargs="+", type=float, default=[.03, .316227766, 1.])
    parser.add_argument("--eval-scales", nargs="+", type=float, default=[.316227766, 1.])
    args = parser.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
