"""Fixed-scale function-preserving input-coordinate control, without training.

Transport alpha -> a*alpha and W_input -> W_input/a. Native distributions
change scale, while the represented canonical function stays unchanged.
This isolates instantaneous coordinate effects from learned-state differences.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_mechanism_gradient_transfer import block_name, sample
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, population_posterior
from experiments.fm_mechanism_regional_transfer import observed_regions
from models.affine_flow import affine_schedule_state


def relative_error(actual, expected):
    numerator = sum(float((a - e).square().sum()) for a, e in zip(actual, expected))
    denominator = sum(float(e.square().sum()) for e in expected)
    return math.sqrt(numerator / max(denominator, 1e-100))


def source_gradient(field, points, target, mask, scale, schedule):
    field.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(points), 256):
        pred = canonical_posterior(field, points[start:start + 256], scale, schedule)
        error = (pred - target[start:start + 256]).square().mean(-1)
        loss = (error * mask[start:start + 256]).sum() / (len(points) * scale**2)
        loss.backward()
        total += float(loss.detach())
    return total, [p.grad.detach().clone() for p in field.parameters()]


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    source_digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fixture_path = args.checkpoint.parent / "data.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    if payload["config"].get("input_coordinate", "native") != "native":
        raise ValueError("Expected the raw native-input control")
    if payload["config"].get("output_coordinate", "data") != "data":
        raise ValueError("Expected data-coordinate posterior outputs")
    clean, noise = sample(fixture, 2048, 20260905 + 7101, args.device)
    queries, _ = sample(fixture, 32, 20260905 + 7102, args.device)
    points = clean + args.scale * noise
    target = population_posterior(points, args.scale, fixture).detach()
    regions = observed_regions(points, fixture, args.scale)
    regions["all"] = torch.ones(len(points), device=args.device, dtype=torch.bool)
    scale_tensor = torch.tensor(args.scale, dtype=torch.float64, device=args.device)
    rows = []
    schedules = ("rectified_linear", "vp_trigonometric")
    for source_schedule, destination_schedule in (schedules, tuple(reversed(schedules))):
        field, _ = load_pilot_checkpoint(args.checkpoint, source_schedule, args.device)
        field = field.double().eval()
        initial_state = {n: p.detach().clone() for n, p in field.state_dict().items()}
        transported = copy.deepcopy(field)
        alpha_source = float(affine_schedule_state(scale_tensor, source_schedule).alpha)
        alpha_destination = float(affine_schedule_state(scale_tensor, destination_schedule).alpha)
        factor = alpha_destination / alpha_source
        with torch.no_grad():
            transported.input.weight.div_(factor)
        names = [name for name, _ in field.named_parameters()]
        input_weight_index = names.index("input.weight")
        conditioning_indices = [i for i, name in enumerate(names) if block_name(name) == "conditioning"]
        with torch.no_grad():
            value, trace = fast_value_and_trace(field, queries, args.scale, source_schedule)
            new_value, new_trace = fast_value_and_trace(transported, queries, args.scale, destination_schedule)
        value_max_error = float((new_value - value).abs().max())
        trace_max_error = float((new_trace - trace).abs().max())
        assert value_max_error < 1e-10 and trace_max_error < 1e-10
        for region in ("all", "boundary", "interior"):
            loss, grads = source_gradient(field, points, target, regions[region], args.scale, source_schedule)
            new_loss, new_grads = source_gradient(transported, points, target, regions[region], args.scale, destination_schedule)
            expected = [g * factor if i == input_weight_index else g for i, g in enumerate(grads)]
            gradient_error = relative_error(new_grads, expected)
            condition_error = relative_error([new_grads[i] for i in conditioning_indices], [grads[i] for i in conditioning_indices])
            assert gradient_error < 1e-9 and condition_error < 1e-9
            squared_norm = sum(float(g.square().sum()) for g in grads)
            new_squared_norm = sum(float(g.square().sum()) for g in new_grads)
            input_squared_norm = float(grads[input_weight_index].square().sum())
            clip_limit = float(payload["config"]["gradient_clip"])
            clip_factor = min(1.0, clip_limit / (math.sqrt(squared_norm) + 1e-6)) if clip_limit > 0 else 1.0
            new_clip_factor = min(1.0, clip_limit / (math.sqrt(new_squared_norm) + 1e-6)) if clip_limit > 0 else 1.0
            rows.append({
                "source_schedule": source_schedule, "transported_to": destination_schedule,
                "region": region, "alpha_factor": factor, "source_loss": loss,
                "loss_absolute_difference": abs(new_loss - loss),
                "value_max_absolute_difference": value_max_error,
                "canonical_trace_max_absolute_difference": trace_max_error,
                "gradient_transport_relative_error": gradient_error,
                "conditioning_gradient_relative_error": condition_error,
                "raw_gradient_norm": math.sqrt(squared_norm),
                "transported_raw_gradient_norm": math.sqrt(new_squared_norm),
                "input_weight_gradient_energy_fraction": input_squared_norm / max(squared_norm, 1e-100),
                "hypothetical_clip_factor": clip_factor,
                "transported_hypothetical_clip_factor": new_clip_factor,
                "conditioning_clip_multiplier_ratio": new_clip_factor / clip_factor,
            })
        assert all(torch.equal(p, initial_state[n]) for n, p in field.state_dict().items())
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == source_digest
    result = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": source_digest,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scale": args.scale, "source_count": 2048, "query_count": 32,
        "scope": "Exact fixed-scale coordinate control; no training or saved-state mutation. Clipping numbers apply to analytic-target regional diagnostic gradients, not historical mixed-scale minibatches.",
        "intervention": "alpha_destination/alpha_source=factor; first-layer weight divided by factor; all other parameters unchanged",
        "rows": rows,
    }
    (args.output / "transport.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=.316227766)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())
