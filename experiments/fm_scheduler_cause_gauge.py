"""Function-preserving conditioning gauge and original-loss continuations.

The embedding's final affine map is multiplied by c and every linear consumer
is divided by c. Forward predictions at ALL noise levels are unchanged before
continuation. Adam moments and epsilon follow the gradient coordinate change.
Subsequent steps use ordinary clean-X MSE, backward, clipping and AdamW; no
synthetic output residual or gradient hook is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_mechanism_crossgeometry_optimizer import sample_batch
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, state_digest
from experiments.fm_mechanism_readout_transfer import readout


def gauge_block(name):
    if name.startswith("condition_embedding.projection.2."):
        return "embedding_final"
    if name == "input_condition.weight" or name.endswith(".condition.weight"):
        return "condition_consumers"
    return "other"


def parameter_factor(name, factor):
    block = gauge_block(name)
    return factor if block == "embedding_final" else 1 / factor if block == "condition_consumers" else 1.0


def independent_adam(field, source_state):
    """Restore either the original one-group state or our named gauge groups."""
    named = dict(field.named_parameters())
    groups = []
    for source_group in source_state["param_groups"]:
        names = source_group.get("param_names")
        if names is None:
            if len(source_state["param_groups"]) != 1:
                raise ValueError("Unnamed optimizer checkpoint must have one group")
            names = list(named)
        if len(names) != len(source_group["params"]):
            raise ValueError("Optimizer parameter names/order mismatch")
        group = {k: copy.deepcopy(v) for k, v in source_group.items() if k != "params"}
        group["params"] = [named[name] for name in names]
        groups.append(group)
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(copy.deepcopy(source_state))
    return optimizer


@torch.no_grad()
def apply_condition_gauge(field, optimizer, factor):
    """Apply theta'=q theta, m'=m/q, v'=v/q^2, epsilon'=epsilon/q.

    This preserves the represented function, not the future Adam trajectory.
    Group learning rates and decay are deliberately unchanged. The continuation
    explicitly chooses source- or gauge-coordinate clipping norms.
    The identity factor leaves both parameters and optimizer byte-equivalent.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Gauge factor must be finite and positive")
    named = dict(field.named_parameters())
    required = {"condition_embedding.projection.2.weight", "condition_embedding.projection.2.bias",
                "input_condition.weight"}
    required.update(f"blocks.{i}.condition.weight" for i in range(len(field.blocks)))
    if not required <= named.keys() or not isinstance(field.condition_embedding.projection[2], torch.nn.Linear):
        raise ValueError("Unsupported conditioning architecture")
    factors = {name: parameter_factor(name, factor) for name in named}
    if factor == 1:
        return factors
    reverse = {id(parameter): name for name, parameter in named.items()}
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != set(reverse):
        raise ValueError("Optimizer parameters do not match this field")
    for name, parameter in named.items():
        q = factors[name]
        if q == 1:
            continue
        parameter.mul_(q)
        state = optimizer.state.get(parameter, {})
        if "exp_avg" in state:
            state["exp_avg"].div_(q)
            state["exp_avg_sq"].div_(q * q)
        if "max_exp_avg_sq" in state:
            state["max_exp_avg_sq"].div_(q * q)
    regrouped = []
    for group in optimizer.param_groups:
        by_factor = {}
        for parameter in group["params"]:
            name = reverse[id(parameter)]
            q = factors[name]
            if q not in by_factor:
                item = {k: copy.deepcopy(v) for k, v in group.items() if k not in {"params", "param_names"}}
                item.update(params=[], param_names=[], eps=group["eps"] / q)
                by_factor[q] = item
            by_factor[q]["params"].append(parameter)
            by_factor[q]["param_names"].append(name)
        regrouped.extend(by_factor.values())
    optimizer.param_groups = regrouped
    return factors


def original_loss(field, clean, noise, scales, schedule):
    inputs = clean + scales[:, None] * noise
    prediction = canonical_posterior(field, inputs, scales, schedule)
    return ((prediction - clean).square().mean(-1) / scales.square()).mean()


def draw_batch(config, fixture, generator, device):
    if config.get("data_law", "gaussian") != "gaussian":
        return sample_batch(config, fixture, generator, device)
    latent = torch.randn(config["batch_size"], config["intrinsic_dim"], device=device, generator=generator)
    clean = (float(fixture["latent_std"]) * latent) @ fixture["tangent_basis"].to(device).T
    noise = torch.randn(config["batch_size"], config["ambient_dim"], device=device, generator=generator)
    uniforms = torch.rand(config["batch_size"], device=device, generator=generator)
    scales = (math.log(config["lambda_min"]) + uniforms * math.log(config["lambda_max"] / config["lambda_min"])).exp()
    return clean, noise, scales


def optimizer_step(field, optimizer, batch, schedule, clip, *, clip_parameter_factors=None):
    optimizer.zero_grad(set_to_none=True)
    loss = original_loss(field, *batch, schedule)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite original training loss")
    loss.backward()
    if clip_parameter_factors is None or all(q == 1 for q in clip_parameter_factors.values()):
        norm = torch.nn.utils.clip_grad_norm_(field.parameters(), clip)
        native_norm = norm
    else:
        # g_original = q*g_gauge for theta_gauge=q*theta_original. Clip using
        # the original-coordinate norm, then scale the actual gradient by the
        # same common coefficient. No output residual or loss is changed.
        gradients = [(name, p.grad) for name, p in field.named_parameters() if p.grad is not None]
        native_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for _, g in gradients]))
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g * clip_parameter_factors[name]) for name, g in gradients]))
        coefficient = (clip / (norm + 1e-6)).clamp(max=1.0)
        for _, gradient in gradients:
            gradient.mul_(coefficient)
    optimizer.step()
    return {"loss": float(loss.detach()), "unclipped_gradient_norm": float(norm),
            "native_coordinate_gradient_norm": float(native_norm)}


def inspect_real_steps(source, output, *, batches=4, queries=64, device="cpu", seed=20260906):
    """Attribute fresh real-loss Adam steps to the two gauge parameter groups.

    Every batch starts at the SAME checkpoint, not a continuation chain. The
    block-only output changes are finite interventions using components of a
    real Adam step, not donor-error VJPs. Their sum is only first-order additive.
    """
    torch.set_num_threads(4)
    source, output = Path(source), Path(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent / "data.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = [draw_batch(config, fixture, generator, device) for _ in range(batches)]
    points = fixture["clean_queries"][:queries].to(device)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for schedule in ("rectified_linear", "vp_trigonometric"):
        original, _ = load_pilot_checkpoint(source, schedule, device)
        original.eval()
        saved = {name: value.detach().clone() for name, value in original.state_dict().items()}
        with torch.no_grad():
            pred0 = canonical_posterior(original, points, .316227766, schedule)
            gain0 = float((pred0 * points).sum() / points.square().sum())
        for index, batch in enumerate(draws):
            field = copy.deepcopy(original)
            optimizer = independent_adam(field, payload["optimizers"][schedule])
            values = optimizer_step(field, optimizer, batch, schedule, config["gradient_clip"])
            updated = {name: value.detach().clone() for name, value in field.state_dict().items()}
            for block in ("all", "embedding_final", "condition_consumers", "other"):
                with torch.no_grad():
                    for name, parameter in field.named_parameters():
                        parameter.copy_(updated[name] if block == "all" or gauge_block(name) == block else saved[name])
                    prediction = canonical_posterior(field, points, .316227766, schedule)
                    gain = float((prediction * points).sum() / points.square().sum())
                    rows.append({"schedule": schedule, "batch": index, "block": block, **values,
                                 "gain_before": gain0, "gain_after": gain, "gain_change": gain - gain0,
                                 "mean_squared_value_change": float((prediction - pred0).square().mean())})
        assert all(torch.equal(v, saved[n]) for n, v in original.state_dict().items())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    result = {"source": str(source), "source_sha256": source_hash, "fresh_batch_seed": seed,
              "batches": batches, "query_count": len(points), "rows": rows,
              "scope": "Actual clean-X mixed-lambda backward+clipping+AdamW; independent one-step copies. Block-only finite effects are not training trajectories."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def run(source, output, *, factor, steps=1000, device="cuda:1", interval=100, queries=32,
        schedules=("rectified_linear", "vp_trigonometric"), clip_coordinates="source"):
    if steps <= 0 or interval <= 0 or queries <= 0 or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Positive finite factor and positive counts required")
    if clip_coordinates not in {"source", "gauge"}:
        raise ValueError("clip_coordinates must be source or gauge")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, output = Path(source), Path(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent / "data.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    if config.get("input_coordinate", "native") != "native" or config.get("output_coordinate", "data") != "data":
        raise ValueError("Requires the native-input, data-output posterior pilot")
    if config.get("head_preconditioner", "elementwise") != "elementwise":
        raise ValueError("Requires standard AdamW")
    if config.get("input_multiplier", 1.0) != 1.0:
        raise ValueError("Nonstandard native input multiplier needs an explicit adapter")
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    variants = {"control": 1.0, "gauge": factor}
    fields, optimizers, factors = {}, {}, {}
    for label, c in variants.items():
        (output / label).mkdir()
        torch.save(fixture, output / label / "data.pt")
        for schedule in schedules:
            field, _ = load_pilot_checkpoint(source, schedule, device)
            optimizer = independent_adam(field, payload["optimizers"][schedule])
            factors[label, schedule] = apply_condition_gauge(field, optimizer, c)
            fields[label, schedule], optimizers[label, schedule] = field, optimizer
    checks = []
    points = fixture["clean_queries"][:queries].to(device)
    for schedule in schedules:
        for scale in (.01, .1, .316227766, 1.):
            with torch.no_grad():
                a = canonical_posterior(fields["control", schedule], points, scale, schedule)
                b = canonical_posterior(fields["gauge", schedule], points, scale, schedule)
            difference = float((a - b).abs().max())
            if not torch.allclose(a, b, atol=1e-5, rtol=2e-6):
                raise AssertionError((schedule, scale, difference))
            checks.append({"schedule": schedule, "lambda": scale, "initial_prediction_max_difference": difference})
    for schedule in schedules:
        left = dict(fields["control", schedule].named_parameters())
        right = dict(fields["gauge", schedule].named_parameters())
        for name in left:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                a = optimizers["control", schedule].state[left[name]].get(key)
                b = optimizers["gauge", schedule].state[right[name]].get(key)
                if a is not None and a.data_ptr() == b.data_ptr():
                    raise AssertionError((name, key, "Shared optimizer storage"))
    generator = torch.Generator(device=device)
    generator.set_state(payload["training_generator_state"])
    start = time.monotonic()
    manifest = {"source": str(source), "source_sha256": source_hash, "source_step": payload["step"],
                "source_config": config, "factor": factor, "steps": steps, "interval": interval,
                "schedules": list(schedules), "target": "Original sampled clean X", "same_draws": True,
                "same_initial_function_checks": checks, "moment_transport": "m/q; v/q^2; eps/q; unchanged step, LR, decay",
                "clipping_coordinates": clip_coordinates,
                "clipping": "Original clipping threshold; source-coordinate norm removes gauge-induced clipping-factor differences" if clip_coordinates == "source" else "Actual gauge-coordinate gradient norm; combined metric and clipping intervention",
                "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Real original-loss continuation; optimizer parameterization intervention, not coordinate normalization or a surrogate-error probe."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    history, metrics = [], []
    oracle, _ = make_oracle(config, fixture, device)

    def snapshot(added):
        for label in variants:
            result = copy.deepcopy(payload)
            result["step"] = payload["step"] + added
            result["config"].update(device=device, steps=payload["step"] + steps)
            result["models"] = {schedule: {n: v.detach().cpu().clone() for n, v in fields[label, schedule].state_dict().items()}
                                for schedule in schedules}
            result["optimizers"] = {schedule: copy.deepcopy(optimizers[label, schedule].state_dict()) for schedule in schedules}
            result["training_generator_state"] = generator.get_state()
            result["torch_cpu_rng_state"] = torch.get_rng_state()
            result["torch_device_rng_state"] = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
            result["examples_seen_per_schedule"] = (payload["step"] + added) * config["batch_size"]
            result["gauge_metadata"] = {"factor": variants[label],
                                        "cumulative_factor": payload.get("gauge_metadata", {}).get("cumulative_factor", 1.) * variants[label],
                                        "source_sha256": source_hash, "original_clean_X_training": True,
                                        "clip_coordinates": clip_coordinates}
            torch.save(result, output / label / f"checkpoint_step_{result['step']:06d}.pt")

    def observe(added):
        for (label, schedule), field in fields.items():
            evaluator = copy.deepcopy(field).double().eval()
            evaluator.pilot_intrinsic = config["intrinsic_dim"]
            for scale in (.1, .316227766, 1.):
                result = readout(evaluator, points.double(), scale, schedule, fixture, oracle)
                metrics.append({"step": payload["step"] + added, "variant": label, "schedule": schedule,
                                "lambda": scale, "readout": result})
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    snapshot(0)
    observe(0)
    for added in range(1, steps + 1):
        batch = draw_batch(config, fixture, generator, device)
        for (label, schedule), field in fields.items():
            record = optimizer_step(field, optimizers[label, schedule], batch, schedule, config["gradient_clip"],
                                    clip_parameter_factors=factors[label, schedule] if clip_coordinates == "source" else None)
            history.append({"step": payload["step"] + added, "variant": label, "schedule": schedule, **record})
        if added % interval == 0 or added == steps:
            observe(added)
            (output / "history.json").write_text(json.dumps(history) + "\n")
            print(json.dumps({"added_steps": added, "elapsed_seconds": time.monotonic() - start}), flush=True)
        if added in {steps // 2, steps}:
            snapshot(added)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    for optimizer in optimizers.values():
        assert all(int(state["step"]) == payload["step"] + steps for state in optimizer.state.values())
    result = {"complete": True, "source_unchanged": True, "all_optimizer_steps_correct": True,
              "elapsed_seconds": time.monotonic() - start, "final_rng_sha256": hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
              "final_model_hashes": {f"{label}/{s}": state_digest(field.state_dict()) for (label, s), field in fields.items()}}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--clip-coordinates", choices=("source", "gauge"), default="source")
    parser.add_argument("--inspect-real-steps", action="store_true")
    args = parser.parse_args()
    if args.inspect_real_steps:
        result = inspect_real_steps(args.source, args.output, device=args.device, queries=args.queries)
    else:
        if args.factor is None:
            parser.error("--factor is required for continuation")
        result = run(args.source, args.output, factor=args.factor, steps=args.steps, device=args.device,
                     interval=args.interval, queries=args.queries, clip_coordinates=args.clip_coordinates)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
