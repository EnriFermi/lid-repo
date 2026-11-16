"""Frozen paired late-LR continuation on the existing Uniform and Circle fits.

No new architecture, schedule, target, or data normalization is introduced.
Every optimizer owns its restored step and moment tensors independently.
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

from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, state_digest


SCALES = (.1, .316227766, 1.)
LEARNING_RATES = {"lr1e3": .001, "lr1e4": .0001}


def independent_optimizer(field, source_state, *, learning_rate, weight_decay):
    optimizer = torch.optim.AdamW(field.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(copy.deepcopy(source_state))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return optimizer


def sample_batch(config, fixture, generator, device):
    """Keep operation ordering identical to fm_mechanism_pilot.py."""
    if config["data_law"] == "uniform":
        latent = math.sqrt(3) * (2 * torch.rand(config["batch_size"], config["intrinsic_dim"],
                                               device=device, generator=generator) - 1)
        clean = (float(fixture["latent_std"]) * latent) @ fixture["tangent_basis"].to(device).T
    elif config["data_law"] == "circle":
        latent = 2 * math.pi * torch.rand(config["batch_size"], 1, device=device, generator=generator)
        clean = math.sqrt(2) * torch.cat((torch.cos(latent), torch.sin(latent)), dim=1)
    else:
        raise ValueError("This frozen generalization control supports only Uniform and Circle")
    noise = torch.randn(config["batch_size"], config["ambient_dim"], device=device, generator=generator)
    uniforms = torch.rand(config["batch_size"], device=device, generator=generator)
    lam = (math.log(config["lambda_min"]) + uniforms * math.log(config["lambda_max"] / config["lambda_min"])).exp()
    return clean, noise, lam


def verify_independence(optimizers, schedules):
    checks = 0
    for schedule in schedules:
        left = list(optimizers["lr1e3", schedule].state.values())
        right = list(optimizers["lr1e4", schedule].state.values())
        for a, b in zip(left, right, strict=True):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(a[key], b[key])
                assert a[key].data_ptr() != b[key].data_ptr(), (schedule, key)
                checks += 1
    return checks


def run(source: Path, output: Path, *, device="cuda:1", steps=1000, queries=32, interval=20):
    if steps <= 0 or steps % 2 or interval <= 0:
        raise ValueError("steps must be positive/even; interval must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent / "data.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    assert config["data_law"] in {"uniform", "circle"}
    assert config.get("head_preconditioner", "elementwise") == "elementwise"
    assert config.get("output_coordinate", "data") == "data"
    assert config.get("input_coordinate", "native") == "native"
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    start_step = payload["step"]
    schedules = tuple(payload["models"])
    fields, optimizers, averages = {}, {}, {}
    for variant, lr in LEARNING_RATES.items():
        directory = output / variant
        directory.mkdir()
        torch.save(fixture, directory / "data.pt")
        for schedule in schedules:
            key = variant, schedule
            fields[key], _ = load_pilot_checkpoint(source, schedule, device)
            fields[key].train()
            optimizers[key] = independent_optimizer(fields[key], payload["optimizers"][schedule],
                learning_rate=lr, weight_decay=config["weight_decay"])
            assert state_digest(fields[key].state_dict()) == state_digest(payload["models"][schedule])
    independence_checks = verify_independence(optimizers, schedules)
    generator = torch.Generator(device=device)
    generator.set_state(payload["training_generator_state"])
    probes = fixture["clean_queries"][:queries].to(device)
    torch.save({"queries": probes.cpu(), "query_noise": fixture["query_noise"][:queries],
                "scales": SCALES}, output / "probes.pt")
    manifest = {"source": str(source), "source_sha256": source_hash,
                "source_config": config, "added_steps": steps, "source_step": start_step,
                "learning_rates": LEARNING_RATES, "device": device,
                "query_count": len(probes), "scales": SCALES, "telemetry_interval": interval,
                "parameter_average_added_steps": [steps//2+1, steps],
                "independent_state_tensor_checks": independence_checks,
                "initial_weights_exactly_source": True,
                "same_private_draw_stream": True, "target": "Original clean X",
                "protocol": "Paired continuation; only learning rate differs. Tail average is inference-only.",
                "frozen_before_outcomes": True, "torch_version": torch.__version__,
                "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    observations, history = [], []
    started = time.monotonic()

    def snapshot(added_step, averaged=False):
        for variant, lr in LEARNING_RATES.items():
            directory = output / (variant + ("_tailavg" if averaged else ""))
            if averaged:
                directory.mkdir(exist_ok=False)
                torch.save(fixture, directory / "data.pt")
            result = copy.deepcopy(payload)
            result["step"] = start_step + added_step
            result["config"].update(device=device, steps=start_step+steps, learning_rate=lr)
            result["models"] = {s: {k: v.detach().cpu() for k, v in
                (averages[variant, s] if averaged else fields[variant, s].state_dict()).items()}
                for s in schedules}
            result["optimizers"] = {} if averaged else {
                s: copy.deepcopy(optimizers[variant, s].state_dict()) for s in schedules}
            if not averaged:
                assert all(float(st["step"]) == start_step+added_step
                           for o in result["optimizers"].values() for st in o["state"].values())
            result["training_generator_state"] = generator.get_state()
            result["torch_cpu_rng_state"] = torch.get_rng_state()
            result["torch_device_rng_state"] = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
            result["examples_seen_per_schedule"] = (start_step+added_step)*config["batch_size"]
            result["probe_metadata"] = {"source_sha256": source_hash, "averaged": averaged,
                "target_mode": "clean_X", "inference_only_no_optimizer_state": averaged,
                "average_added_steps": [steps//2+1, steps] if averaged else None}
            torch.save(result, directory / f"checkpoint_step_{start_step+added_step:06d}.pt")

    @torch.no_grad()
    def observe(added_step):
        for (variant, schedule), field in fields.items():
            for lam in SCALES:
                value = canonical_posterior(field, probes, lam, schedule,
                    input_coordinate=config["input_coordinate"],
                    output_coordinate=config.get("output_coordinate", "data"))
                observations.append({"step": start_step+added_step, "variant": variant,
                    "schedule": schedule, "lambda": lam, "values": value.cpu()})

    snapshot(0)
    observe(0)
    for step in range(1, steps+1):
        clean, noise, lam = sample_batch(config, fixture, generator, device)
        inputs = clean + lam[:, None]*noise
        for key, field in fields.items():
            optimizer = optimizers[key]
            optimizer.zero_grad(set_to_none=True)
            prediction = canonical_posterior(field, inputs, lam, key[1],
                input_coordinate=config["input_coordinate"],
                output_coordinate=config.get("output_coordinate", "data"))
            loss = ((prediction-clean).square().mean(-1)/lam.square()).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError((step, key))
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(field.parameters(), config["gradient_clip"])
            optimizer.step()
            history.append({"step": start_step+step, "variant": key[0], "schedule": key[1],
                            "loss": float(loss), "unclipped_gradient_norm": float(gradnorm)})
            if step > steps//2:
                with torch.no_grad():
                    if key not in averages:
                        averages[key] = {k: v.detach().clone() for k,v in field.state_dict().items()}
                    else:
                        count = step-steps//2
                        for name, value in field.state_dict().items():
                            if value.is_floating_point():
                                averages[key][name].add_((value-averages[key][name])/count)
        if step % interval == 0 or step == steps:
            observe(step)
        if step in {steps//2, steps}:
            snapshot(step)
        if step % 200 == 0 or step == steps:
            print(json.dumps({"data_law": config["data_law"], "added_step": step,
                              "elapsed": time.monotonic()-started}), flush=True)
    snapshot(steps, averaged=True)
    torch.save({"queries": probes.cpu(), "observations": observations,
                "source_manifest": manifest}, output / "paired_telemetry.pt")
    (output / "history.json").write_text(json.dumps(history)+"\n")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    summary = {"status": "complete", "elapsed_seconds": time.monotonic()-started,
               "final_step": start_step+steps, "source_unchanged": True,
               "all_optimizer_steps_correct": True,
               "independent_state_tensor_checks": independence_checks,
               "telemetry_observations": len(observations),
               "final_private_rng_sha256": hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
