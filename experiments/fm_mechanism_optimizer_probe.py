"""Paired late-learning-rate and parameter-averaging control on fixed FM fits."""

import copy
import argparse
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-verify", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:1"
    root = Path("artifacts/fm_mechanism_sep05")
    source = root / "gaussian_native_adamw/checkpoint_step_008000.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent / "data.pt", map_location="cpu", weights_only=False)
    cfg = payload["config"]
    schedules = tuple(payload["models"])
    variants = {"lr1e3_oracle": 0.001} if args.oracle_only else {"lr1e3": 0.001, "lr1e4": 0.0001}
    fields, optimizers, averages, directories = {}, {}, {}, {}
    for variant, lr in variants.items():
        directory = root / ("gaussian_resume_" + variant + "_adamw" + args.run_suffix)
        directory.mkdir(exist_ok=args.replay_verify)
        directories[variant] = directory
        if not (directory / "data.pt").exists():
            torch.save(fixture, directory / "data.pt")
        for schedule in schedules:
            key = (variant, schedule)
            fields[key], _ = load_pilot_checkpoint(source, schedule, device)
            fields[key].train()
            optimizer = torch.optim.AdamW(fields[key].parameters(), lr=lr, weight_decay=cfg["weight_decay"])
            optimizer.load_state_dict(copy.deepcopy(payload["optimizers"][schedule]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizers[key] = optimizer
        (directory / "manifest.json").write_text(json.dumps({
            "source": str(source), "steps_added": 1000, "learning_rate": lr,
            "control": "Identical source weights, Adam moments and 1000 shared batches; only late learning rate differs",
            "target": "Exact conditional posterior; same population-gradient mean, reduced target noise" if args.oracle_only else "Original clean X",
            "tail_average": "Uniform parameter average of added steps 501--1000, inference artifact only",
        }, indent=2) + "\n")
    generator = torch.Generator(device=device)
    generator.set_state(payload["training_generator_state"])
    basis = fixture["tangent_basis"].to(device)
    clean_queries = fixture["clean_queries"][:32].to(device)
    probes = torch.cat((clean_queries, -clean_queries, clean_queries.new_zeros(1, cfg["ambient_dim"])))
    observations = []
    telemetry = []
    started = time.monotonic()
    for step in range(1, 1001):
        latent = torch.randn(cfg["batch_size"], cfg["intrinsic_dim"], device=device, generator=generator)
        clean = (fixture["latent_std"] * latent) @ basis.T
        noise = torch.randn(clean.shape, device=device, generator=generator)
        uniform = torch.rand(cfg["batch_size"], device=device, generator=generator)
        lam = (math.log(cfg["lambda_min"]) + uniform * math.log(cfg["lambda_max"] / cfg["lambda_min"])).exp()
        inputs = clean + lam[:, None] * noise
        target = clean
        if args.oracle_only:
            variance = fixture["latent_std"] ** 2
            target = ((inputs @ basis) @ basis.T) * (variance / (variance + lam.square()))[:, None]
        for key, field in fields.items():
            variant, schedule = key
            optimizer = optimizers[key]
            optimizer.zero_grad(set_to_none=True)
            prediction = canonical_posterior(field, inputs, lam, schedule)
            loss = ((prediction - target).square().mean(-1) / lam.square()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {key} at {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), cfg["gradient_clip"])
            optimizer.step()
            if step > 500:
                with torch.no_grad():
                    current = field.state_dict()
                    if key not in averages:
                        averages[key] = {name: tensor.detach().clone() for name, tensor in current.items()}
                    else:
                        for name, tensor in current.items():
                            if tensor.is_floating_point():
                                averages[key][name].add_((tensor - averages[key][name]) / (step - 500))
            if step % 20 == 0:
                with torch.no_grad():
                    for scale in (0.01, 0.316227766, 1.0):
                        values = canonical_posterior(field, probes, scale, schedule)
                        projection = (values - probes) @ basis
                        correction = projection[:-1].square().sum(-1) / scale**2
                        telemetry.append({"step": step + 8000, "variant": variant,
                                          "schedule": schedule, "lambda": scale,
                                          "tangent_correction_mean": float(correction.mean()),
                                          "center_tangent_squared_over_lambda2": float((values[-1] @ basis).square().sum()/scale**2)})
                        observations.append({"step": step + 8000, "variant": variant,
                                             "schedule": schedule, "lambda": scale,
                                             "values": values.detach().cpu()})
        if step % 200 == 0:
            print(json.dumps({"added_step": step, "elapsed": time.monotonic() - started}), flush=True)
    for variant, lr in variants.items():
        for averaged in (False, True):
            directory = directories[variant]
            if averaged:
                directory = root / ("gaussian_resume_" + variant + "_tailavg" + args.run_suffix)
                directory.mkdir(exist_ok=args.replay_verify)
                if not (directory / "data.pt").exists():
                    torch.save(fixture, directory / "data.pt")
            result = copy.deepcopy(payload)
            result["step"] = 9000
            result["config"].update(device=device, steps=9000, learning_rate=lr)
            result["models"] = {
                schedule: {name: value.detach().cpu() for name, value in
                           (averages[(variant, schedule)] if averaged else fields[(variant, schedule)].state_dict()).items()}
                for schedule in schedules
            }
            result["optimizers"] = {} if averaged else {s: optimizers[(variant, s)].state_dict() for s in schedules}
            if not averaged:
                assert all(float(state["step"]) == 9000 for s in schedules
                           for state in result["optimizers"][s]["state"].values() if "step" in state)
            result["training_generator_state"] = generator.get_state()
            result["torch_cpu_rng_state"] = torch.get_rng_state()
            result["torch_device_rng_state"] = torch.cuda.get_rng_state(device)
            result["examples_seen_per_schedule"] = 9000 * cfg["batch_size"]
            result["probe_metadata"] = {"source": str(source), "averaged": averaged,
                                        "target_mode": "conditional_mean" if args.oracle_only else "clean_X",
                                        "average_added_steps": [501, 1000] if averaged else None,
                                        "inference_only_no_optimizer_state": averaged}
            checkpoint = directory / "checkpoint_step_009000.pt"
            if checkpoint.exists():
                assert args.replay_verify
                previous = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert all(torch.equal(value, previous["models"][s][name])
                           for s in schedules for name, value in result["models"][s].items())
                assert torch.equal(previous["training_generator_state"], result["training_generator_state"])
            else:
                torch.save(result, checkpoint)
            (directory / "summary.json").write_text(json.dumps({"status": "complete", **result["probe_metadata"]}, indent=2) + "\n")
    telemetry_path = directories[next(iter(variants))] / "paired_telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n")
    torch.save({"queries": probes.cpu(), "observations": observations}, telemetry_path.with_suffix(".pt"))
    print(json.dumps({"status": "complete", "elapsed": time.monotonic() - started, "telemetry": str(telemetry_path)}), flush=True)


if __name__ == "__main__":
    main()
