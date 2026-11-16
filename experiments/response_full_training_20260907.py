"""Paired VP posterior/noise training on the pinned complex coefficient laws.

The canonical scalar objective, physical scale/noise/data draws, trainable
architecture and initial parameters are shared. The posterior parameterization
is the intervention; the predicted clean functions are not equal at step zero.
No labels or oracle posterior values are used for training or checkpoint choice.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import Tensor

from experiments.ve_known_lid_recoverability_campaign import make_splits, DATASET_SPECS
from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


KINDS = ("posterior", "noise")


@dataclass(frozen=True)
class Config:
    dataset: str
    output: str
    device: str = "cuda:0"
    seed: int = 20260907
    train_count: int = 100000
    selection_count: int = 1000
    test_count: int = 1000
    steps: int = 16000
    batch_size: int = 1024
    hidden_dim: int = 512
    depth: int = 4
    learning_rate: float = 0.001
    final_learning_rate: float = 0.00002
    lambda_min: float = 0.01
    lambda_max: float = 1.0
    average_steps: int = 1000
    query_count: int = 256
    probe_count: int = 512
    temporal_interval: int = 25


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def posterior(field, r: Tensor, lam: Tensor | float, kind: str) -> Tensor:
    scale = torch.as_tensor(lam, dtype=r.dtype, device=r.device)
    if scale.ndim == 0:
        scale = scale.expand(len(r))
    raw = field(r / torch.sqrt(1 + scale[:, None].square()), scale.log())
    if kind == "posterior":
        return raw
    if kind == "noise":
        return r - scale[:, None] * raw
    raise ValueError(kind)


def objective(field, x: Tensor, z: Tensor, lam: Tensor, kind: str) -> Tensor:
    b = posterior(field, x + lam[:, None] * z, lam, kind)
    return ((b - x).square().mean(-1) / lam.square()).mean()


def readout(field, x: Tensor, lam: float, kind: str) -> tuple[Tensor, Tensor, Tensor]:
    raw, tr = fast_value_and_trace(field, x, lam, "vp_trigonometric", output_coordinate="data")
    if kind == "posterior":
        b, response = raw, tr
    elif kind == "noise":
        b, response = x - lam * raw, x.shape[1] - lam * tr
    else:
        raise ValueError(kind)
    correction = (b - x).square().sum(-1) / lam**2
    return b, response, correction


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_state(path, fields, optimizers, rng, step, cfg, architecture, mean, rms, averages=None):
    torch.save({
        "schema": "response_full_complex_v1", "config": asdict(cfg),
        "architecture": asdict(architecture), "step": step,
        "models": {name: cpu_state(model) for name, model in fields.items()},
        "optimizers": {name: copy.deepcopy(opt.state_dict()) for name, opt in optimizers.items()},
        "generator_state": rng.get_state(), "normalization_mean": mean.cpu(),
        "normalization_scale": rms, "averages": averages,
    }, path)


def run(cfg: Config) -> None:
    if cfg.steps < 1 or cfg.average_steps < 1 or cfg.average_steps > cfg.steps:
        raise ValueError("invalid step/average budget")
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    train_np, selection_np, test_np = make_splits(cfg.dataset, train_size=cfg.train_count,
        validation_size=cfg.selection_count, test_size=cfg.test_count, data_seed=0)
    train_raw = torch.from_numpy(train_np)
    mean = train_raw.mean(0)
    rms = float((train_raw - mean).square().mean().sqrt())
    train = ((train_raw - mean) / rms).to(cfg.device)
    ids = np.random.default_rng(cfg.seed + 101).choice(cfg.test_count, cfg.query_count, replace=False)
    queries = ((torch.from_numpy(test_np[ids]) - mean) / rms).to(cfg.device)
    validation = ((torch.from_numpy(selection_np[:cfg.probe_count]) - mean) / rms).to(cfg.device)
    fixture = {"mean": mean, "rms": rms, "query_ids": ids,
               "query_raw": test_np[ids], "queries": queries.cpu(),
               "validation": validation.cpu(), "dataset": cfg.dataset,
               "split_hashes": {"train": tensor_sha(train_raw),
                   "selection": tensor_sha(torch.from_numpy(selection_np)),
                   "test": tensor_sha(torch.from_numpy(test_np))}}
    torch.save(fixture, out / "fixture.pt")
    arch = NeuralFieldConfig(ambient_dim=30, hidden_dim=cfg.hidden_dim, depth=cfg.depth,
        condition_dim=64, fourier_features=32, condition_transform="linear")
    initial = ScaleConditionedNeuralField(arch)
    fields = {name: copy.deepcopy(initial).to(cfg.device) for name in KINDS}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                    weight_decay=1e-4) for name, model in fields.items()}
    rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 1)
    eval_rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 2)
    eval_noise = torch.randn(validation.shape, generator=eval_rng, device=cfg.device)
    scales = [0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1,
              0.177827941, 0.316227766, 0.562341325, 1.0]
    sources = [Path(__file__), Path("models/neural_fields.py"),
               Path("experiments/fm_mechanism_fast_trace.py"),
               Path("experiments/ve_known_lid_recoverability_campaign.py")]
    manifest = {"schema": "response_full_complex_v1", "config": asdict(cfg),
        "architecture": asdict(arch), "source_hashes": {str(p): sha(p) for p in sources},
        "source_law": "pinned upstream regenerated coefficients, not exact archive",
        "physical_objective": "E ||b(X+lambda Z,lambda)-X||²/(D lambda²)",
        "paired_data_noise_scales": True, "same_initial_parameters": True,
        "same_initial_function": False, "scale_grid": scales,
        "checkpoint_policy": "fixed last step and fixed last-window parameter average; no LID selection",
        "query_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "dataset": asdict(DATASET_SPECS[cfg.dataset]), "normalization_scale": rms}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_state(out / "step_000000.pt", fields, optimizers, rng, 0, cfg, arch, mean, rms)
    average = {name: {k: torch.zeros_like(v, dtype=torch.float64, device="cpu")
                for k, v in model.state_dict().items()} for name, model in fields.items()}
    history, temporal = [], {name: [] for name in KINDS}
    temporal_steps = []
    save_steps = {min(1000, cfg.steps), min(4000, cfg.steps), min(8000, cfg.steps), cfg.steps}
    for step in range(1, cfg.steps + 1):
        progress = (step - 1) / max(1, cfg.steps - 1)
        lr = cfg.final_learning_rate + .5 * (cfg.learning_rate-cfg.final_learning_rate) * (1+math.cos(math.pi*progress))
        index = torch.randint(len(train), (cfg.batch_size,), generator=rng, device=cfg.device)
        x = train[index]
        u = torch.rand(cfg.batch_size, generator=rng, device=cfg.device)
        lam = (math.log(cfg.lambda_min) + u*math.log(cfg.lambda_max/cfg.lambda_min)).exp()
        z = torch.randn(x.shape, generator=rng, device=cfg.device)
        row = {"step": step, "lr": lr}
        for name, model in fields.items():
            model.train()
            optimizer = optimizers[name]
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x, z, lam, name)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} step{step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            row[name] = float(loss.detach())
            row[name + "_grad_norm"] = float(grad_norm)
            if step > cfg.steps - cfg.average_steps:
                for key, value in model.state_dict().items():
                    average[name][key].add_(value.detach().cpu().double(), alpha=1/cfg.average_steps)
        if step > cfg.steps - cfg.average_steps and step % cfg.temporal_interval == 0:
            temporal_steps.append(step)
            with torch.no_grad():
                for name, model in fields.items():
                    model.eval()
                    temporal[name].append(torch.stack([posterior(model, queries, s, name).cpu() for s in scales]))
        if step % 250 == 0 or step == cfg.steps:
            with torch.no_grad():
                for name, model in fields.items():
                    model.eval()
                    row[name + "_validation_risk"] = [float(objective(model, validation, eval_noise,
                        torch.full((len(validation),), s, device=cfg.device), name)) for s in scales]
            history.append(row)
            (out / "status.json").write_text(json.dumps({"status":"training", "step":step,
                "elapsed_seconds":time.monotonic()-start, "latest":row}, indent=2)+"\n")
            print(json.dumps(row), flush=True)
        if step in save_steps:
            save_state(out / f"step_{step:06d}.pt", fields, optimizers, rng, step, cfg, arch, mean, rms)
    averages = {name: {key: value.float() for key,value in state.items()} for name,state in average.items()}
    save_state(out / "final.pt", fields, optimizers, rng, cfg.steps, cfg, arch, mean, rms, averages)
    torch.save({"steps":temporal_steps, "scales":scales,
                "predictions":{name:torch.stack(values) if values else torch.empty(0)
                   for name,values in temporal.items()}}, out / "temporal.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    (out / "status.json").write_text(json.dumps({"status":"training_complete",
        "step":cfg.steps, "elapsed_seconds":time.monotonic()-start}, indent=2)+"\n")
    arrays = {"queries":queries.cpu().numpy(), "query_ids":ids, "scales":np.asarray(scales)}
    metric_rows = []
    for name, field in fields.items():
        for mode in ["last", "average"]:
            model = copy.deepcopy(field).double().eval()
            if mode == "average":
                model.load_state_dict(averages[name])
            values, responses, corrections = [], [], []
            with torch.no_grad():
                for scale in scales:
                    chunks = [readout(model, q.double(), scale, name) for q in queries.split(32)]
                    b, r, c = [torch.cat([item[i] for item in chunks]).cpu().numpy() for i in range(3)]
                    values.append(b); responses.append(r); corrections.append(c)
                    metric_rows.append({"kind":name,"mode":mode,"lambda":scale,
                        "response_mean":float(r.mean()),"correction_mean":float(c.mean()),
                        "response_mae":float(np.abs(r-DATASET_SPECS[cfg.dataset].true_lid).mean()),
                        "full_mae":float(np.abs(r+c-DATASET_SPECS[cfg.dataset].true_lid).mean())})
            for suffix, value in [("b",values),("response",responses),("correction",corrections)]:
                arrays[f"{name}_{mode}_{suffix}"] = np.asarray(value)
    np.savez_compressed(out / "predictions.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metric_rows, indent=2)+"\n")
    (out / "status.json").write_text(json.dumps({"status":"complete", "step":cfg.steps,
        "elapsed_seconds":time.monotonic()-start,
        "source_unchanged": all(sha(Path(p))==h for p,h in manifest["source_hashes"].items())},indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--average-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260907)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
