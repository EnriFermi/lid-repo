"""Time-boxed paired LR canary on exact-archive rendered geometries.

This is a reduced-width fresh-training diagnostic, not a historical campaign
resume. Native posterior FM objective and conditioning are unchanged. Every
continuation has independent Adam state and identical batches/noise/scales.
"""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn

from datasets.archive import EXACT_ARCHIVE_PASSWORD, verify_exact_archive
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def posterior(model: nn.Module, points: Tensor, lam: Tensor | float) -> Tensor:
    scale = torch.as_tensor(lam, device=points.device, dtype=points.dtype)
    if scale.ndim == 0:
        scale = scale.expand(len(points))
    return model(points / (1 + scale[:, None]), scale.log())


def exact_trace(model: nn.Module, points: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    """Exact batch-separable VJPs, chunked across output coordinates."""
    values, traces = [], []
    for batch in points.split(8):
        inputs = batch.detach().requires_grad_(True)
        outputs = posterior(model, inputs, lam)
        trace = torch.zeros(len(batch), device=batch.device, dtype=batch.dtype)
        dimension = batch.shape[1]
        for start in range(0, dimension, 32):
            indices = torch.arange(start, min(start + 32, dimension), device=batch.device)
            probes = torch.zeros(len(indices), len(batch), dimension,
                                 device=batch.device, dtype=batch.dtype)
            probes[torch.arange(len(indices), device=batch.device), :, indices] = 1
            gradients = torch.autograd.grad(
                outputs, inputs, grad_outputs=probes, is_grads_batched=True,
                retain_graph=start + 32 < dimension,
            )[0]
            trace += gradients[torch.arange(len(indices), device=batch.device), :, indices].sum(0)
        values.append(outputs.detach())
        traces.append(trace.detach())
    return torch.cat(values), torch.cat(traces)


def check_trace() -> dict[str, float]:
    class Linear(nn.Module):
        def forward(self, x: Tensor, condition: Tensor) -> Tensor:
            return 2 * x

    points = torch.randn(3, 7, dtype=torch.float64)
    _, trace = exact_trace(Linear(), points, .5)
    error = float((trace - 7 * 2 / 1.5).abs().max())
    assert error < 1e-12
    return {"linear_trace_max_error": error}


def prepare(cfg: DictConfig, out: Path) -> tuple[Path, dict]:
    archive = Path(cfg.archive).resolve()
    verified = verify_exact_archive(archive)
    data_root = out / "exact_subset"
    data_root.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        selected = []
        for dataset in cfg.datasets:
            for split in ("train", "val", "test"):
                for filename in ("dataset.npy", "lid.npy"):
                    suffix = f"/{dataset}/{split}/{filename}"
                    matches = [name for name in names if name.endswith(suffix)]
                    if len(matches) != 1:
                        raise ValueError(f"Missing or ambiguous archive member {suffix}")
                    selected.extend(matches)
    subprocess.run(["unzip", "-qq", "-P", EXACT_ARCHIVE_PASSWORD,
                    str(archive), *selected, "-d", str(data_root)], check=True)
    roots = list(data_root.rglob(str(cfg.datasets[0])))
    if len(roots) != 1:
        raise ValueError("Ambiguous extracted dataset root")
    manifest = {
        "archive_sha256": verified.archive_sha256,
        "archive_size": verified.archive_size_bytes,
        "exact_archive_subset_not_regenerated": True,
        "files": {name: digest(data_root / name) for name in selected},
    }
    write_json(out / "data_manifest.json", manifest)
    return roots[0].parent, manifest


def native_risk(model: nn.Module, queries: Tensor, cfg: DictConfig) -> list[float]:
    generator = torch.Generator(device=queries.device).manual_seed(int(cfg.seed) + 91)
    noise = torch.randn(queries.shape, generator=generator, device=queries.device)
    model.eval()
    with torch.no_grad():
        return [float(((posterior(model, queries + float(lam) * noise, float(lam))
                        - queries) / float(lam)).square().mean()) for lam in cfg.scales]


def checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
               generator: torch.Generator, step: int, cfg: DictConfig,
               mean: Tensor, rms: Tensor) -> None:
    torch.save({
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "generator_state": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "step": step, "examples_seen": step * int(cfg.batch_size),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mean": mean.cpu(), "rms": rms.cpu(),
    }, path)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, clean: Tensor,
               generator: torch.Generator, cfg: DictConfig) -> float:
    model.train()
    indices = torch.randint(len(clean), (int(cfg.batch_size),),
                            generator=generator, device=clean.device)
    x = clean[indices]
    u = torch.rand(len(x), generator=generator, device=x.device)
    scale = torch.exp(np.log(float(cfg.lambda_min))
                      + u * np.log(float(cfg.lambda_max) / float(cfg.lambda_min)))
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    loss = ((posterior(model, x + scale[:, None] * noise, scale) - x)
            / scale[:, None]).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip))
    optimizer.step()
    return float(loss.detach())


def evaluate(model: nn.Module, queries: Tensor, cfg: DictConfig) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"response": [], "full": [], "correction": []}
    for lam in cfg.scales:
        value, response = exact_trace(model, queries, float(lam))
        correction = ((value.double() - queries.double()) / float(lam)).square().sum(-1)
        rows["response"].append(response.double().cpu().numpy())
        rows["full"].append((response.double() + correction).cpu().numpy())
        rows["correction"].append(correction.cpu().numpy())
    return {key: np.stack(value) for key, value in rows.items()}


def run_dataset(name: str, root: Path, cfg: DictConfig, out: Path) -> dict:
    start = time.monotonic()
    destination = out / name
    destination.mkdir(exist_ok=False)
    raw = np.load(root / name / "train/dataset.npy", mmap_mode="r")
    raw = raw.reshape(len(raw), -1)
    rng = np.random.default_rng(int(cfg.seed))
    order = rng.permutation(len(raw))
    selection_ids = order[:int(cfg.holdout_count)]
    fit_ids = order[int(cfg.holdout_count):]
    train_cpu = torch.from_numpy(np.array(raw[fit_ids], dtype=np.float32))
    mean = train_cpu.double().mean(0).float()
    rms = (train_cpu.double() - mean.double()).square().mean().sqrt().float()
    train = ((train_cpu - mean) / rms).to(str(cfg.device))
    selection = ((torch.from_numpy(np.array(raw[selection_ids], dtype=np.float32))
                  - mean) / rms).to(str(cfg.device))
    test_raw = np.load(root / name / "test/dataset.npy")
    test_raw = test_raw.reshape(len(test_raw), -1)
    test = ((torch.from_numpy(test_raw[:int(cfg.query_count)].astype(np.float32))
             - mean) / rms).to(str(cfg.device))
    source_labels = np.unique(np.load(root / name / "train/lid.npy")).tolist()
    if source_labels != ([20] if "spaghetti" in name else [1]):
        raise ValueError(f"Unexpected stored labels: {source_labels}")
    np.savez(destination / "split_ids.npz", fit=fit_ids, selection=selection_ids,
             test=np.arange(len(test)))
    torch.manual_seed(int(cfg.seed))
    architecture = NeuralFieldConfig(
        ambient_dim=train.shape[1], hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth), condition_dim=int(cfg.condition_dim),
        zero_init_output=True,
    )
    model = ScaleConditionedNeuralField(architecture).to(str(cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate),
                                 weight_decay=float(cfg.weight_decay))
    generator = torch.Generator(device=str(cfg.device)).manual_seed(int(cfg.seed) + 5)
    history = []
    deadline = datetime.fromisoformat(str(cfg.deadline_utc)).timestamp()

    def log(stage: str, step: int, loss: float) -> None:
        risks = native_risk(model, selection[:int(cfg.risk_count)], cfg)
        row = {"stage": stage, "step": step, "train_loss": loss,
               "risk_by_scale": risks, "mean_grid_risk": float(np.mean(risks)),
               "seconds": time.monotonic() - start}
        history.append(row)
        write_json(destination / "history.json", history)
        print(json.dumps({"dataset": name, **row}), flush=True)

    for step in range(1, int(cfg.base_steps) + 1):
        loss = train_step(model, optimizer, train, generator, cfg)
        if step % int(cfg.log_interval) == 0:
            log("base", step, loss)
            checkpoint(destination / "progress.pt", model, optimizer, generator,
                       step, cfg, mean, rms)
            if time.time() > deadline:
                raise TimeoutError("Time box reached; progress saved")
    checkpoint(destination / "base.pt", model, optimizer, generator,
               int(cfg.base_steps), cfg, mean, rms)
    base_model = copy.deepcopy(model.state_dict())
    base_optimizer = copy.deepcopy(optimizer.state_dict())
    base_rng = generator.get_state().clone()
    saved = {"base": copy.deepcopy(base_model)}
    continuation_states = {}
    for stage, factor in (("same_lr", 1.), ("lower_lr", .1)):
        model.load_state_dict(copy.deepcopy(base_model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate),
                                     weight_decay=float(cfg.weight_decay))
        optimizer.load_state_dict(copy.deepcopy(base_optimizer))
        for group in optimizer.param_groups:
            group["lr"] = float(cfg.learning_rate) * factor
        generator.set_state(base_rng.clone())
        averaged = {key: value.clone() for key, value in base_model.items()}
        average_count = 0
        for extra in range(1, int(cfg.continuation_steps) + 1):
            loss = train_step(model, optimizer, train, generator, cfg)
            if extra > int(cfg.continuation_steps) - int(cfg.average_steps):
                average_count += 1
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        averaged[key].add_((value - averaged[key]) / average_count)
            if extra % int(cfg.log_interval) == 0:
                step = int(cfg.base_steps) + extra
                log(stage, step, loss)
                checkpoint(destination / f"{stage}.pt", model, optimizer, generator,
                           step, cfg, mean, rms)
                if time.time() > deadline:
                    raise TimeoutError("Time box reached; continuation saved")
        saved[stage] = copy.deepcopy(model.state_dict())
        saved[stage + "_average"] = averaged
        torch.save({"model_state": averaged, "architecture": architecture.to_dict(),
                    "averaged_updates": average_count}, destination / f"{stage}_average.pt")
        continuation_states[stage] = generator.get_state().clone()
        counters = [int(value["step"]) for value in optimizer.state.values()]
        assert set(counters) == {int(cfg.base_steps) + int(cfg.continuation_steps)}
        assert {int(value["step"]) for value in base_optimizer["state"].values()} == {int(cfg.base_steps)}
    assert torch.equal(continuation_states["same_lr"], continuation_states["lower_lr"])

    results = []
    # All policies are declared before test access; no policy chosen by test MAE.
    for stage, state in saved.items():
        model.load_state_dict(state)
        arrays = {}
        for split, queries in (("selection", selection[:int(cfg.query_count)]), ("test", test)):
            predicted = evaluate(model, queries, cfg)
            arrays.update({f"{split}_{key}": value for key, value in predicted.items()})
        np.savez(destination / f"{stage}_predictions.npz", **arrays)
        risk = native_risk(model, selection[:int(cfg.risk_count)], cfg)
        for readout in ("response", "full"):
            selection_mae = np.abs(arrays[f"selection_{readout}"] - 1).mean(1)
            test_mae = np.abs(arrays[f"test_{readout}"] - 1).mean(1)
            selected = int(np.argmin(selection_mae))
            row = {"dataset": name, "stage": stage, "readout": readout,
                   "selected_scale": float(cfg.scales[selected]),
                   "boundary_scale": selected in (0, len(cfg.scales) - 1),
                   "selection_mae": float(selection_mae[selected]),
                   "test_mae": float(test_mae[selected]),
                   "selection_mae_curve": selection_mae.tolist(),
                   "test_mae_curve": test_mae.tolist(),
                   "risk_by_scale": risk, "mean_grid_risk": float(np.mean(risk))}
            results.append(row)
            print(json.dumps(row), flush=True)
        write_json(destination / "results.json", results)
    result = {"dataset": name, "true_lid": 1, "stored_lid_values": source_labels,
              "representation": "exact_archive_rendered", "ambient_dim": train.shape[1],
              "parameter_count": sum(value.numel() for value in model.parameters()),
              "normalization_rms": float(rms), "fit_count": len(train),
              "selection_count": len(selection), "query_count": len(test),
              "constant_one_mae": 0., "seconds": time.monotonic() - start,
              "paired_rng_verified": True, "independent_adam_counters_verified": True,
              "results": results}
    write_json(destination / "summary.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="spiral_spaghetti_convergence")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg.output).resolve()
    # Hydra has created only its configuration subdirectory here.
    if (out / "data_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory; runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "protocol.json", {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scope": "fresh reduced-width FM rectified; not historical checkpoint continuation",
        "loss": "mean(((posterior(alpha*(X+lambda*Z),log(lambda))-X)/lambda)^2)",
        "alpha": "1/(1+lambda)", "true_lid": 1,
        "selection": "64 held-out source-train queries; independently frozen per readout/policy",
        "primary_readout": "response", "secondary_readout": "full",
        "trace_backend": "exact chunked batched VJP; no stochastic trace probes",
        "fixed_baseline": "LID=1, exact for both datasets; never a stopping criterion",
        "trace_test": check_trace(), "source_sha256": digest(Path(__file__)),
    })
    root, _ = prepare(cfg, out)
    summaries = []
    for name in cfg.datasets:
        summaries.append(run_dataset(str(name), root, cfg, out))
        write_json(out / "summary.json", summaries)
    with (out / "summary.csv").open("w", newline="") as stream:
        keys = ["dataset", "stage", "readout", "selected_scale", "boundary_scale",
                "selection_mae", "test_mae", "mean_grid_risk"]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerows(summary["results"])


if __name__ == "__main__":
    main()
