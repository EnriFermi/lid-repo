"""Matched posterior-FM mechanism pilot on known affine Gaussian/uniform laws.

This is an isolated diagnostic trainer, not a replacement campaign runner.
Every schedule receives exactly the same clean samples, Gaussian noise and
physical scales at every optimizer step, from identically initialized fields.
The population data law has no ambient-normal thickness.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

from models.affine_flow import affine_schedule_state, canonical_schedule
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField
from experiments.fm_mechanism_oracles import AffineFixture


SCHEMA_VERSION = 1
SCHEDULES = ("log_noise", "rectified_linear", "vp_trigonometric")


@dataclass(frozen=True)
class PilotConfig:
    ambient_dim: int = 128
    intrinsic_dim: int = 3
    hidden_dim: int = 256
    depth: int = 3
    batch_size: int = 512
    steps: int = 8000
    seed: int = 20260905
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    lambda_min: float = 0.01
    lambda_max: float = 1.0
    validation_count: int = 2048
    log_interval: int = 200
    input_coordinate: str = "native"
    output_coordinate: str = "data"
    output_normalization: str = "layernorm"
    head_preconditioner: str = "elementwise"
    embedding: str = "random"
    data_law: str = "gaussian"
    schedules: tuple[str, ...] = SCHEDULES
    checkpoints: tuple[int, ...] = (0, 1000, 3000, 8000)
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not 0 < self.intrinsic_dim < self.ambient_dim:
            raise ValueError("Require 0 < intrinsic_dim < ambient_dim")
        if not 0 < self.lambda_min < self.lambda_max:
            raise ValueError("Require 0 < lambda_min < lambda_max")
        if self.input_coordinate not in {"native", "channel"}:
            raise ValueError("input_coordinate must be native or channel")
        if self.output_coordinate not in {"data", "native"}:
            raise ValueError("output_coordinate must be data or native")
        if self.output_normalization not in {"layernorm", "none"}:
            raise ValueError("output_normalization must be layernorm or none")
        if self.head_preconditioner not in {"elementwise", "isotropic_column"}:
            raise ValueError("head_preconditioner must be elementwise or isotropic_column")
        if self.embedding not in {"random", "axis"}:
            raise ValueError("embedding must be random or axis")
        if self.data_law not in {"gaussian", "uniform", "circle"}:
            raise ValueError("data_law must be gaussian, uniform or circle")
        if self.data_law == "circle" and (self.ambient_dim, self.intrinsic_dim) != (2, 1):
            raise ValueError("The curved circle control requires D=2, d=1")
        if min(self.steps, self.batch_size, self.validation_count, self.log_interval) <= 0:
            raise ValueError("steps, batch size and validation/log counts must be positive")
        if len(set(self.schedules)) != len(self.schedules) or not self.schedules:
            raise ValueError("schedules must be unique and nonempty")
        for schedule in self.schedules:
            if canonical_schedule(schedule) != schedule:
                raise ValueError("Use canonical schedule names")


def tensor_digest(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_fixture(config: PilotConfig) -> dict[str, Any]:
    """Return a centered affine-law fixture in population RMS-one units."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
    matrix = torch.randn(config.ambient_dim, config.ambient_dim, generator=generator, dtype=torch.float64)
    orthonormal, _ = torch.linalg.qr(matrix)
    if config.embedding == "axis":
        orthonormal = torch.eye(config.ambient_dim, dtype=torch.float64)
    basis = orthonormal[:, :config.intrinsic_dim].float()
    normal_basis = orthonormal[:, config.intrinsic_dim:].float()
    latent_std = math.sqrt(config.ambient_dim / config.intrinsic_dim)
    if config.data_law == "gaussian":
        query_latents = latent_std * torch.randn(config.validation_count, config.intrinsic_dim, generator=generator)
    elif config.data_law == "uniform":
        query_latents = latent_std * math.sqrt(3) * (2 * torch.rand(config.validation_count, config.intrinsic_dim, generator=generator) - 1)
    else:
        query_latents = 2 * math.pi * torch.rand(config.validation_count, 1, generator=generator)
    if config.data_law == "circle":
        clean = math.sqrt(2) * torch.cat((torch.cos(query_latents), torch.sin(query_latents)), dim=1)
    else:
        clean = query_latents @ basis.T
    noise = torch.randn(config.validation_count, config.ambient_dim, generator=generator)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "circle_exact_manifold" if config.data_law == "circle" else f"rank_d_{config.data_law}_exact_manifold",
        "data_law": config.data_law,
        "tangent_basis": basis,
        "normal_basis": normal_basis,
        "orthonormal_basis": orthonormal.float(),
        "center": torch.zeros(config.ambient_dim),
        "latent_std": latent_std,
        "latent_half_width": latent_std * math.sqrt(3) if config.data_law == "uniform" else None,
        "circle_radius": math.sqrt(2) if config.data_law == "circle" else None,
        "population_coordinate_rms": 1.0,
        "normal_std": 0.0,
        "embedding": config.embedding,
        "clean_queries": clean,
        "query_noise": noise,
        "query_latents": clean @ basis,
        "evaluation_noise_ratios": torch.tensor([0.001, 0.003, 0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1, 0.177827941, 0.316227766, 0.562341325, 1.0]),
        "evaluation_extrapolation_below_training_min": [0.001, 0.003],
    }


def canonical_posterior(
    field: ScaleConditionedNeuralField,
    canonical_inputs: Tensor,
    noise_ratio: Tensor | float,
    schedule: str,
    *,
    input_coordinate: str = "native",
    output_coordinate: str | None = None,
) -> Tensor:
    """Evaluate B(alpha*r, log(lambda)); derivatives w.r.t. r are canonical."""
    lam = torch.as_tensor(noise_ratio, device=canonical_inputs.device, dtype=canonical_inputs.dtype)
    if lam.ndim == 0:
        lam = lam.expand(canonical_inputs.shape[0])
    if lam.shape != (canonical_inputs.shape[0],):
        raise ValueError("noise_ratio must be scalar or have one value per input")
    if input_coordinate == "native":
        alpha = affine_schedule_state(lam, schedule).alpha
    elif input_coordinate == "channel":
        alpha = torch.ones_like(lam)
    else:
        raise ValueError("input_coordinate must be native or channel")
    prediction = field(alpha[:, None] * canonical_inputs, lam.log())
    output_coordinate = output_coordinate or getattr(field, "pilot_output_coordinate", "data")
    if output_coordinate == "native":
        schedule_alpha = affine_schedule_state(lam, schedule).alpha
        prediction = prediction / schedule_alpha[:, None]
    elif output_coordinate != "data":
        raise ValueError("output_coordinate must be data or native")
    return prediction


def gaussian_posterior(inputs: Tensor, noise_ratio: Tensor | float, fixture: dict[str, Any]) -> Tensor:
    basis = fixture["tangent_basis"].to(device=inputs.device, dtype=inputs.dtype)
    lam = torch.as_tensor(noise_ratio, device=inputs.device, dtype=inputs.dtype)
    variance = float(fixture["latent_std"]) ** 2
    gain = variance / (variance + lam.square())
    if gain.ndim:
        gain = gain[:, None]
    return ((inputs @ basis) * gain) @ basis.T


def population_posterior(inputs: Tensor, noise_ratio: Tensor | float, fixture: dict[str, Any]) -> Tensor:
    if fixture.get("data_law", "gaussian") == "gaussian":
        return gaussian_posterior(inputs, noise_ratio, fixture)
    if fixture["data_law"] == "circle":
        r = inputs.double()
        radius = r.norm(dim=-1, keepdim=True)
        lam = torch.as_tensor(noise_ratio, dtype=r.dtype, device=r.device)
        if lam.ndim:
            lam = lam[:, None]
        kappa = float(fixture["circle_radius"]) * radius / lam.square()
        ratio = torch.special.i1e(kappa) / torch.special.i0e(kappa)
        return float(fixture["circle_radius"]) * ratio * r / radius.clamp_min(1e-100)
    basis = fixture["tangent_basis"].to(device=inputs.device, dtype=torch.float64)
    population = AffineFixture(
        "uniform", basis, fixture["center"].to(device=inputs.device, dtype=torch.float64),
        torch.full((basis.shape[1],), fixture["latent_half_width"], device=inputs.device, dtype=torch.float64),
    )
    return population.posterior(inputs, noise_ratio).mean


def _new_fields(config: PilotConfig) -> tuple[dict[str, ScaleConditionedNeuralField], NeuralFieldConfig]:
    torch.manual_seed(config.seed)
    architecture = NeuralFieldConfig(
        ambient_dim=config.ambient_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        condition_dim=64,
        fourier_features=32,
        max_condition_frequency=100.0,
        dropout=0.0,
        condition_transform="linear",
        zero_init_output=True,
    )
    template = ScaleConditionedNeuralField(architecture)
    # Construct the identical baseline first: all common initialization tensors
    # and the RNG stream must survive this one-layer architecture intervention.
    if config.output_normalization == "none":
        template.output[0] = torch.nn.Identity()
    fields = {schedule: copy.deepcopy(template).to(config.device) for schedule in config.schedules}
    for field in fields.values():
        field.pilot_output_coordinate = config.output_coordinate
    return fields, architecture


class ColumnIsotropicAdamW(torch.optim.Optimizer):
    """AdamW for a linear head with output-isotropic second moments.

For weights [output, hidden], each hidden column shares one denominator over
output coordinates. A bias vector shares one scalar denominator. First moments,
bias corrections and decoupled weight decay follow AdamW. This preserves any
linear output subspace containing the initial weights and every gradient.
"""

    def __init__(self, parameters: Any, *, lr: float, weight_decay: float) -> None:
        super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999), "eps": 1e-8})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        result = None
        if closure is not None:
            with torch.enable_grad():
                result = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.ndim not in (1, 2):
                    raise ValueError("Head parameters must be a matrix or bias vector")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(gradient.square().mean(dim=0, keepdim=True))
                state["step"] += 1
                first, second = state["exp_avg"], state["exp_avg_sq"]
                first.lerp_(gradient, 1 - beta1)
                second.mul_(beta2).add_(gradient.square().mean(dim=0, keepdim=True), alpha=1 - beta2)
                denominator = second.sqrt() / math.sqrt(1 - beta2 ** state["step"])
                denominator.add_(group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(first, denominator, value=-group["lr"] / (1 - beta1 ** state["step"]))
        return result


class SplitHeadOptimizer:
    """Keep standard hidden AdamW and the isolated head intervention together."""

    def __init__(self, field: ScaleConditionedNeuralField, config: PilotConfig) -> None:
        head = field.output[-1]
        head_parameters = list(head.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        hidden = [parameter for parameter in field.parameters() if id(parameter) not in head_ids]
        self.hidden = torch.optim.AdamW(hidden, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.head = ColumnIsotropicAdamW(head_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.hidden.zero_grad(set_to_none=set_to_none)
        self.head.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.hidden.step()
        self.head.step()

    def state_dict(self) -> dict[str, Any]:
        return {"hidden": self.hidden.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.hidden.load_state_dict(state["hidden"])
        self.head.load_state_dict(state["head"])


def make_optimizer(field: ScaleConditionedNeuralField, config: PilotConfig) -> Any:
    if config.head_preconditioner == "isotropic_column":
        return SplitHeadOptimizer(field, config)
    return torch.optim.AdamW(field.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def load_pilot_checkpoint(
    path: str | Path,
    schedule: str,
    device: str | torch.device = "cpu",
) -> tuple[ScaleConditionedNeuralField, dict[str, Any]]:
    """Load one independent field; payload contains config, states and metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported pilot checkpoint schema")
    field = ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    if payload["config"].get("output_normalization", "layernorm") == "none":
        field.output[0] = torch.nn.Identity()
    field.load_state_dict(payload["models"][schedule], strict=True)
    field.pilot_output_coordinate = payload["config"].get("output_coordinate", "data")
    field.to(device).eval()
    return field, payload


@torch.no_grad()
def value_validation(
    fields: dict[str, ScaleConditionedNeuralField],
    fixture: dict[str, Any],
    config: PilotConfig,
    step: int,
) -> list[dict[str, float | str | int]]:
    """Fixed-query value errors; no claim about derivative accuracy follows."""
    clean = fixture["clean_queries"].to(config.device)
    noise = fixture["query_noise"].to(config.device)
    basis = fixture["tangent_basis"].to(config.device)
    scales = fixture["evaluation_noise_ratios"]
    rows: list[dict[str, float | str | int]] = []
    for schedule, field in fields.items():
        field.eval()
        for scale in scales.tolist():
            noisy = clean + scale * noise
            oracle_noisy = population_posterior(noisy, scale, fixture)
            oracle_clean = population_posterior(clean, scale, fixture)
            pred_noisy = canonical_posterior(field, noisy, scale, schedule, input_coordinate=config.input_coordinate)
            pred_clean = canonical_posterior(field, clean, scale, schedule, input_coordinate=config.input_coordinate)
            row: dict[str, float | str | int] = {"step": step, "schedule": schedule, "lambda": scale}
            row["outside_training_range"] = bool(scale < config.lambda_min * (1 - 1e-6) or scale > config.lambda_max * (1 + 1e-6))
            row["noisy_target_weighted_mse"] = float((pred_noisy - clean).square().mean() / scale**2)
            row["oracle_target_weighted_mse"] = float((oracle_noisy - clean).square().mean() / scale**2)
            row["clean_correction_mean"] = float((pred_clean - clean).square().sum(-1).mean() / scale**2)
            row["oracle_clean_correction_mean"] = float((oracle_clean - clean).square().sum(-1).mean() / scale**2)
            for where, prediction, oracle in (("noisy", pred_noisy, oracle_noisy), ("clean", pred_clean, oracle_clean)):
                error = prediction - oracle
                error_basis = basis.to(dtype=error.dtype)
                if fixture.get("data_law") == "circle":
                    radial = clean.to(dtype=error.dtype) / float(fixture["circle_radius"])
                    local_tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=1)
                    tangent = (error * local_tangent).sum(-1, keepdim=True) * local_tangent
                else:
                    tangent = (error @ error_basis) @ error_basis.T
                normal = error - tangent
                row[f"{where}_posterior_mse"] = float(error.square().mean())
                row[f"{where}_tangent_error_energy_over_lambda2"] = float(tangent.square().sum(-1).mean() / scale**2)
                row[f"{where}_normal_error_energy_over_lambda2"] = float(normal.square().sum(-1).mean() / scale**2)
            rows.append(row)
        field.train()
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    output: Path, step: int, fields: dict[str, ScaleConditionedNeuralField],
    optimizers: dict[str, Any], generator: torch.Generator,
    config: PilotConfig, architecture: NeuralFieldConfig,
) -> Path:
    path = output / f"checkpoint_step_{step:06d}.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "config": asdict(config),
        "architecture": architecture.to_dict(),
        "models": {name: {key: value.detach().cpu() for key, value in field.state_dict().items()} for name, field in fields.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "training_generator_state": generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_device_rng_state": torch.cuda.get_rng_state(config.device) if config.device.startswith("cuda") else None,
        "fixture_file": "data.pt",
        "examples_seen_per_schedule": step * config.batch_size,
    }
    torch.save(payload, path)
    return path


def train_pilot(output: str | Path, config: PilotConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    output = Path(output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        fixture = make_fixture(config)
        torch.save(fixture, output / "data.pt")
    else:
        fixture = torch.load(output / "data.pt", map_location="cpu", weights_only=False)
    fields, architecture = _new_fields(config)
    optimizers = {name: make_optimizer(field, config) for name, field in fields.items()}
    generator = torch.Generator(device=config.device).manual_seed(config.seed + 202)
    start_step = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        old_config = payload["config"]
        for key, value in asdict(config).items():
            original_value = old_config.get(key, getattr(PilotConfig(), key))
            if key not in {"steps", "checkpoints", "log_interval"} and original_value != value:
                raise ValueError(f"Resume cannot change {key}: {original_value} -> {value}")
        for schedule, field in fields.items():
            field.load_state_dict(payload["models"][schedule])
            optimizers[schedule].load_state_dict(payload["optimizers"][schedule])
        generator.set_state(payload["training_generator_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        if payload["torch_device_rng_state"] is not None:
            torch.cuda.set_rng_state(payload["torch_device_rng_state"], config.device)
        start_step = int(payload["step"])
        if (output / "history.json").exists():
            history = [row for row in json.loads((output / "history.json").read_text()) if row["step"] <= start_step]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": f"matched-posterior-fm-exact-{config.data_law}",
        "config": asdict(config),
        "architecture": architecture.to_dict(),
        "parameter_count_per_schedule": sum(value.numel() for value in next(iter(fields.values())).parameters()),
        "initial_or_resumed_state_hashes": {name: state_digest(field.state_dict()) for name, field in fields.items()},
        "basis_sha256": tensor_digest(fixture["tangent_basis"]),
        "query_sha256": tensor_digest(fixture["clean_queries"]),
        "torch_version": torch.__version__,
        "training_data": f"Online iid {config.data_law} samples; same draws across schedules; exact noiseless support",
        "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loss": "mean_batch(mean_coordinate((canonical_posterior(X+lambda*Z,lambda)-X)^2)/lambda^2)",
        "native_output_interpretation": "H predicts alpha_s*X; dividing H by alpha_s in the loss gives the identical physical posterior risk",
        "resumed_from": str(resume) if resume else None,
        "claim_boundary": "Fixed-seed paired mechanism pilot; no retraining-uncertainty inference",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    basis = fixture["tangent_basis"].to(config.device)
    latent_std = float(fixture["latent_std"])
    checkpoints = set(config.checkpoints) | {config.steps}
    elapsed_start = time.monotonic()
    if start_step == 0 and resume is None:
        _save_checkpoint(output, 0, fields, optimizers, generator, config, architecture)
        _write_csv(output / "validation_step_000000.csv", value_validation(fields, fixture, config, 0))
    accumulated = {name: 0.0 for name in fields}
    accumulated_steps = 0
    for step in range(start_step + 1, config.steps + 1):
        if config.data_law == "gaussian":
            latent = torch.randn(config.batch_size, config.intrinsic_dim, device=config.device, generator=generator)
        elif config.data_law == "uniform":
            latent = math.sqrt(3) * (2 * torch.rand(config.batch_size, config.intrinsic_dim, device=config.device, generator=generator) - 1)
        else:
            latent = 2 * math.pi * torch.rand(config.batch_size, 1, device=config.device, generator=generator)
        if config.data_law == "circle":
            clean = math.sqrt(2) * torch.cat((torch.cos(latent), torch.sin(latent)), dim=1)
        else:
            clean = (latent_std * latent) @ basis.T
        noise = torch.randn(config.batch_size, config.ambient_dim, device=config.device, generator=generator)
        uniforms = torch.rand(config.batch_size, device=config.device, generator=generator)
        lam = (math.log(config.lambda_min) + uniforms * math.log(config.lambda_max / config.lambda_min)).exp()
        canonical_inputs = clean + lam[:, None] * noise
        for schedule, field in fields.items():
            optimizer = optimizers[schedule]
            optimizer.zero_grad(set_to_none=True)
            prediction = canonical_posterior(field, canonical_inputs, lam, schedule, input_coordinate=config.input_coordinate)
            loss = ((prediction - clean).square().mean(-1) / lam.square()).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at step={step}, schedule={schedule}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), config.gradient_clip)
            optimizer.step()
            accumulated[schedule] += float(loss.detach())
        accumulated_steps += 1
        if step % config.log_interval == 0 or step in checkpoints:
            elapsed = time.monotonic() - elapsed_start
            record: dict[str, Any] = {"step": step, "examples_seen_per_schedule": step * config.batch_size, "elapsed_seconds_this_invocation": elapsed}
            record.update({f"loss_{name}": value / accumulated_steps for name, value in accumulated.items()})
            history.append(record)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            _write_csv(output / "training_history.csv", history)
            (output / "status.json").write_text(json.dumps({"status": "running", **record}, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            accumulated = {name: 0.0 for name in fields}
            accumulated_steps = 0
        if step in checkpoints:
            _save_checkpoint(output, step, fields, optimizers, generator, config, architecture)
            validation = value_validation(fields, fixture, config, step)
            _write_csv(output / f"validation_step_{step:06d}.csv", validation)
    summary = {"schema_version": SCHEMA_VERSION, "status": "complete", "step": config.steps,
               "elapsed_seconds_this_invocation": time.monotonic() - elapsed_start,
               "examples_seen_per_schedule": config.steps * config.batch_size,
               "final_state_hashes": {name: state_digest(field.state_dict()) for name, field in fields.items()},
               "checkpoints": [path.name for path in sorted(output.glob("checkpoint_step_*.pt"))]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    defaults = PilotConfig()
    for name in ("ambient_dim", "intrinsic_dim", "hidden_dim", "depth", "batch_size", "steps", "seed", "validation_count", "log_interval"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "gradient_clip", "lambda_min", "lambda_max"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--input-coordinate", choices=("native", "channel"), default="native")
    parser.add_argument("--output-coordinate", choices=("data", "native"), default="data")
    parser.add_argument("--output-normalization", choices=("layernorm", "none"), default="layernorm")
    parser.add_argument("--head-preconditioner", choices=("elementwise", "isotropic_column"), default="elementwise")
    parser.add_argument("--embedding", choices=("random", "axis"), default="random")
    parser.add_argument("--data-law", choices=("gaussian", "uniform", "circle"), default="gaussian")
    parser.add_argument("--schedules", nargs="+", default=list(SCHEDULES), choices=SCHEDULES)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(defaults.checkpoints))
    values = vars(parser.parse_args())
    output, resume = values.pop("output"), values.pop("resume")
    values["schedules"] = tuple(values["schedules"])
    values["checkpoints"] = tuple(values["checkpoints"])
    train_pilot(output, PilotConfig(**values), resume=resume)


if __name__ == "__main__":
    main()
