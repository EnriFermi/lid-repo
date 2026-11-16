"""Paired exact noise-energy moment regularization of preserved VP posteriors.

The source LAST weights, AdamW moments/counters and private draw stream are
continued. Both branches promote the same original float32 sampler/data to
float64 before every forward/loss/backward. Only the added scalar moment loss
differs. No model inference/selection belongs to this training-only producer.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor

from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.response_full_training_20260907 import cpu_state, objective, sha, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import make_splits
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


ARMS = ("control", "moment")


@dataclass(frozen=True)
class Config:
    source: str
    output: str
    device: str = "cuda:0"
    steps: int = 2048
    batch_size: int = 1024
    regularizer_batch_size: int = 64
    regularizer_microbatch_size: int = 64
    average_steps: int = 512
    gamma: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    log_interval: int = 32

    def validate(self):
        if not 1 <= self.average_steps <= self.steps:
            raise ValueError("Require 1 <= average_steps <= steps")
        if not 1 <= self.regularizer_batch_size <= self.batch_size:
            raise ValueError("Regularizer uses the fixed first rows of the FM batch")
        if self.regularizer_microbatch_size < 1 or self.log_interval < 1:
            raise ValueError("Positive microbatch and logging intervals required")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay != 1e-4 or self.clip_norm != 1.0:
            raise ValueError("Preserve the declared original decay and clip")


def moment_residual(field, x: Tensor, z: Tensor, lam: Tensor):
    """Exact F(Y)-||Z||² without subtracting the ambient noise energy.

    trace is with respect to canonical Y, holding each lambda fixed. The
    analytic propagation retains all parameter gradients, including those
    through the trace. No detach or stochastic trace is used here.
    """
    y = x + lam[:, None]*z
    b, response = fast_value_and_trace(field, y, lam, "vp_trigonometric",
                                      output_coordinate="data")
    clean_residual = (b-x)/lam[:, None]
    residual = response + clean_residual.square().sum(-1) - 2*(z*clean_residual).sum(-1)
    return residual, response


def moment_loss(field, x, z, lam):
    residual, _ = moment_residual(field, x, z, lam)
    return residual.square().mean()/x.shape[1]


def draw_batch(train32: Tensor, rng: torch.Generator, batch_size: int,
               lambda_min: float = .01, lambda_max: float = 1.):
    """Replay the source sampler's dtype, device, order and RNG consumption."""
    if train32.dtype != torch.float32:
        raise ValueError("Keep original normalized float32 training points before promotion")
    index = torch.randint(len(train32), (batch_size,), generator=rng, device=train32.device)
    x = train32[index]
    u = torch.rand(batch_size, generator=rng, device=train32.device, dtype=torch.float32)
    lam = (math.log(lambda_min)+u*math.log(lambda_max/lambda_min)).exp()
    z = torch.randn(x.shape, generator=rng, device=train32.device, dtype=torch.float32)
    return x.double(), z.double(), lam.double(), index


def paired_state(payload, device, learning_rate=2e-5, weight_decay=1e-4):
    """Independent double copies of LAST model and its compatible Adam state."""
    arch = NeuralFieldConfig.from_mapping(payload["architecture"])
    if arch.dropout != 0 or arch.condition_transform != "linear":
        raise ValueError("This exact trace continuation requires source dropout0 and log-lambda input")
    with torch.random.fork_rng(devices=[]):
        template = ScaleConditionedNeuralField(arch).double()
    template.load_state_dict(payload["models"]["posterior"])
    models = {arm:copy.deepcopy(template).to(device) for arm in ARMS}
    optimizers = {}
    for arm, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizers"]["posterior"]))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    # Noncapturable Adam's scalar step remains on its source
                    # device (CPU); moments already follow double parameters.
                    state[key] = value.to(dtype=torch.float64)
        optimizers[arm] = optimizer
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), payload["models"]["posterior"][name].double().cpu(),
                                       rtol=0, atol=0)
        saved = optimizer.state_dict()["state"]
        for index, original in payload["optimizers"]["posterior"]["state"].items():
            for key, value in original.items():
                if torch.is_tensor(value):
                    expected = value.double() if value.is_floating_point() else value
                    torch.testing.assert_close(saved[index][key].cpu(), expected.cpu(), rtol=0, atol=0)
    return models, optimizers


def restored_generator(payload, device):
    source_type = torch.device(payload["config"]["device"]).type
    if torch.device(device).type != source_type:
        raise ValueError("Cannot reinterpret a CUDA RNG state as CPU; use the source device family")
    rng = torch.Generator(device=device)
    rng.set_state(payload["generator_state"].cpu())
    assert torch.equal(rng.get_state().cpu(), payload["generator_state"].cpu())
    return rng


def gradient_norm(field):
    return torch.stack([parameter.grad.detach().square().sum()
                        for parameter in field.parameters() if parameter.grad is not None]).sum().sqrt()


def training_step(field, optimizer, x, z, lam, *, gamma, regularizer_batch_size,
                  regularizer_microbatch_size, clip_norm=1.0):
    """One ordinary accumulated-loss backward and one clipped AdamW update."""
    field.train()
    optimizer.zero_grad(set_to_none=True)
    fm = objective(field, x, z, lam, "posterior")
    if not torch.isfinite(fm):
        raise FloatingPointError("Nonfinite original FM loss")
    fm.backward()
    fm_norm = float(gradient_norm(field))
    rho2_sum, response_sum, residual_sum = 0., 0., 0.
    for start in range(0, regularizer_batch_size, regularizer_microbatch_size):
        stop = min(start+regularizer_microbatch_size, regularizer_batch_size)
        # Control computes the same diagnostic at the pre-update model, but
        # its gamma is exactly zero and no regularizer backward is performed.
        with torch.set_grad_enabled(gamma != 0):
            residual, response = moment_residual(field, x[start:stop], z[start:stop], lam[start:stop])
            part = residual.square().sum()/(regularizer_batch_size*x.shape[1])
            if not torch.isfinite(part):
                raise FloatingPointError("Nonfinite exact moment loss")
            if gamma != 0:
                (gamma*part).backward()
        rho2_sum += float(residual.detach().square().sum())
        response_sum += float(response.detach().sum())
        residual_sum += float(residual.detach().sum())
    unweighted = rho2_sum/(regularizer_batch_size*x.shape[1])
    norm = float(torch.nn.utils.clip_grad_norm_(field.parameters(), clip_norm, error_if_nonfinite=True))
    optimizer.step()
    return dict(fm=float(fm.detach()), moment=unweighted, applied_gamma=gamma,
        weighted_moment=gamma*unweighted, total=float(fm.detach())+gamma*unweighted,
        fm_gradient_norm=fm_norm, preclip_gradient_norm=norm, clipped=norm > clip_norm,
        clip_factor=min(1., clip_norm/(norm+1e-6)),
        response_mean=response_sum/regularizer_batch_size,
        residual_mean=residual_sum/regularizer_batch_size,
        regularizer_query_count=regularizer_batch_size)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(path, fields, optimizers, rng, step, cfg, source, source_hash,
                    fixture_path, fixture_hash, averages=None, average_count=0):
    torch.save(dict(schema="response_full_moment_continuation_v1", config=asdict(cfg),
        source_config=source["config"], source_step=source["step"], continuation_step=step,
        step=source["step"]+step, architecture=source["architecture"],
        source_checkpoint=cfg.source, source_checkpoint_sha256=source_hash,
        source_fixture=str(fixture_path), source_fixture_sha256=fixture_hash,
        models={arm:cpu_state(model) for arm,model in fields.items()},
        optimizers={arm:cpu_tree(opt.state_dict()) for arm,opt in optimizers.items()},
        generator_state=rng.get_state().cpu(), normalization_mean=source["normalization_mean"].cpu(),
        normalization_scale=source["normalization_scale"], averages=averages,
        average_count=average_count), path)


def prepare_training_data(payload, fixture, device):
    cfg = payload["config"]
    train, validation, test = make_splits(cfg["dataset"], train_size=cfg["train_count"],
        validation_size=cfg["selection_count"], test_size=cfg["test_count"], data_seed=0)
    split_hashes = {name:tensor_sha(torch.from_numpy(value)) for name,value in
        (("train",train), ("selection",validation), ("test",test))}
    assert split_hashes == fixture["split_hashes"]
    raw = torch.from_numpy(train)
    mean = raw.mean(0)
    rms = float((raw-mean).square().mean().sqrt())
    torch.testing.assert_close(mean, fixture["mean"], rtol=0, atol=0)
    torch.testing.assert_close(mean, payload["normalization_mean"], rtol=0, atol=0)
    assert rms == fixture["rms"] == payload["normalization_scale"]
    normalized = (raw-mean)/rms
    return normalized.to(device), split_hashes, tensor_sha(normalized)


def run(cfg: Config):
    cfg.validate()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, source_path = Path(cfg.output), Path(cfg.source)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; no checkpoint/result overwrites")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload["schema"] != "response_full_complex_v1" or payload["config"]["dataset"] not in (
            "e1_spiral_pca", "e6_exp_pca"):
        raise ValueError("Only the preserved Spiral/Exp VP checkpoints are in scope")
    if (payload["config"]["lambda_min"], payload["config"]["lambda_max"]) != (.01, 1.):
        raise ValueError("Preserve the original physical scale support")
    fixture_path = source_path.parent/"fixture.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    train, split_hashes, train_hash = prepare_training_data(payload, fixture, cfg.device)
    fields, optimizers = paired_state(payload, cfg.device, cfg.learning_rate, cfg.weight_decay)
    rng = restored_generator(payload, cfg.device)
    source_hash, fixture_hash = sha(source_path), sha(fixture_path)
    sources = [Path(__file__), source_path, fixture_path, Path("models/neural_fields.py")]
    sources += [Path("experiments")/name for name in (
        "response_full_training_20260907.py", "fm_mechanism_fast_trace.py",
        "fm_mechanism_diagnostics.py", "ve_known_lid_recoverability_campaign.py")]
    hashes = {str(path):sha(path) for path in sources}
    manifest = dict(completed=False, schema="response_full_moment_continuation_v1", config=asdict(cfg),
        dataset=payload["config"]["dataset"], architecture=payload["architecture"], sources=hashes,
        source_step=payload["step"], split_hashes=split_hashes, normalized_train_sha256=train_hash,
        state_contract="Source models.posterior LAST and matching full AdamW state, never averages; both promoted float64",
        optimizer_contract="All moments and counters preserved, independent copies; both fixed LR/decay/clip",
        random_contract="Source private generator restored; original Float32 randint, uniform-logscale, Gaussian order, then X/Z/lambda promoted Float64; one shared draw per update",
        objective="FM E||b(X+lambda Z)-X||²/(Dlambda²) + applied_gamma E[(F(Y)-||Z||²)²]/D",
        moment_contract="Stable traceJb+||(b-X)/lambda||²-2<Z,(b-X)/lambda>; exact canonical trace and ordinary parameter backward, no detach or Hutchinson",
        subset_contract="Fixed first regularizer_batch_size rows; microbatch sums divided by total regularizer count, not chunk means",
        training_scales=[.01,1.], dtype="Float64 models/moments/counters/loss/backward; originalFloat32 data+random sampling preserved",
        checkpoint_policy="Initial, fixed midpoint, final; arithmetic final-window PARAMETER average with unchanged buffers; no LID evaluation/selection",
        initial_generator_state_sha256=tensor_sha(rng.get_state()))
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save_checkpoint(out/"initial.pt", fields, optimizers, rng, 0, cfg, payload, source_hash, fixture_path, fixture_hash)
    averages = {arm:{name:torch.zeros_like(value, dtype=torch.float64, device="cpu")
                    for name,value in model.named_parameters()} for arm,model in fields.items()}
    history, clipped = [], {arm:0 for arm in ARMS}
    count = 0
    for step in range(1, cfg.steps+1):
        x, z, lam, index = draw_batch(train, rng, cfg.batch_size)
        row = dict(continuation_step=step, absolute_step=payload["step"]+step)
        for arm in ARMS:
            row[arm] = training_step(fields[arm], optimizers[arm], x, z, lam,
                gamma=cfg.gamma if arm == "moment" else 0., regularizer_batch_size=cfg.regularizer_batch_size,
                regularizer_microbatch_size=cfg.regularizer_microbatch_size, clip_norm=cfg.clip_norm)
            clipped[arm] += int(row[arm]["clipped"])
        history.append(row)
        if step > cfg.steps-cfg.average_steps:
            count += 1
            for arm, model in fields.items():
                for name,value in model.named_parameters():
                    averages[arm][name].add_(value.detach().cpu())
        if step == max(1, cfg.steps//2):
            save_checkpoint(out/"mid.pt", fields, optimizers, rng, step, cfg, payload, source_hash, fixture_path, fixture_hash)
        if step == 1 or step % cfg.log_interval == 0 or step == cfg.steps:
            (out/"history.json").write_text(json.dumps(history, indent=2)+"\n")
            status = dict(status="training", completed=False, continuation_step=step, total_steps=cfg.steps,
                elapsed_seconds=time.monotonic()-started, clipped_counts=clipped, latest=row)
            (out/"status.json").write_text(json.dumps(status, indent=2)+"\n")
            print(json.dumps(status), flush=True)
    assert count == cfg.average_steps
    states = {}
    for arm,model in fields.items():
        states[arm] = cpu_state(model)
        for name,total in averages[arm].items():
            states[arm][name] = total/count
        for name, value in model.named_buffers():
            torch.testing.assert_close(value.cpu(), payload["models"]["posterior"][name].double().cpu(), rtol=0, atol=0)
    save_checkpoint(out/"final.pt", fields, optimizers, rng, cfg.steps, cfg, payload, source_hash, fixture_path, fixture_hash,
                    averages=states, average_count=count)
    assert all(sha(Path(path)) == expected for path,expected in hashes.items())
    manifest.update(completed=True, all_sources_unchanged=True, elapsed_seconds=time.monotonic()-started,
        final_generator_state_sha256=tensor_sha(rng.get_state()), average_count=count,
        clipped_counts=clipped, final_absolute_step=payload["step"]+cfg.steps)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"status.json").write_text(json.dumps(dict(status="complete", **manifest), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--regularizer-batch-size", type=int, default=64)
    parser.add_argument("--regularizer-microbatch-size", type=int, default=64)
    parser.add_argument("--average-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--log-interval", type=int, default=32)
    run(Config(**vars(parser.parse_args())))
