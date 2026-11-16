"""Isolated product-circle external-validity control for the paired FM trainer.

Prepare a NEW step-zero source, preserving the original model/optimizer/RNG
tensors, then train with the unchanged scalar-loss engine::

    python -m experiments.fm_scheduler_cause_torus prepare --source ORIGINAL --output NEW_SOURCE_DIR
    python -m experiments.fm_scheduler_cause_torus train --source NEW_SOURCE_DIR/checkpoint_step_000000.pt --output NEW_RUN_DIR --device cuda:0

This is not a pure curvature intervention relative to an affine fixture: its
local dimension is d, linear span is 2d, and clean radius is constant. It tests
whether the intervention depends on a single fixed linear-projection shortcut.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments import fm_scheduler_cause_training as training


LAW = "product_circles"
FIXTURE_SCHEMA = "fm_product_circles_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(ambient_dim, intrinsic_dim, seed=202609067711):
    if not 0 < intrinsic_dim or 2*intrinsic_dim > ambient_dim:
        raise ValueError("Require 0 < 2*intrinsic_dim <= ambient_dim")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(ambient_dim, 2*intrinsic_dim, generator=generator, dtype=torch.float64)
    basis = torch.linalg.qr(matrix, mode="reduced").Q
    return {"schema_version": FIXTURE_SCHEMA, "data_law": LAW,
            "ambient_dim": ambient_dim, "intrinsic_dim": intrinsic_dim,
            "circle_count": intrinsic_dim, "circle_radius": math.sqrt(ambient_dim/intrinsic_dim),
            "embedding_basis": basis, "center": torch.zeros(ambient_dim, dtype=torch.float64),
            "fixture_seed": seed, "linear_span_dimension": 2*intrinsic_dim,
            "geometry": "Product of independent uniform circles in mutually orthogonal planes"}


def _clean(fixture, count, generator, device, dtype):
    angle = 2*math.pi*torch.rand(count, fixture["circle_count"], generator=generator,
                                device=device, dtype=dtype)
    paired = float(fixture["circle_radius"])*torch.stack((angle.cos(), angle.sin()), dim=-1)
    basis = fixture["embedding_basis"].to(device=device, dtype=dtype)
    return paired.flatten(1)@basis.T + fixture["center"].to(device=device, dtype=dtype)


def sample(fixture, count, seed, device):
    """Independent, repeatable float64 evaluation draws; never advance training RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    clean = _clean(fixture, count, generator, "cpu", torch.float64)
    noise = torch.randn(count, fixture["ambient_dim"], generator=generator, dtype=torch.float64)
    return clean.to(device), noise.to(device)


def draw_batch(config, fixture, generator, device):
    """One common clean/noise/log-uniform-scale draw for all four arms."""
    count, D = config["batch_size"], config["ambient_dim"]
    clean = _clean(fixture, count, generator, device, torch.float32)
    noise = torch.randn(count, D, device=device, generator=generator)
    uniform = torch.rand(count, device=device, generator=generator)
    scales = (math.log(config["lambda_min"])+uniform*math.log(config["lambda_max"]/config["lambda_min"])).exp()
    return clean, noise, scales


@dataclass(frozen=True)
class ProductCirclePosterior:
    mean: torch.Tensor
    response: torch.Tensor
    correction: torch.Tensor
    full: torch.Tensor


def posterior(query, noise_ratio, fixture):
    """Exact factorized posterior; correction includes the ambient complement."""
    query = query.double()
    if query.shape[-1] != fixture["ambient_dim"]:
        raise ValueError("Query ambient dimension does not match fixture")
    scale = torch.as_tensor(noise_ratio, device=query.device, dtype=query.dtype)
    scale = torch.broadcast_to(scale, query.shape[:-1])
    basis = fixture["embedding_basis"].to(query)
    center = fixture["center"].to(query)
    pairs = ((query-center)@basis).reshape(*query.shape[:-1], fixture["circle_count"], 2)
    circles = circle_posterior(pairs, scale[..., None], radius=fixture["circle_radius"])
    mean = circles.mean.flatten(-2)@basis.T + center
    response = circles.response.sum(-1)
    # Summing only per-circle correction would omit all D-2d normal directions.
    correction = (mean-query).square().sum(-1)/scale.square()
    return ProductCirclePosterior(mean, response, correction, response+correction)


def tangent_energy(error, query, fixture):
    """Projection onto the d local tangent lines at a clean on-torus query."""
    basis = fixture["embedding_basis"].to(query)
    center = fixture["center"].to(query)
    shape = (*query.shape[:-1], fixture["circle_count"], 2)
    pairs = ((query-center)@basis).reshape(shape)
    radius = pairs.norm(dim=-1, keepdim=True)
    if bool((radius <= 0).any()):
        raise ValueError("A local circle tangent is undefined at its plane origin")
    normal = pairs/radius
    tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
    projected_error = (error@basis).reshape(shape)
    return (projected_error*tangent).sum(-1).square().sum(-1)


def make_oracle(config, fixture, device):
    return (lambda query, scale: posterior(query, scale, fixture),
            lambda error, query: tangent_energy(error, query, fixture))


def prepare_source(source, output, *, fixture_seed=202609067711):
    source, output = Path(source), Path(output)
    source_hash = sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("step") != 0:
        raise ValueError("Require an original pilot step-zero checkpoint")
    config = payload["config"]
    if config.get("data_law") not in {"uniform", "gaussian"}:
        raise ValueError("Prepare from a preserved affine-law source, not another derived source")
    if config.get("output_coordinate", "data") != "data" or config.get("output_normalization", "layernorm") != "layernorm":
        raise ValueError("Require the original data-output, final-LayerNorm source")
    derived = copy.deepcopy(payload)
    fixture = make_fixture(config["ambient_dim"], config["intrinsic_dim"], fixture_seed)
    derived["config"]["data_law"] = LAW
    output.mkdir(parents=True, exist_ok=False)
    torch.save(fixture, output/"data.pt")
    provenance = {"source": str(source), "source_sha256": source_hash,
                  "source_fixture_sha256": sha256(source.parent/"data.pt"),
                  "adapter_sha256": sha256(__file__), "fixture_seed": fixture_seed,
                  "circle_oracle_sha256": sha256(Path(__file__).with_name("fm_mechanism_circle_oracle.py")),
                  "trainer_sha256": sha256(Path(training.__file__)),
                  "fixture_sha256": sha256(output/"data.pt"),
                  "modified_payload_fields": ["config.data_law", "product_circle_provenance"],
                  "contract": "All original model, optimizer and private RNG tensors are copied unchanged"}
    derived["product_circle_provenance"] = provenance
    torch.save(derived, output/"checkpoint_step_000000.pt")
    provenance = {**provenance, "derived_checkpoint_sha256": sha256(output/"checkpoint_step_000000.pt")}
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if sha256(source) != source_hash:
        raise RuntimeError("Original source changed during preparation")
    return provenance


@contextmanager
def installed_adapters():
    """Process-local, exception-safe hooks; non-product laws retain original functions."""
    names = ("sample", "draw_batch", "make_oracle")
    originals = {name: getattr(training, name) for name in names}
    def sampled(fixture, *args, **kwargs):
        fn = sample if fixture.get("data_law") == LAW else originals["sample"]
        return fn(fixture, *args, **kwargs)
    def drawn(config, fixture, *args, **kwargs):
        fn = draw_batch if config.get("data_law") == LAW else originals["draw_batch"]
        return fn(config, fixture, *args, **kwargs)
    def oracle(config, fixture, *args, **kwargs):
        fn = make_oracle if config.get("data_law") == LAW else originals["make_oracle"]
        return fn(config, fixture, *args, **kwargs)
    try:
        training.sample, training.draw_batch, training.make_oracle = sampled, drawn, oracle
        yield
    finally:
        for name, function in originals.items():
            setattr(training, name, function)


def run(args):
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    if payload["config"].get("data_law") != LAW or "product_circle_provenance" not in payload:
        raise ValueError("Use a product-circle source created by this module's prepare command")
    if args.data_law is not None:
        raise ValueError("Do not override the prepared product-circle population")
    if args.output_normalization == "population":
        raise ValueError("Population-normalization training is not part of this external-validity control")
    source_provenance = payload["product_circle_provenance"]
    if sha256(args.source.parent/"data.pt") != source_provenance["fixture_sha256"]:
        raise ValueError("Prepared fixture hash does not match provenance")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = {"law": LAW, "source_provenance": source_provenance,
                "adapter_sha256": sha256(__file__),
                "circle_oracle_sha256": sha256(Path(__file__).with_name("fm_mechanism_circle_oracle.py")),
                "trainer_sha256": sha256(Path(training.__file__)),
                "local_dimension": payload["config"]["intrinsic_dim"],
                "linear_span_dimension": 2*payload["config"]["intrinsic_dim"],
                "scope": "External-validity control; not an isolated curvature intervention"}
    # Record start-time source hashes even when the engine raises after creating
    # its fresh directory; do not replace provenance on an existing run.
    try:
        with installed_adapters():
            result = training.run(args)
        manifest["status"] = "completed"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        if args.output.is_dir():
            (args.output/"product_circle_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--fixture-seed", type=int, default=202609067711)
    commands.add_parser("train", parents=[training.parser()], add_help=False)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = (prepare_source(arguments.source, arguments.output, fixture_seed=arguments.fixture_seed)
              if arguments.command == "prepare" else run(arguments))
    print(json.dumps(result), flush=True)
