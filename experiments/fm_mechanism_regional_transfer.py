"""Decompose one population-risk step by observed noisy-input regions.

The input-defined partition preserves the posterior-target gradient identity.
Using the latent clean X to define the partition would not preserve it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_mechanism_gradient_transfer import sample, adam_metric, inner_product, masked_direction
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint, population_posterior
from experiments.fm_mechanism_readout_transfer import readout


def observed_regions(points, fixture, scale):
    if fixture.get("data_law") != "uniform":
        raise ValueError("Regional decomposition is specified only for Uniform")
    basis = torch.linalg.qr(fixture["tangent_basis"].to(points)).Q
    distance = (float(fixture["latent_half_width"]) - (points @ basis).abs()).min(-1).values
    return {"interior": distance > 3*scale, "boundary": distance < scale,
            "transition": (distance >= scale) & (distance <= 3*scale)}


def run(source, output, device="cuda:0"):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    fixture = torch.load(source.parent/"data.pt", map_location="cpu", weights_only=False)
    oracle, _ = make_oracle(payload["config"], fixture, device)
    scale = .316227766
    train_x, train_z = sample(fixture, 2048, 20260905+7101, device)
    queries, _ = sample(fixture, 512, 20260905+7102, device)
    inputs = train_x + scale*train_z
    target = population_posterior(inputs, scale, fixture).detach()
    regions = observed_regions(inputs, fixture, scale)
    assert torch.equal(sum(m.int() for m in regions.values()), torch.ones(len(inputs), device=device, dtype=torch.int))
    rows, checks = [], []
    for schedule in ("rectified_linear", "vp_trigonometric"):
        field, _ = load_pilot_checkpoint(source, schedule, device)
        field = field.double().eval()
        field.pilot_intrinsic = payload["config"]["intrinsic_dim"]
        names, params, metric = adam_metric(field, payload["optimizers"][schedule])
        backup = [p.detach().clone() for p in params]
        gradients = {}
        for region, selected in {"all": torch.ones(len(inputs), device=device, dtype=torch.bool), **regions}.items():
            field.zero_grad(set_to_none=True)
            for begin in range(0, len(inputs), 256):
                prediction = canonical_posterior(field, inputs[begin:begin+256], scale, schedule)
                energy = (prediction-target[begin:begin+256]).square().mean(-1)/scale**2
                (energy.mul(selected[begin:begin+256]).sum()/len(inputs)).backward()
            gradients[region] = [p.grad.detach().clone() for p in params]
        summed = [sum(parts) for parts in zip(*(gradients[r] for r in regions))]
        residual = math.sqrt(sum(float((a-b).square().sum()) for a,b in zip(summed,gradients["all"])))
        norm = math.sqrt(inner_product(gradients["all"], gradients["all"]))
        assert residual/max(norm,1e-30) < 1e-10
        full_direction = [m*g for m,g in zip(metric,gradients["all"])]
        eta = 1e-6*math.sqrt(inner_product(backup,backup)/inner_product(full_direction,full_direction))
        before = readout(field, queries, scale, schedule, fixture, oracle)
        with torch.no_grad():
            base_prediction = canonical_posterior(field,inputs,scale,schedule)
            base_energy = (base_prediction-target).square().mean(-1)/scale**2
        for source_region, gradient in gradients.items():
            for block in ("all", "conditioning_only", "without_conditioning"):
                delta = masked_direction(names,[m*g for m,g in zip(metric,gradient)],block)
                with torch.no_grad():
                    for p,saved,change in zip(params,backup,delta):
                        p.copy_(saved-eta*change)
                    new_prediction = canonical_posterior(field,inputs,scale,schedule)
                    new_energy = (new_prediction-target).square().mean(-1)/scale**2
                after = readout(field,queries,scale,schedule,fixture,oracle)
                for stratum in before:
                    rows.append({"schedule":schedule,"source_region":source_region,"block_mask":block,"eval_region":stratum,
                        "step_multiplier":eta,"all_source_risk_change":float((new_energy-base_energy).mean()),
                        **{f"source_risk_change_{r}":float((new_energy-base_energy)[m].sum()/len(inputs)) for r,m in regions.items()},
                        **{f"change_{k}":after[stratum][k]-before[stratum][k] for k in before[stratum] if k!="count"}})
                with torch.no_grad():
                    for p,saved in zip(params,backup):p.copy_(saved)
        assert all(torch.equal(p,saved) for p,saved in zip(params,backup))
        checks.append({"schedule":schedule,"gradient_sum_relative_error":residual/max(norm,1e-30)})
    with (output/"regional_steps.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"scale":scale,
        "partition":"OBSERVED noisy input tangent coordinates, not latent X; preserves oracle-gradient identity",
        "counts":{r:int(m.sum()) for r,m in regions.items()},"normalization":"Every component normalized by all2048 examples; gradients add exactly",
        "step":"same full-gradient eta across all regional components and block masks; frozen Adam-v metric; restored after each",
        "checks":checks,"scope":"Local regional causal contribution; not full resumed Adam training"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    run(**vars(parser.parse_args()))
