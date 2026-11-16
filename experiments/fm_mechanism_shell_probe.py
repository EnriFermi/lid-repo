"""Frozen posterior errors along normal-noise shells, with exact oracles.

For affine populations, posterior values do not depend on normal noisy
coordinates. Sweeping only those coordinates therefore changes model inputs
without changing the correct posterior target. Circle uses its nonlinear
oracle instead of importing this affine invariance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_oracles import AffineFixture
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint


def make_population(fixture, count, seed):
    law = fixture.get("data_law", "gaussian")
    generator = torch.Generator().manual_seed(seed)
    if law == "circle":
        radius = float(fixture["circle_radius"])
        angles = 2*torch.pi*torch.rand(count, generator=generator, dtype=torch.float64)
        normal = torch.stack((angles.cos(), angles.sin()), dim=-1)
        clean = radius*normal
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=-1)[..., None]
        population = None
    else:
        basis = torch.linalg.qr(fixture["tangent_basis"].double()).Q
        scale = fixture["latent_std"] if law == "gaussian" else fixture["latent_half_width"]
        population = AffineFixture(law, basis, torch.zeros(len(basis), dtype=torch.float64),
                                   torch.full((basis.shape[1],), float(scale), dtype=torch.float64))
        clean = population.sample(count, seed, dtype=torch.float64)
        tangent = basis[None].expand(count, -1, -1)
        normal = None
    # Independent RNG stream: AffineFixture.sample owns its own seeded generator.
    # Reusing the same initial seed here would recycle latent random numbers.
    noise_generator = torch.Generator().manual_seed(seed+1701)
    noise = torch.randn(clean.shape, generator=noise_generator, dtype=torch.float64)
    noise_t = torch.einsum("ndi,ni->nd", tangent, torch.einsum("ndi,nd->ni", tangent, noise))
    noise_n = noise-noise_t
    direction_n = noise_n/noise_n.norm(dim=-1, keepdim=True)
    return population, clean, tangent, noise_t, noise_n, direction_n


def oracle_at(points, lam, fixture, population, frame):
    if population is not None:
        moments = population.posterior(points, lam)
        tangent_jac = torch.diag_embed(moments.tangent_gain)
        return moments.mean, tangent_jac, points.new_zeros(len(points))
    exact = circle_posterior(points, lam, radius=float(fixture["circle_radius"]))
    tangent_jac = torch.einsum("nai,nab,nbj->nij", frame, exact.jacobian, frame)
    return exact.mean, tangent_jac, exact.jacobian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--derivative-queries", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixture = torch.load(args.checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    fixture.setdefault("data_law", "gaussian")
    population, clean, frame, noise_t, noise_n, normal_direction = make_population(fixture, args.queries, 20260905+2301)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, arrays = [], {"clean": clean, "frame": frame, "noise_t": noise_t, "noise_n": noise_n}
    started = time.monotonic()
    oracle_invariance_error = 0.0
    for schedule in ("log_noise", "rectified_linear", "vp_trigonometric"):
        field, payload = load_pilot_checkpoint(args.checkpoint, schedule, args.device)
        field = field.double().eval()
        for lam in (.1, .316227766, 1.0):
            oracle_reference = {}
            for tangent_noise in (0, 1):
                for rho in (0., .25, .5, 1.):
                    points = clean + lam*(tangent_noise*noise_t + rho*noise_n)
                    oracle, oracle_tangent_jac, oracle_jac = oracle_at(points, lam, fixture, population, frame)
                    if rho == 0:
                        oracle_reference[tangent_noise] = oracle
                    if population is not None:
                        oracle_invariance_error = max(oracle_invariance_error,
                            float((oracle-oracle_reference[tangent_noise]).abs().max()))
                    points_device = points.to(args.device)
                    def fn(r):
                        return canonical_posterior(field, r, lam, schedule,
                            input_coordinate=payload["config"].get("input_coordinate", "native"))
                    with torch.no_grad():
                        value = fn(points_device).cpu()
                    error = value-oracle
                    error_t = torch.einsum("ndi,nd->ni", frame, error)
                    error_n = error-torch.einsum("ndi,ni->nd", frame, error_t)
                    count = min(args.derivative_queries, args.queries)
                    tangent_frame = frame[:count].to(args.device)
                    direction = normal_direction[:count].to(args.device)
                    def tangent_fn(r):
                        return torch.einsum("ndi,nd->ni", tangent_frame, fn(r))
                    _, jta = value_and_jacobian(tangent_fn, points_device[:count], chunk=count)
                    jtt = torch.einsum("nia,naj->nij", jta, tangent_frame)
                    normal_cross = torch.einsum("nia,na->ni", jta, direction)
                    inputs = points_device[:count].detach().requires_grad_(True)
                    output_n = (fn(inputs)*direction).sum(-1)
                    grad = torch.autograd.grad(output_n.sum(), inputs)[0]
                    gain_n = (grad*direction).sum(-1).detach().cpu()
                    if population is None:
                        oracle_gain_n = torch.einsum("ni,nij,nj->n", normal_direction[:count], oracle_jac[:count], normal_direction[:count])
                        oracle_cross = torch.einsum("nai,nab,nb->ni", frame[:count], oracle_jac[:count], normal_direction[:count])
                    else:
                        oracle_gain_n = torch.zeros_like(gain_n)
                        oracle_cross = torch.zeros_like(normal_cross.cpu())
                    jtt = jtt.cpu()
                    row = {
                        "run": args.checkpoint.parent.name, "law": fixture["data_law"],
                        "schedule": schedule, "step": payload["step"], "lambda": lam,
                        "tangent_noise_multiplier": tangent_noise, "normal_noise_multiplier": rho,
                        "query_count": args.queries, "derivative_query_count": count,
                        "value_error_energy_over_lambda2": float(error.square().sum(-1).mean()/lam**2),
                        "tangent_value_error_energy_over_lambda2": float(error_t.square().sum(-1).mean()/lam**2),
                        "normal_value_error_energy_over_lambda2": float(error_n.square().sum(-1).mean()/lam**2),
                        "tangent_trace_mean": float(jtt.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
                        "tangent_trace_oracle_mae": float((jtt-oracle_tangent_jac[:count]).diagonal(dim1=-2, dim2=-1).sum(-1).abs().mean()),
                        "tangent_jacobian_oracle_frobenius_sq": float((jtt-oracle_tangent_jac[:count]).square().sum((-2,-1)).mean()),
                        "normal_direction_gain_mean": float(gain_n.mean()),
                        "normal_direction_gain_oracle_mae": float((gain_n-oracle_gain_n).abs().mean()),
                        "normal_to_tangent_direction_oracle_sq": float((normal_cross.cpu()-oracle_cross).square().sum(-1).mean()),
                        "actual_normal_noise_radius_mean": float((rho*lam*noise_n).norm(dim=-1).mean()),
                        "nominal_normal_shell_radius": lam*(clean.shape[-1]-frame.shape[-1])**.5,
                    }
                    if population is not None and fixture["data_law"] == "uniform":
                        latent_query = points @ population.basis
                        interior = (population.latent_scale-latent_query.abs()).amin(-1) > 3*lam
                        row["interior_query_count"] = int(interior.sum())
                        row["interior_tangent_value_error_energy_over_lambda2"] = float(error_t[interior].square().sum(-1).mean()/lam**2) if interior.any() else None
                    rows.append(row)
                    prefix = f"{schedule}/{lam}/{tangent_noise}/{rho}"
                    arrays[prefix+"/posterior"] = value
                    arrays[prefix+"/oracle"] = oracle
                    arrays[prefix+"/tangent_jacobian"] = jtt
            print(json.dumps({"run": args.checkpoint.parent.name, "schedule": schedule, "lambda": lam, "elapsed": time.monotonic()-started}), flush=True)
    if population is not None:
        assert oracle_invariance_error < 1e-10, oracle_invariance_error
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    torch.save(arrays, args.output / "query_arrays.pt")
    summary = {"source": str(args.checkpoint), "query_seed": 20260905+2301, "noise_seed": 20260905+2301+1701,
        "population": fixture["data_law"], "ambient_dim": clean.shape[-1], "intrinsic_dim": frame.shape[-1],
        "oracle_normal_sweep_invariance_max_error": oracle_invariance_error,
        "geometry_scope": "Exact tangent Jacobian and one normal-direction gain, not full ambient trace. Circle uses fixed clean-point frames and its exact off-circle oracle.",
        "intervention_scope": "Frozen-input normal-noise sweep; affine oracle target stays fixed along each sweep. k=1,rho=1 is actual isotropic training-noise distribution; k=0,rho=0 is clean support.",
        "elapsed_seconds": time.monotonic()-started, "metrics": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"completed": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
