"""Exact readouts and temporal value decomposition for the frozen LR controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_oracles import AffineFixture
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint
from experiments.fm_mechanism_crossgeometry_optimizer import SCALES


VARIANTS = ("lr1e3", "lr1e4", "lr1e3_tailavg", "lr1e4_tailavg")


def geometry(fixture, queries):
    query = fixture["clean_queries"][:queries].double()
    if fixture["data_law"] == "circle":
        radius = float(fixture["circle_radius"])
        query = radius * query/query.norm(dim=-1, keepdim=True)
        normal = query/radius
        tangent = torch.stack((-normal[:, 1], normal[:, 0]), -1)
        project = lambda value: (value*tangent).sum(-1, keepdim=True)
        oracle = lambda scale: circle_posterior(query, scale, radius=radius)
    else:
        basis = torch.linalg.qr(fixture["tangent_basis"].double()).Q
        population = AffineFixture("uniform", basis, fixture["center"].double(),
            torch.full((basis.shape[1],), fixture["latent_half_width"], dtype=torch.float64))
        project = lambda value: value@basis
        oracle = lambda scale: population.posterior(query, scale)
    return query, project, oracle


def temporal_energy(values, query, project, scale):
    """Exact time-mean energy = energy of time mean + temporal variance."""
    mean = values.mean(0)
    deviations = values-mean
    displacement = values-query
    mean_displacement = mean-query
    tangent = project(displacement)
    mean_tangent = project(mean_displacement)
    deviations_tangent = project(deviations)
    total = displacement.square().sum(-1).mean()/scale**2
    bias = mean_displacement.square().sum(-1).mean()/scale**2
    variance = deviations.square().sum(-1).mean()/scale**2
    ct = tangent.square().sum(-1).mean()/scale**2
    bt = mean_tangent.square().sum(-1).mean()/scale**2
    vt = deviations_tangent.square().sum(-1).mean()/scale**2
    assert abs(float(total-bias-variance)) < 1e-8
    assert abs(float(ct-bt-vt)) < 1e-8
    return {"correction_time_mean": float(total), "correction_of_mean_function": float(bias),
            "temporal_variance_energy": float(variance),
            "tangent_correction_time_mean": float(ct), "tangent_correction_of_mean_function": float(bt),
            "tangent_temporal_variance_energy": float(vt),
            "normal_temporal_variance_energy": float(variance-vt),
            "normal_correction_of_mean_function": float(bias-bt),
            "energy_identity_residual": abs(float(total-bias-variance))}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values, response, full, oracle, query, project, scale, intrinsic):
    delta = values-query
    ct = project(delta).square().sum(-1)/scale**2
    total = delta.square().sum(-1)/scale**2
    return {"full_mae_lid": float((full-intrinsic).abs().mean()),
            "response_mae_lid": float((response-intrinsic).abs().mean()),
            "full_oracle_mae": float((full-oracle.full).abs().mean()),
            "response_oracle_mae": float((response-oracle.response).abs().mean()),
            "full_mean": float(full.mean()), "response_mean": float(response.mean()),
            "correction_mean": float(total.mean()), "tangent_correction_mean": float(ct.mean()),
            "normal_correction_mean": float((total-ct).mean()),
            "oracle_full_mean": float(oracle.full.mean()),
            "oracle_response_mean": float(oracle.response.mean()),
            "oracle_correction_mean": float(oracle.correction.mean()),
            "posterior_oracle_error_energy_over_lambda2": float((values-oracle.mean).square().sum(-1).mean()/scale**2)}


def analyze(output: Path, *, device="cuda:1"):
    torch.set_num_threads(4)
    manifest = json.loads((output/"manifest.json").read_text())
    source = Path(manifest["source"])
    fixture = torch.load(source.parent/"data.pt", map_location="cpu", weights_only=False)
    query, project, oracle_at = geometry(fixture, manifest["query_count"])
    law = fixture["data_law"]
    intrinsic = manifest["source_config"]["intrinsic_dim"]
    observations = torch.load(output/"paired_telemetry.pt", map_location="cpu", weights_only=False)["observations"]
    temporal_rows, readout_rows, stored_arrays = [], [], {}
    for variant in ("lr1e3", "lr1e4"):
        for schedule in manifest["source_config"]["schedules"]:
            for scale in SCALES:
                chosen = [r for r in observations if r["variant"] == variant and r["schedule"] == schedule
                          and r["lambda"] == scale and r["step"] > manifest["source_step"]+manifest["added_steps"]//2]
                values = torch.stack([r["values"].double() for r in chosen])
                moments = oracle_at(scale)
                row = {"data_law": law, "variant": variant, "schedule": schedule, "lambda": scale,
                       "query_count": len(query), "temporal_count": len(chosen),
                       **temporal_energy(values, query, project, scale)}
                row["posterior_error_of_mean_function"] = float((values.mean(0)-moments.mean).square().sum(-1).mean()/scale**2)
                row["posterior_error_time_mean"] = float((values-moments.mean).square().sum(-1).mean()/scale**2)
                row["oracle_correction_mean"] = float(moments.correction.mean())
                temporal_rows.append(row)
    write_csv(output/"temporal_summary.csv", temporal_rows)

    # Existing unchanged source arrays retain exactly these first query IDs.
    old = source.parent/("circle_diagnostics_008000" if law == "circle" else "diagnostics_008000")/"query_arrays.npz"
    with np.load(old) as baseline:
        old_query = baseline["query" if law == "circle" else "clean"][:len(query)]
        assert np.allclose(old_query, query.numpy(), atol=1e-14, rtol=0)
        for schedule in manifest["source_config"]["schedules"]:
            for scale in SCALES:
                prefix = f"{schedule}__{scale:.9g}__" + ("" if law == "circle" else "clean__")
                val, resp, full = [torch.from_numpy(baseline[prefix+k][:len(query)]) for k in ("posterior", "response", "full")]
                readout_rows.append({"data_law": law, "variant": "source8000", "schedule": schedule,
                    "lambda": scale, "query_count": len(query),
                    **summarize(val, resp, full, oracle_at(scale), query, project, scale, intrinsic)})
    started = time.monotonic()
    for variant in VARIANTS:
        checkpoint = output/variant/"checkpoint_step_009000.pt"
        for schedule in manifest["source_config"]["schedules"]:
            field, payload = load_pilot_checkpoint(checkpoint, schedule, device)
            field = field.double()
            for scale in SCALES:
                fn = lambda r: canonical_posterior(field, r, scale, schedule,
                    input_coordinate=payload["config"]["input_coordinate"],
                    output_coordinate=payload["config"].get("output_coordinate", "data"))
                value, jac = value_and_jacobian(fn, query.to(device))
                response = jac.diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu()
                value = value.detach().cpu()
                full = response+(value-query).square().sum(-1)/scale**2
                row = {"data_law": law, "variant": variant, "schedule": schedule,
                       "lambda": scale, "query_count": len(query),
                       **summarize(value, response, full, oracle_at(scale), query, project, scale, intrinsic)}
                readout_rows.append(row)
                prefix = f"{variant}__{schedule}__{scale:.9g}__"
                for name, tensor in {"posterior":value, "response":response, "full":full}.items():
                    stored_arrays[prefix+name] = tensor.numpy()
                del jac
            print(json.dumps({"law":law,"variant":variant,"schedule":schedule,"elapsed":time.monotonic()-started}), flush=True)
            del field
        write_csv(output/"readout_summary.csv", readout_rows)
    stored_arrays["query"] = query.numpy()
    np.savez_compressed(output/"readout_arrays.npz", **stored_arrays)
    receipt = {"status":"complete", "readout_cells":len(readout_rows), "temporal_cells":len(temporal_rows),
               "exact_derivatives":True, "trace_sampling":False, "baseline_query_match":True,
               "maximum_energy_identity_residual":max(r["energy_identity_residual"] for r in temporal_rows),
               "elapsed_seconds":time.monotonic()-started,
               "precision":"trajectory model values float32; temporal accounting and final exact derivatives float64",
               "interpretation":"Sequential states quantify temporal variation, not independent seed uncertainty. Parameter averaging is distinct from averaging predictions."}
    (output/"analysis_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    analyze(**vars(parser.parse_args()))
