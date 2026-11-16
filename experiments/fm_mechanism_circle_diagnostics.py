"""Exact curved-support audit for posterior FM; no stochastic trace estimate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint


def audit(path: Path, device: str = "cpu", queries: int = 256):
    torch.set_num_threads(4)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fixture = torch.load(path.parent / "data.pt", map_location="cpu", weights_only=False)
    assert fixture["data_law"] == "circle"
    radius = float(fixture["circle_radius"])
    query = fixture["clean_queries"][:queries].to(device=device, dtype=torch.float64)
    raw_query = query.clone()
    query = radius * query / query.norm(dim=-1, keepdim=True)
    normal = query / radius
    tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
    scales = np.array(sorted({float(f"{v:.9g}") for v in np.r_[
        np.geomspace(1e-4, 1, 65), [.001, .003, .01, .03, .1, .316227766, 1]
    ]}))
    rows, arrays = [], {"query": query.cpu().numpy(), "normal": normal.cpu().numpy(), "tangent": tangent.cpu().numpy()}
    output = path.parent / f"circle_diagnostics_{payload['step']:06d}"
    output.mkdir(exist_ok=True)
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(path, schedule, device)
        field = field.double()
        for lam in scales:
            fn = lambda r: canonical_posterior(field, r, float(lam), schedule,
                                              input_coordinate=payload["config"]["input_coordinate"])
            val, jac = value_and_jacobian(fn, query, chunk=queries)
            oracle = circle_posterior(query, float(lam), radius=radius)
            tn = torch.einsum("bi,bij,bj->b", tangent, jac, tangent)
            nn = torch.einsum("bi,bij,bj->b", normal, jac, normal)
            delta = val-query
            dt = (delta*tangent).sum(-1)
            dn = (delta*normal).sum(-1)
            correction = delta.square().sum(-1)/lam**2
            response = tn+nn
            full = response+correction
            row = {
                "schedule": schedule, "step": payload["step"], "lambda": float(lam),
                "outside_training_range": bool(lam < payload["config"]["lambda_min"]),
                "response_mean": float(response.mean()),
                "full_mean": float(full.mean()),
                "normal_trace_mean": float(nn.mean()), "tangent_trace_mean": float(tn.mean()),
                "normal_trace_std_queries": float(nn.std()),
                "correction_normal_mean": float((dn.square()/lam**2).mean()),
                "correction_tangent_mean": float((dt.square()/lam**2).mean()),
                "radial_displacement_mean": float(dn.mean()),
                "radial_displacement_rms": float(dn.square().mean().sqrt()),
                "response_mae_lid": float((response-1).abs().mean()),
                "full_mae_lid": float((full-1).abs().mean()),
                "response_oracle_mae": float((response-oracle.response).abs().mean()),
                "full_oracle_mae": float((full-oracle.full).abs().mean()),
                "normal_trace_oracle_mae": float((nn-oracle.radial_gain).abs().mean()),
                "oracle_response": float(oracle.response.mean()),
                "oracle_full": float(oracle.full.mean()),
                "oracle_normal_trace": float(oracle.radial_gain.mean()),
                "oracle_tangent_trace": float(oracle.tangent_gain.mean()),
                "oracle_correction": float(oracle.correction.mean()),
                "posterior_oracle_error_energy_over_lambda2": float((val-oracle.mean).square().sum(-1).mean()/lam**2),
            }
            # Excess normal gain cannot automatically be fitted by a tube:
            # signs and scale dependence are retained, including extrapolation.
            excess_n = nn-oracle.radial_gain
            valid = (excess_n>0) & (excess_n<1)
            row["tube_gain_valid_fraction"] = float(valid.double().mean())
            row["implied_tube_width_median"] = float((lam*torch.sqrt(excess_n[valid]/(1-excess_n[valid]))).median()) if valid.any() else float("nan")
            rows.append(row)
            key = f"{schedule}__{lam:.9g}"
            for name, tensor in {"posterior":val,"jacobian":jac,"response":response,"full":full,
                                 "normal_trace":nn,"tangent_trace":tn,"normal_displacement":dn,
                                 "tangent_displacement":dt,"oracle_response":oracle.response,"oracle_full":oracle.full}.items():
                arrays[key+"__"+name] = tensor.detach().cpu().numpy()
        print(json.dumps({"schedule":schedule,"status":"audited","step":payload["step"]}), flush=True)
    with (output/"summary.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/"query_arrays.npz", **arrays)
    (output/"manifest.json").write_text(json.dumps({
        "source_checkpoint":path.name,"step":payload["step"],"query_count":queries,
        "query_radial_roundoff_projection_max":float((query-raw_query).abs().max()),
        "exact_derivatives":True,"stochastic_trace":False,"radius":radius,
        "scale_min":float(scales.min()),"scale_max":float(scales.max()),
        "extrapolation":"lambda<0.01 is outside training; no zero-limit claim from finite scales",
    },indent=2)+"\n")
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--queries",type=int,default=256)
    args=parser.parse_args()
    audit(args.checkpoint,args.device,args.queries)
