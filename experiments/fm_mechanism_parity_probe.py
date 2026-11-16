"""Fixed-checkpoint Gaussian parity and sideways-error diagnostic (CPU only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.fm_mechanism_diagnostics import value_and_jacobian
from experiments.fm_mechanism_pilot import canonical_posterior, load_pilot_checkpoint


def energy(v, lam):
    return v.square().sum(-1) / lam**2


def sideways(v, directions):
    return v - (v * directions).sum(-1, keepdim=True) * directions


def mean(value):
    return float(value.detach().mean())


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(checkpoint, output, derivative_count=24):
    started = time.monotonic()
    torch.set_num_threads(4)
    dtype = torch.float64
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(checkpoint.parent / "data.pt", map_location="cpu", weights_only=False)
    if data.get("data_law", "gaussian") != "gaussian" or data["center"].abs().max() != 0:
        raise ValueError("This parity oracle requires the centered Gaussian fixture")
    cache = np.load(checkpoint.parent / "diagnostics_008000/query_arrays.npz")
    d = int(payload["config"]["intrinsic_dim"])
    q = torch.from_numpy(cache["basis"]).to(dtype)[:, :d]
    clean = data["clean_queries"].to(dtype)
    test = clean[:64]
    calibration = clean[128:640]
    points = torch.cat([test, calibration])
    xt = points @ q
    u = xt / xt.norm(dim=-1, keepdim=True)
    variance = float(data["latent_std"])**2
    legacy_path = checkpoint.parent / "value_calibration/coefficients.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else {}
    value_rows, readout_rows, fit_arrays, arrays = [], [], {}, {}
    max_parity_error = max_jac_error = 0.0
    for schedule in payload["models"]:
        field, _ = load_pilot_checkpoint(checkpoint, schedule, "cpu")
        field = field.to(dtype).eval()
        for lam in (0.316227766, 0.1, 1.0, 0.01):
            def fn(x):
                return canonical_posterior(
                    field, x, lam, schedule,
                    input_coordinate=payload["config"].get("input_coordinate", "native"),
                    output_coordinate=payload["config"].get("output_coordinate", "data"),
                )

            with torch.no_grad():
                bp, bm = fn(points), fn(-points)
                bzero = fn(torch.zeros_like(points[:1]))[0]
            odd, even = (bp-bm)/2, (bp+bm)/2
            odd_t, even_t = odd @ q, even @ q
            gain = variance/(variance+lam**2)
            odd_error = odd_t-gain*xt
            linear = torch.linalg.lstsq(xt[64:], odd_error[64:]).solution
            constant = even_t[64:].mean(0)
            even_nonconstant = even_t-constant
            odd_nonlinear = odd_error-xt@linear
            side_even = sideways(even_t, u)
            side_odd = sideways(odd_t, u)
            side_linear = sideways(xt@linear, u)
            side_nonlinear = sideways(odd_nonlinear, u)
            side_constant = sideways(constant.expand_as(xt), u)
            side_even_nonconstant = sideways(even_nonconstant, u)
            side_zero = sideways((bzero@q).expand_as(xt), u)
            side_even_minuszero = sideways(even_t-bzero@q, u)
            side_pair = (energy(sideways(bp@q,u),lam)+energy(sideways(bm@q,u),lam))/2
            split_side = energy(side_even,lam)+energy(side_odd,lam)
            max_parity_error = max(max_parity_error, float((side_pair-split_side).abs().max()))
            key = f"{schedule}__{lam:.9g}"
            legacy_key = key+"__clean"
            legacy_affine = torch.tensor(legacy[legacy_key]["affine"],dtype=dtype) if legacy_key in legacy else None
            row = {
                "schedule":schedule,"lambda":lam,"test_pairs":64,
                "side_correction_original_positive":mean(energy(sideways(bp[:64]@q,u[:64]),lam)),
                "side_correction_pair":mean(side_pair[:64]),
                "side_even":mean(energy(side_even[:64],lam)),
                "side_odd":mean(energy(side_odd[:64],lam)),
                "side_odd_linear":mean(energy(side_linear[:64],lam)),
                "side_odd_nonlinear":mean(energy(side_nonlinear[:64],lam)),
                "side_odd_linear_nonlinear_cross":mean(2*(side_linear[:64]*side_nonlinear[:64]).sum(-1)/lam**2),
                "side_even_constant":mean(energy(side_constant[:64],lam)),
                "side_even_nonconstant":mean(energy(side_even_nonconstant[:64],lam)),
                "side_even_constant_nonconstant_cross":mean(2*(side_constant[:64]*side_even_nonconstant[:64]).sum(-1)/lam**2),
                "side_even_bzero":mean(energy(side_zero[:64],lam)),
                "side_even_minus_bzero":mean(energy(side_even_minuszero[:64],lam)),
                "tangent_even_energy":mean(energy(even_t[:64],lam)),
                "tangent_even_nonconstant_energy":mean(energy(even_nonconstant[:64],lam)),
                "tangent_even_minus_bzero_energy":mean(energy(even_t[:64]-bzero@q,lam)),
                "tangent_odd_oracle_error_energy":mean(energy(odd_error[:64],lam)),
                "tangent_odd_nonlinear_energy":mean(energy(odd_nonlinear[:64],lam)),
                "fit_constant_norm":float(constant.norm()),
                "bzero_tangent_norm":float((bzero@q).norm()),
                "bzero_minus_fit_constant_norm":float((bzero@q-constant).norm()),
                "odd_error_linear_scalar":float(torch.trace(linear)/d),
                "odd_error_linear_anisotropic_frobenius":float((linear-torch.eye(d,dtype=dtype)*torch.trace(linear)/d).norm()),
                "legacy_affine_linear_difference":float((linear-legacy_affine[:-1]).norm()) if legacy_affine is not None else "not_available",
            }
            value_rows.append(row)
            fit_arrays[key] = {"odd_linear":linear.tolist(),"even_constant":constant.tolist(),"bzero_tangent":(bzero@q).tolist()}
            for name,value in {"positive":bp[:64],"negative":bm[:64],"odd":odd[:64],"even":even[:64],"side_even":side_even[:64],"side_odd":side_odd[:64]}.items():
                arrays[key+"__"+name]=value.numpy()

            x = test[:derivative_count]
            vp,jp = value_and_jacobian(fn,x)
            vm,jm = value_and_jacobian(fn,-x)
            jo=(jp+jm)/2
            bo=(vp-vm)/2
            be=(vp+vm)/2
            def odd_fn(z):
                return (fn(z)-fn(-z))/2
            checkv,checkj=value_and_jacobian(odd_fn,x[:2])
            max_jac_error=max(max_jac_error,float((checkj-jo[:2]).abs().max()),float((checkv-bo[:2]).abs().max()))
            tracep=jp.diagonal(dim1=-2,dim2=-1).sum(-1)
            tracem=jm.diagonal(dim1=-2,dim2=-1).sum(-1)
            response=(tracep+tracem)/2
            oracle_full=d*gain+energy((gain-1)*(x@q),lam)
            correction_pair=(energy(vp-x,lam)+energy(vm+x,lam))/2
            correction_odd=energy(bo-x,lam)
            even_energy=energy(be,lam)
            max_parity_error=max(max_parity_error,float((correction_pair-correction_odd-even_energy).abs().max()))
            anisotropic = linear-torch.eye(d,dtype=dtype)*torch.trace(linear)/d
            anisotropic_shift=(x@q)@anisotropic@q.T
            fitted_constant=constant@q.T
            for method,values_plus,values_minus,rp,rm in (
                ("original",vp,vm,tracep,tracem),
                ("remove_bzero",vp-bzero,vm-bzero,tracep,tracem),
                ("remove_fitted_tangent_constant",vp-constant@q.T,vm-constant@q.T,tracep,tracem),
                ("remove_even_constant_and_odd_anisotropy",vp-fitted_constant-anisotropic_shift,vm-fitted_constant+anisotropic_shift,tracep,tracem),
                ("odd_symmetrized",bo,-bo,response,response),
            ):
                cp,cm=energy(values_plus-x,lam),energy(values_minus+x,lam)
                fp,fm=rp+cp,rm+cm
                readout_rows.append({
                    "schedule":schedule,"lambda":lam,"method":method,"test_pairs":derivative_count,
                    "response_pair_mean":mean((rp+rm)/2),
                    "response_point_mae_oracle":mean(torch.cat([(rp-d*gain).abs(),(rm-d*gain).abs()])),
                    "correction_pair_mean":mean((cp+cm)/2),
                    "full_pair_mean":mean((fp+fm)/2),
                    "full_point_mae_oracle":mean(torch.cat([(fp-oracle_full).abs(),(fm-oracle_full).abs()])),
                    "full_pairmean_mae_oracle":mean(((fp+fm)/2-oracle_full).abs()),
                    "full_pairmean_signed_error_oracle":mean((fp+fm)/2-oracle_full),
                    "oracle_full_mean":mean(oracle_full),
                    "original_even_energy":mean(even_energy),
                })
            print(json.dumps(row),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    write_rows(output/"parity_value_decomposition.csv",value_rows)
    write_rows(output/"parity_readouts.csv",readout_rows)
    (output/"coefficients.json").write_text(json.dumps(fit_arrays,indent=2))
    np.savez_compressed(output/"query_arrays.npz",**arrays,clean=test.numpy(),tangent_basis=q.numpy())
    manifest={"checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "query_indices":[0,64],"calibration_indices":[128,640],"derivative_query_indices":[0,derivative_count],
              "antipodal_queries":"plus/minus each held-out query; two points of a pair are dependent",
              "max_exact_parity_identity_error":max_parity_error,"max_direct_odd_jacobian_error":max_jac_error,
              "dtype":"float64","device":"cpu","threads":4,"elapsed_seconds":time.monotonic()-started,
              "scope":"oracle-known posthoc decomposition, no training or deployability claim"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if max_jac_error>1e-10 or max_parity_error>1e-7:
        raise AssertionError(manifest)
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--derivative-count",type=int,default=24)
    args=parser.parse_args()
    run(args.checkpoint,args.output,args.derivative_count)
