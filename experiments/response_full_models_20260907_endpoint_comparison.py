"""Pointwise exact-Spiral model/oracle window ledger; no operating-point choice."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256


def accuracy_flags(response,full,oracle_response,oracle_full,tolerance=.1):
    rg=np.abs(response-1)<tolerance;fg=np.abs(full-1)<tolerance
    org=np.abs(oracle_response-1)<tolerance;ofg=np.abs(oracle_full-1)<tolerance
    return {"model_response_good":rg,"model_full_good":fg,"oracle_response_good":org,"oracle_full_good":ofg,
        "joint_response_good":rg&org,"joint_full_good":fg&ofg,
        "oracle_full_and_model_response_good":ofg&rg}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];sources={};max_c_replay=0.;max_identity=0.
    for family in ("ve_edm","vp_noise","vp_posterior"):
        path=args.models/f"spiral_{family}.npz"
        reference=args.oracle/("ve_oracle.npz" if family=="ve_edm" else "vp_oracle.npz")
        model=np.load(path);oracle=np.load(reference)
        sources[str(path)]=sha256(path);sources[str(reference)]=sha256(reference)
        for key in ("queries","query_raw","query_ids","scales","normalization_mean","normalization_rms"):
            np.testing.assert_array_equal(model[key],oracle[key])
        q=model["queries"];s=model["scales"]
        oracle_shift=(oracle["posterior"]-q[None])/s[:,None,None]
        oracle_c=(oracle_shift**2).sum(-1)
        np.testing.assert_allclose(oracle_c,oracle["correction"],rtol=1e-7,atol=1e-11)
        max_c_replay=max(max_c_replay,float(np.abs(oracle_c-oracle["correction"]).max()))
        shift=model["normalized_displacement"];error=shift-oracle_shift
        cross=2*(oracle_shift*error).sum(-1);energy=(error**2).sum(-1)
        identity=oracle["correction"]+cross+energy
        np.testing.assert_allclose(model["correction"],identity,rtol=1e-10,atol=1e-9)
        max_identity=max(max_identity,float((np.abs(model["correction"]-identity)/(1+model["correction"])).max()))
        response_error=model["response"]-oracle["response"]
        np.testing.assert_allclose(model["full"]-oracle["full"],response_error+cross+energy,rtol=1e-10,atol=1e-9)
        flags=accuracy_flags(model["response"],model["full"],oracle["response"],oracle["full"])
        for i,scale in enumerate(s):
            rows.append({"family":family,"lambda":float(scale),"count":len(q),"out_of_training_range":float(scale)<.01,
                "model_response_mean":float(model["response"][i].mean()),"model_correction_mean":float(model["correction"][i].mean()),
                "model_response_d_mae":float(abs(model["response"][i]-1).mean()),"model_full_d_mae":float(abs(model["full"][i]-1).mean()),
                "oracle_response_mean":float(oracle["response"][i].mean()),"oracle_correction_mean":float(oracle["correction"][i].mean()),
                "oracle_response_d_mae":float(abs(oracle["response"][i]-1).mean()),"oracle_full_d_mae":float(abs(oracle["full"][i]-1).mean()),
                "full_oracle_mae":float(abs(model["full"][i]-oracle["full"][i]).mean()),
                "response_error_mean":float(response_error[i].mean()),"correction_cross_mean":float(cross[i].mean()),
                "posterior_error_energy_mean":float(energy[i].mean()),
                **{k+"_fraction":float(v[i].mean()) for k,v in flags.items()}})
        np.savez_compressed(args.output/f"{family}.npz",scales=s,query_ids=model["query_ids"],
            response_error=response_error,oracle_normalized_displacement=oracle_shift,
            posterior_error_normalized=error,correction_cross=cross,posterior_error_energy=energy,**flags)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),"sources":sources,
        "fraction_definition":"Strict absolute error<.1 against d1, fixed from oracle bridge; no threshold tuning or selector",
        "joint_definition":"Same query at same scale passes both oracle and model gates; no aggregate-MAE proxy",
        "max_oracle_canonical_c_replay_error":max_c_replay,"max_c_identity_error_relative_to_1_plus_c":max_identity,
        "scope":"Exact continuous queries; below.01 allmodelresults extrapolate outside training; finite13-pointgrid doesnotprove absence at everycontinuousscale"},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",type=Path,default=Path("artifacts/response_full_20260907/spiral_exact_endpoint_models"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/spiral_scale_window"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
