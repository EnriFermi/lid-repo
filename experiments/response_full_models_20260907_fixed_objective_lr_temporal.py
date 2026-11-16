"""Rao--Blackwell control: temporal variation, linear drift, and mean-function bias.

Sixteen sparse observations are not a stationary-process estimate. All variance
identities here are deterministic identities of the saved training trajectory.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_oracle_audit import decompose


def temporal_trend(posterior,steps,scales):
    b=np.asarray(posterior);t=np.asarray(steps,dtype=np.float64);t-=t.mean()
    centered=b-b.mean(0)
    slope=np.einsum("t,t...->...",t,centered)/(t@t)
    trend=t.reshape((-1,)+(1,)*(b.ndim-1))*slope
    residual=centered-trend
    scale2=np.asarray(scales)[:,None]**2
    energy=lambda z:(z**2).sum(-1).mean(0)/scale2
    total,linear,nonlinear=energy(centered),energy(trend),energy(residual)
    np.testing.assert_allclose(total,linear+nonlinear,rtol=1e-11,atol=1e-11)
    first,second=np.split(b,2,axis=0)
    between=.25*((second.mean(0)-first.mean(0))**2).sum(-1)/scale2
    within=.5*(energy(first-first.mean(0))+energy(second-second.mean(0)))
    np.testing.assert_allclose(total,between+within,rtol=1e-11,atol=1e-11)
    return {"total_variation":total,"linear_trend_variation":linear,"detrended_variation":nonlinear,
        "half_window_between_variation":between,"half_window_within_variation":within}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    oracle=np.load(args.oracle);rows=[];sources={}
    for arm in ["original_lr","low_lr"]:
        source=args.run/f"{arm}_temporal.npz";data=np.load(source);sources[str(source)]=sha256(source)
        np.testing.assert_array_equal(data["query_ids"],oracle["query_ids"][:64])
        idx=[int(np.argmin(np.abs(oracle["scales"]-s))) for s in data["scales"]]
        np.testing.assert_allclose(data["scales"],oracle["scales"][idx],rtol=1e-9,atol=0)
        truth=oracle["posterior"][idx,:64];rtruth=oracle["response"][idx,:64]
        q=oracle["queries"][:64];scales=data["scales"]
        trend=temporal_trend(data["posterior"],data["steps"].copy(),scales)
        result=decompose(q,data["mean_function_posterior"],truth,data["mean_function_response"],rtruth,scales)
        mean_full=data["full"].mean(0)
        np.testing.assert_allclose(mean_full,result["full"]+trend["total_variation"],rtol=1e-10,atol=1e-10)
        snapshot_e=((data["posterior"]-truth[None])**2).sum(-1).mean(0)/scales[:,None]**2
        np.testing.assert_allclose(snapshot_e,result["posterior_error_energy"]+trend["total_variation"],rtol=1e-11,atol=1e-11)
        for i,s in enumerate(scales):
            total=float(trend["total_variation"][i].mean())
            rows.append({"arm":arm,"lambda":float(s),"snapshots":len(data["steps"]),
                "first_step":int(data["steps"][0]),"last_step":int(data["steps"][-1]),
                "mean_snapshot_full_d_mae":float(np.abs(data["full"][:,i]-1).mean()),
                "mean_of_full_d_mae":float(np.abs(mean_full[i]-1).mean()),
                "full_of_mean_d_mae":float(np.abs(result["full"][i]-1).mean()),
                "mean_of_full_oracle_mae":float(np.abs(mean_full[i]-result["oracle_full"][i]).mean()),
                "full_of_mean_oracle_mae":float(np.abs(result["full"][i]-result["oracle_full"][i]).mean()),
                "snapshot_posterior_error_energy":float(snapshot_e[i].mean()),
                **{key+"_mean":float(v[i].mean()) for key,v in result.items() if v.ndim==2},
                **{key:float(v[i].mean()) for key,v in trend.items()},
                "linear_trend_fraction":float(trend["linear_trend_variation"][i].mean()/total),
                "half_window_between_fraction":float(trend["half_window_between_variation"][i].mean()/total)})
        np.savez_compressed(args.output/f"{arm}.npz",scales=scales,query_ids=data["query_ids"],
            mean_of_full=mean_full,snapshot_posterior_error_energy=snapshot_e,**trend,**result)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),"sources":sources,
        "oracle_sha256":sha256(args.oracle),"trend":"Per-query/per-coordinate OLS linear step trend, aggregate energy fraction",
        "interpretation":"Deterministic sparse-trajectory variation, not stationarity or stochastic-noise identification",
        "jacobian_scope":"Mean function has exact total response only; no saved snapshot matrices or per-block decomposition"},indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_lr_spaghetti"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
