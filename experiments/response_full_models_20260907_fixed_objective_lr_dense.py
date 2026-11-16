"""Descriptive dense-trajectory drift and lag audit, not a cycle estimator."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve_temporal import temporal_trend
from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase
from experiments.response_full_models_20260907_ve_refinement_audit import value_components


def lag_summary(sequence,steps):
    b=np.asarray(sequence);t=np.asarray(steps,dtype=float);t=t-t.mean()
    centered=b-b.mean(0)
    slope=np.einsum("t,t...->...",t,centered)/(t@t)
    residual=centered-t.reshape((-1,)+(1,)*(b.ndim-1))*slope
    rows=[]
    for name,x in (("centered",centered),("linear_detrended",residual)):
        for lag in range(1,17):
            a,c=x[lag:],x[:-lag];den=np.sqrt((a*a).sum()*(c*c).sum())
            rows.append({"series":name,"lag":lag,"normalized_inner_product":float((a*c).sum()/den) if den else None})
    power=np.sum(np.abs(np.fft.rfft(residual,axis=0))**2,axis=tuple(range(1,b.ndim)))
    power[0]=0.;index=int(np.argmax(power))
    spectral={"peak_frequency_cycles_per_update":index/len(b),"peak_period_updates":len(b)/index if index else None,
        "peak_fraction_of_nonzero_rfft_power":float(power[index]/power.sum()) if power.sum() else None}
    return rows,spectral


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    oracle=np.load(args.oracle);rows=[];sources={}
    for arm in ("original_lr","low_lr"):
        path=args.run/f"{arm}_per_step_values.npz";data=np.load(path);sources[str(path)]=sha256(path)
        np.testing.assert_array_equal(data["query_ids"],oracle["query_ids"][:64])
        np.testing.assert_array_equal(data["queries"],oracle["queries"][:64])
        b=data["posterior"];sigma=float(data["sigma"]);truth=oracle["posterior"][0,:64]
        assert sigma==.01 and np.all(np.diff(data["steps"])==1)
        trend=temporal_trend(b[:,None],data["steps"].copy(),np.array([sigma]))
        mean=b.mean(0);centered=b-mean[None];c=((b-data["queries"][None])**2).sum(-1)/sigma**2
        cmean=((mean-data["queries"])**2).sum(-1)/sigma**2
        variance=(centered**2).sum(-1).mean(0)/sigma**2
        np.testing.assert_allclose(c.mean(0),cmean+variance,rtol=1e-10,atol=1e-10)
        tangent=tangent_from_phase(oracle["query_phase"][:64])
        parts=value_components(centered[:,None],tangent,np.array([sigma]))
        lags,spectral=lag_summary(b,data["steps"])
        rows.append({"arm":arm,"sigma":sigma,"steps":len(b),"first_step":int(data["steps"][0]),"last_step":int(data["steps"][-1]),
            "mean_correction":float(c.mean()),"mean_function_correction":float(cmean.mean()),
            "variation":float(variance.mean()),"mean_function_oracle_error_energy":float(((mean-truth)**2).sum(-1).mean()/sigma**2),
            "linear_trend_fraction":float(trend["linear_trend_variation"].mean()/variance.mean()),
            "between_half_fraction":float(trend["half_window_between_variation"].mean()/variance.mean()),
            **{"variation_"+k:float(v.mean()) for k,v in parts.items()},"lag_correlations":lags,"spectrum":spectral})
        np.savez_compressed(args.output/f"{arm}.npz",steps=data["steps"],query_ids=data["query_ids"],
            correction=c,mean_function_correction=cmean,variance=variance,
            **trend,**{"variation_"+k:v for k,v in parts.items()})
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps({"sources":sources,"source_sha256":sha256(__file__),
        "oracle_sha256":sha256(args.oracle),"scope":"Descriptive finite trajectory; linear detrending and lags do not establish a stationary process or an asymptotic limit cycle",
        "lag_definition":"Sum inner product over all query/coordinate pairs, divided by each lagged series L2 norms",
        "spectrum":"Aggregate squared rFFT magnitude of linearly detrended coordinate trajectories; no inferential test"},indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_lr_spaghetti"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
