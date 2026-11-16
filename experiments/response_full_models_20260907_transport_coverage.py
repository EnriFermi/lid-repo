"""Fixed endpoint coverage and distance summaries, not a sampler-quality proof.

Projection results come from an independently reviewed continuous support
solver. The baseline is p_.01, not clean data. Fixed32-bin marginals and8x8
tube joint bins flag coverage changes; they are not selected after outcomes.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance,ks_2samp
from experiments.response_full_models_20260907_span import sha256


def circular_summary(theta,reference):
    theta=np.mod(theta,2*np.pi);reference=np.mod(reference,2*np.pi)
    bins=np.linspace(0,2*np.pi,33)
    hist=np.histogram(theta,bins)[0]/len(theta);ref_hist=np.histogram(reference,bins)[0]/len(reference)
    k=np.arange(1,9)
    moment=np.exp(1j*theta[:,None]*k).mean(0);ref_moment=np.exp(1j*reference[:,None]*k).mean(0)
    ordered=np.sort(theta);gaps=np.diff(np.r_[ordered,ordered[0]+2*np.pi])
    return {"histogram32_tv":float(.5*np.abs(hist-ref_hist).sum()),
        "histogram32_occupied":int((hist>0).sum()),"largest_circular_gap":float(gaps.max()),
        "largest_moment1to8_magnitude":float(np.abs(moment).max()),
        "largest_moment1to8_difference":float(np.abs(moment-ref_moment).max())}, {
            "histogram32":hist,"reference_histogram32":ref_hist,
            "moment1to8_real":moment.real,"moment1to8_imag":moment.imag,
            "reference_moment1to8_real":ref_moment.real,"reference_moment1to8_imag":ref_moment.imag}


def tube_coverage(u,theta,ref_u,ref_theta):
    bins=np.linspace(0,8,33);h=np.histogram(u,bins)[0]/len(u);hr=np.histogram(ref_u,bins)[0]/len(ref_u)
    joint=np.histogram2d(u,theta,bins=[np.linspace(0,8,9),np.linspace(0,2*np.pi,9)])[0]/len(u)
    joint_ref=np.histogram2d(ref_u,ref_theta,bins=[np.linspace(0,8,9),np.linspace(0,2*np.pi,9)])[0]/len(ref_u)
    circ,arrays=circular_summary(theta,ref_theta)
    return {"u_histogram32_tv":float(.5*np.abs(h-hr).sum()),"u_histogram32_occupied":int((h>0).sum()),
        "u_wasserstein1":float(wasserstein_distance(u,ref_u)),
        "u_ks_distance":float(ks_2samp(u,ref_u).statistic),
        "u_theta_joint8x8_tv":float(.5*np.abs(joint-joint_ref).sum()),
        "u_theta_joint8x8_occupied":int((joint>0).sum()),
        "u_endpoint_fraction":float(((u==0)|(u==8)).mean()),
        **{"theta_"+k:v for k,v in circ.items()}}, {
            "u_histogram32":h,"reference_u_histogram32":hr,
            "joint8x8":joint,"reference_joint8x8":joint_ref,**arrays}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not (args.geometry/"manifest.json").exists():raise ValueError("Geometry audit is not complete")
    rows=[];hashes={str(args.geometry/"manifest.json"):sha256(args.geometry/"manifest.json")}
    arrays_output={}
    for name in ["spaghetti","exp"]:
        reference=np.load(args.geometry/name/"reference.npz")
        for variant in ["reference","posterior","projected_posterior","calibrated_projected_posterior","noise"]:
            source=args.geometry/name/f"{variant}.npz";a=np.load(source);hashes[str(source)]=sha256(source)
            np.testing.assert_array_equal(a["sample_ids"],reference["sample_ids"])
            rms=float(a["normalization_rms"]);row={"dataset":name,"variant":variant,"count":len(a["sample_ids"]),
                "ambiguous_projection_fraction":float(a["ambiguous_projection"].mean())}
            for key in ["distance_squared","active_distance_squared","ambient_distance_squared"]:
                d=np.sqrt(a[key])/rms;ref=np.sqrt(reference[key])/rms
                prefix=key.removesuffix("_squared")
                row[prefix+"_rms_canonical"]=float(np.sqrt(np.mean(d*d)))
                row[prefix+"_rms_ratio_to_reference"]=float(np.sqrt(np.mean(d*d)/np.mean(ref*ref)))
                row[prefix+"_distribution_wasserstein1"]=float(wasserstein_distance(d,ref))
                for label,p in [("p10",.1),("p50",.5),("p90",.9),("p99",.99)]:row[prefix+"_"+label]=float(np.quantile(d,p))
            if name=="spaghetti":
                coverage,latent=circular_summary(a["phase"],reference["phase"])
            else:
                coverage,latent=tube_coverage(a["u"],a["theta"],reference["u"],reference["theta"])
            row.update(coverage);rows.append(row)
            for k,v in latent.items():arrays_output[name+"_"+variant+"_"+k]=v
            print(json.dumps(row),flush=True)
    assert all(sha256(p)==h for p,h in hashes.items())
    np.savez_compressed(args.output/"coverage_arrays.npz",**arrays_output)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/"summary.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),"source_hashes":hashes,
        "metrics":"Predeclared32-bin latent marginals, tube8x8joint, circularmoments1..8, largest circular gap, uW1/KS; no pvalues/seed sweep",
        "reference":"Independent correct p_.01 marginal on same sourceX/noise fixture, not samplewise terminal coupling",
        "scope":"Coverage summaries detect gross collapse but do not certify equality of complete distributions",
        "distance":"Smaller than p_.01 reference can indicate underdispersion, not automatically better samples",
        "all_sources_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry",type=Path,default=Path("artifacts/response_full_20260907/probability_flow_geometry"))
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
