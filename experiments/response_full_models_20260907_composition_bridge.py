"""Exact saved-array value and derivative mechanism for two-pass composition."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_clean_anchor import write_rows
from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase


def projectors(tangent,dimension=30):
    n,active=tangent.shape;p=np.zeros((n,dimension,dimension));p[:,:active,:active]=np.eye(active)
    t=np.zeros((n,dimension));t[:,:active]=tangent
    pt=t[:,:,None]*t[:,None,:]
    return {"tangent":pt,"curve_normal":p-pt,"ambient_normal":np.eye(dimension)[None]-p}


def project(q,vector):return np.einsum("qij,...qj->...qi",q,vector)


def value_split(offset,oracle_offset,frames,scales):
    result={};s2=np.asarray(scales)[:,None]**2
    for name,q in frames.items():
        a=project(q,offset);truth=project(q,oracle_offset);error=a-truth
        result[name+"_correction"]=(a*a).sum(-1)/s2
        result[name+"_oracle_correction"]=(truth*truth).sum(-1)/s2
        result[name+"_oracle_cross"]=2*(truth*error).sum(-1)/s2
        result[name+"_error_energy"]=(error*error).sum(-1)/s2
        np.testing.assert_allclose(result[name+"_correction"],result[name+"_oracle_correction"]+result[name+"_oracle_cross"]+result[name+"_error_energy"],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(sum(result[k+"_correction"] for k in frames),(offset**2).sum(-1)/s2,rtol=1e-10,atol=1e-11)
    return result


def offset_expansion(first,ja,remainder,frames,scales):
    result={};s2=np.asarray(scales)[:,None]**2
    for name,q in frames.items():
        a,b,r=[project(q,v) for v in (first,ja,remainder)]
        terms={"base":(a*a).sum(-1),"linear_cross":2*(a*b).sum(-1),"propagated_energy":(b*b).sum(-1),
            "remainder_cross":2*((a+b)*r).sum(-1),"remainder_energy":(r*r).sum(-1)}
        for k,v in terms.items():result[name+"_"+k]=v/s2
        np.testing.assert_allclose(sum(terms.values()),((a+b+r)**2).sum(-1),rtol=1e-10,atol=1e-11)
    return result


def jacobian_split(j,truth,frames):
    result={}
    for output,qout in frames.items():
        result[output+"_response"]=np.einsum("qij,...qji->...q",qout,j)
        if truth is not None:result[output+"_response_error"]=np.einsum("qij,...qji->...q",qout,j-truth)
        for inp,qin in frames.items():
            error=j if truth is None else j-truth
            block=qout[None]@error@qin[None]
            result[output+"_output_"+inp+"_input_"+("matrix_squared" if truth is None else "error_squared")]=(block**2).sum((-1,-2))
    np.testing.assert_allclose(sum(result[k+"_response"] for k in frames),np.trace(j,axis1=-2,axis2=-1),rtol=1e-10,atol=1e-10)
    suffix="matrix_squared" if truth is None else "error_squared"
    np.testing.assert_allclose(sum(result[o+"_output_"+i+"_input_"+suffix] for o in frames for i in frames),
        ((j if truth is None else j-truth)**2).sum((-1,-2)),rtol=1e-10,atol=1e-10)
    return result


def chain_split(j2,j1,frames):
    result={}
    for dest,q in frames.items():
        for mid,qmid in frames.items():
            product=q[None]@j2@qmid[None]@j1
            result[dest+"_via_"+mid]=np.trace(product,axis1=-2,axis2=-1)
        np.testing.assert_allclose(sum(result[dest+"_via_"+m] for m in frames),np.einsum("qij,...qji->...q",q,j2@j1),rtol=1e-10,atol=1e-10)
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    helpers=[Path(__file__),Path("experiments/response_full_models_20260907_span.py"),
        Path("experiments/response_full_models_20260907_clean_anchor.py"),Path("experiments/response_full_models_20260907_spaghetti_local.py")]
    hashes={str(p):sha256(p) for p in helpers};rows=[];mechanism_rows=[];records=[]
    def load(p):
        hashes[str(p)]=sha256(p)
        if p.with_name("manifest.json").exists():hashes[str(p.with_name("manifest.json"))]=sha256(p.with_name("manifest.json"))
        return np.load(p)
    os=load(args.root/"population_oracle_ve_spaghetti256/arrays.npz")
    osp=load(args.root/"population_oracle_spiral_fixture_v2/fixture_arrays.npz")
    for name in ("vp_spaghetti_posterior","vp_spaghetti_noise","ve_spaghetti_control","ve_spaghetti_clean_anchor","vp_spiral_posterior","vp_spiral_noise"):
        a=load(args.root/"two_pass_composition"/(name+".npz"));o=osp if "spiral" in name else os
        for key in ("queries","query_ids"):np.testing.assert_array_equal(a[key],o[key])
        np.testing.assert_allclose(a["scales"],o["scales"],rtol=1e-9,atol=0)
        if "spiral" in name:
            t=o["latent_t"];tan=np.stack([2*np.cos(t*t)-np.sin(t*t)/t**2,-2*np.sin(t*t)-np.cos(t*t)/t**2],axis=-1)
            tan/=np.linalg.norm(tan,axis=-1,keepdims=True);truth=None
        else:
            tan=tangent_from_phase(o["query_phase"]);truth=np.zeros_like(a["base_jacobian"]);truth[...,:20,:20]=o["jacobian_active"]
        frames=projectors(tan);scales=a["scales"];first=a["base_posterior"]-a["queries"];second=a["posterior"]-a["queries"]
        ja=np.einsum("sqij,sqj->sqi",a["base_jacobian"],first);remainder=second-first-ja
        np.testing.assert_allclose(first+ja,a["local_predicted_offset"],rtol=1e-10,atol=1e-11)
        np.testing.assert_allclose(a["second_jacobian_at_first_output"]@a["base_jacobian"],a["jacobian"],rtol=1e-10,atol=1e-10)
        accumulated={};oracle_offset=o["posterior"]-a["queries"]
        for state,offset,j in (("base",first,a["base_jacobian"]),("composed",second,a["jacobian"]),("local_prediction",first+ja,a["local_squared_jacobian"])):
            vs=value_split(offset,oracle_offset,frames,scales);js=jacobian_split(j,truth,frames)
            data={**vs,**js}
            for k,v in data.items():accumulated[state+"_"+k]=v
            for i,s in enumerate(scales):
                row={"state":name,"readout":state,"lambda":float(s),"count":len(a["queries"]),"has_full_jacobian_oracle":truth is not None,
                    "correction":float((offset[i]**2).sum(-1).mean()/s**2),
                    "response":float(np.trace(j[i],axis1=-2,axis2=-1).mean()),**{k:float(v[i].mean()) for k,v in data.items()}}
                # Same schema across laws: unavailable Spiral error fields remain
                # null; their actual block energies are separately named.
                for out in frames:
                    row.setdefault(out+"_response_error",None)
                    for inp in frames:
                        for suffix in ("matrix_squared","error_squared"):row.setdefault(out+"_output_"+inp+"_input_"+suffix,None)
                rows.append(row)
        expansion=offset_expansion(first,ja,remainder,frames,scales)
        chain=chain_split(a["second_jacobian_at_first_output"],a["base_jacobian"],frames)
        frozen_chain=chain_split(a["base_jacobian"],a["base_jacobian"],frames)
        for k,v in expansion.items():accumulated["offset_"+k]=v
        for k,v in chain.items():accumulated["chain_"+k]=v
        for k,v in frozen_chain.items():accumulated["frozen_chain_"+k]=v
        for i,s in enumerate(scales):
            mechanism_rows.append({"state":name,"lambda":float(s),"count":len(a["queries"]),
                "response_local_prediction_mae":float(abs(a["response"][i]-a["local_predicted_response"][i]).mean()),
                "correction_local_prediction_mae":float(abs(a["correction"][i]-a["local_predicted_correction"][i]).mean()),
                "offset_remainder_energy":float((remainder[i]**2).sum(-1).mean()/s**2),
                **{k:float(v[i].mean()) for k,v in expansion.items()},**{"chain_"+k:float(v[i].mean()) for k,v in chain.items()}})
        np.savez_compressed(args.output/f"{name}.npz",queries=a["queries"],query_ids=a["query_ids"],scales=scales,tangent=tan,**accumulated)
        records.append({"state":name,"all_scales":scales.tolist(),"count":len(a["queries"]),"jacobian_oracle_available":truth is not None})
    write_rows(args.output,"summary",rows);write_rows(args.output,"mechanism",mechanism_rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"records":records,"completed":True,
        "decomposition":"Fixed truecleanT, curve-normal within global span, ambient-null; alloraclevalueerrors versus ORIGINAL b*, not composedpopulationg*",
        "local_expansion":"a2=a1+J1*a1+remainder; Jactual=J(b1(x))*J1, J1² onlytestedlocalprediction",
        "interpretation":"Contribution shares refer to specified energy or energyincrease, never a fraction of totalcausalhistory. Composition is a differentfield.",
        "no_inference_or_training":True,"all_inputs_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
