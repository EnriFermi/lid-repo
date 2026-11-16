"""Saved-array Spaghetti mechanism audit for the one fixed residual calibration.

No fitting/inference: retain both heads and allnine scales; test matrix error,
signed sensitivity cancellation and posterior-displacement components.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.response_full_composition_20260907 import ROOT
from experiments.response_full_models_20260907_composition_bridge import projectors, jacobian_split, value_split
from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase
from experiments.response_full_models_20260907_span import sha256


def symmetric_mass(matrix):
    eigen=np.linalg.eigvalsh(.5*(matrix+matrix.swapaxes(-1,-2)))
    return dict(positive=np.maximum(eigen,0).sum(-1),
                negative=np.maximum(-eigen,0).sum(-1))


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    dependencies=[Path(__file__),Path("experiments/response_full_composition_20260907.py"),
        Path("experiments/response_full_models_20260907_composition_bridge.py"),
        Path("experiments/response_full_models_20260907_spaghetti_local.py"),
        Path("experiments/response_full_models_20260907_span.py")]
    hashes={str(p):sha256(p) for p in dependencies}
    def load(path):
        hashes[str(path)]=sha256(path)
        return np.load(path)
    oracle=load(ROOT/"population_oracle_ve_spaghetti256/arrays.npz")
    frames=projectors(tangent_from_phase(oracle["query_phase"]))
    normal=frames["curve_normal"][None]
    rows=[]
    for head in ["posterior","noise"]:
        case="vp_spaghetti_"+head
        a=load(ROOT/"two_pass_composition"/(case+".npz"))
        b=load(ROOT/"residual_step_calibration"/(case+".npz"))
        for key in ["queries","query_ids"]:
            np.testing.assert_array_equal(a[key],oracle[key])
        np.testing.assert_array_equal(a["queries"],b["test_queries"])
        np.testing.assert_array_equal(a["query_ids"],b["test_query_ids"])
        np.testing.assert_array_equal(a["scales"],b["scales"])
        np.testing.assert_allclose(a["scales"],oracle["scales"],rtol=1e-9,atol=0)
        truth=np.zeros_like(a["base_jacobian"])
        truth[...,:20,:20]=oracle["jacobian_active"]
        eta=b["eta"][:,None,None,None]
        jac=a["base_jacobian"]+eta*(a["jacobian"]-a["base_jacobian"])
        np.testing.assert_allclose(np.trace(jac,axis1=-2,axis2=-1),b["test_response"],rtol=1e-10,atol=1e-11)
        values=a["base_posterior"]+eta[...,0]*(a["posterior"]-a["base_posterior"])
        np.testing.assert_allclose(values,b["test_posterior"],rtol=1e-10,atol=1e-11)
        s2=a["scales"][:,None]**2
        oc=oracle["posterior"]-a["queries"]
        true_c=(oc*oc).sum(-1)/s2
        true_r=np.trace(truth,axis1=-2,axis2=-1)
        out={}
        for name,v,j in [("base",a["base_posterior"],a["base_jacobian"]),
                         ("two",a["posterior"],a["jacobian"]),("blend",values,jac)]:
            offset=v-a["queries"]
            data={**value_split(offset,oc,frames,a["scales"]),**jacobian_split(j,truth,frames)}
            data["whole_jacobian_error_squared"]=((j-truth)**2).sum((-1,-2))
            for key,value in symmetric_mass(normal@j@normal).items():
                data["curve_normal_symmetric_"+key+"_mass"]=value
            response=np.trace(j,axis1=-2,axis2=-1)
            correction=(offset*offset).sum(-1)/s2
            data.update(response=response,correction=correction,full=response+correction,
                response_dimension_absolute_error=abs(response-1),
                full_dimension_absolute_error=abs(response+correction-1),
                response_oracle_absolute_error=abs(response-true_r),
                full_oracle_absolute_error=abs(response+correction-true_r-true_c))
            for key,value in data.items():out[name+"_"+key]=value
            for i,scale in enumerate(a["scales"]):
                rows.append(dict(case=case,operator=name,scale=float(scale),eta=float(b["eta"][i]),
                    **{k:float(v[i].mean()) for k,v in data.items()}))
        np.savez_compressed(args.output/(case+".npz"),query_ids=a["query_ids"],queries=a["queries"],
            scales=a["scales"],eta=b["eta"],blend_jacobian=jac,**out)
    assert all(sha256(p)==digest for p,digest in hashes.items())
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(dict(completed=True,sources=hashes,
        sources_unchanged=True,no_fit_or_inference=True,
        contract="BothSpaghetti heads/all9/256queries; actualfrozeneta linearlycombines exactJ1,J2; truecontinuousposterior/localframes used ONLY diagnostics",
        eigen_scope="Positive/negative masses of symmetrized within-span curve-normal block, not rawnonsymmetric eigenvalues or a newclippedfield",
        oracle_scope="Originalposterior b*, not a composedoracle; serialization scales differ at most3.385e-10relative",
    ),indent=2)+"\n")
    print(json.dumps([r for r in rows if r["case"]=="vp_spaghetti_noise" and abs(r["scale"]-.0177827941)<1e-12],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
