"""Fixed-checkpoint normal-bundle spectrum diagnostic on Spaghetti.

Follows Stanczuk et al. ICML2024 Algorithm1: K=4D uncentered score vectors,
largest ABSOLUTE adjacent singular-value gap. The training campaign is ours,
not a reproduction of that paper. No geometric labels enter the selector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_training_20260907 import posterior
from models.training import diffusion_denoiser_output, load_checkpoint


OUT = Path("artifacts/response_full_20260907/nb_spaghetti_review")
SCALES = [.01, .1, .316227766]
VP = Path("artifacts/response_full_20260907/vp_spaghetti_paired")
VE = Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective/e8_spaghetti_pca/model.pt")
REFINED = Path("artifacts/response_full_20260907/ve_spaghetti_refinement/final.pt")


def estimate_from_spectrum(singular):
    singular = np.asarray(singular)
    if singular.shape[-1] < 2 or not np.isfinite(singular).all():
        raise ValueError("Finite spectrum of dimension at least two required")
    normal_rank = np.argmax(singular[..., :-1]-singular[..., 1:], axis=-1)+1
    return singular.shape[-1]-normal_rank


def analyze_scores(scores, tangent):
    """Input [queries,K,D]; U columns are ambient-space left singular vectors."""
    scores, tangent = np.asarray(scores), np.asarray(tangent)
    left, singular, _ = np.linalg.svd(scores.transpose(0,2,1), full_matrices=False)
    estimate = estimate_from_spectrum(singular)
    cosine = np.abs(np.einsum("bi,bi->b",left[:,:,-1],tangent))
    captured = np.array([np.square(left[i,:,scores.shape[-1]-d:].T@tangent[i]).sum()
                         for i,d in enumerate(estimate)])
    return {"singular_values":singular,"left_singular_vectors":left,"dimension_estimate":estimate,
            "smallest_vector_tangent_cosine":cosine,"estimated_tangent_subspace_capture":captured}


def prepare(out=OUT):
    out.mkdir(parents=True,exist_ok=False)
    f=torch.load(VP/"fixture.pt",map_location="cpu",weights_only=False)
    x=f["queries"][:64].double().numpy();ids=f["query_ids"][:64]
    seed=2026090709
    noise=np.random.default_rng(seed).standard_normal((len(x),4*x.shape[1],x.shape[1]))
    phase=np.random.RandomState(0).uniform(0,2*np.pi,102000)[101000+ids]
    tangent=np.zeros_like(x);freq=np.arange(2,22)
    tangent[:,:20]=freq*np.cos(phase[:,None]*freq)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    np.savez_compressed(out/"tape.npz",queries=x,query_ids=ids,query_raw=f["query_raw"][:64],
        noise=noise,scales=np.asarray(SCALES),normalization_mean=f["mean"].numpy(),
        normalization_rms=np.array(f["rms"]),query_phase=phase,tangent=tangent)
    sources=[Path(__file__),Path("models/neural_fields.py"),Path("models/training.py"),
        Path("experiments/response_full_training_20260907.py"),
        Path("experiments/response_full_models_20260907_ve_refinement.py"),
        VP/"fixture.pt",VP/"final.pt",VE,REFINED]
    manifest={"source_hashes":{str(p):sha256(p) for p in sources},"tape_sha256":sha256(out/"tape.npz"),
        "seed":seed,"K":120,"D":30,"queries":64,"scales":SCALES,
        "rule":"D - argmax_{i=1,...,D-1}(s_i-s_{i+1}); descending singular values, absolute gap, no centering",
        "score":"(b(x+lambda*z,lambda)-(x+lambda*z))/lambda^2 in canonical coordinates",
        "primary_reference":"https://raw.githubusercontent.com/mlresearch/v235/main/assets/stanczuk24a/stanczuk24a.pdf",
        "scope":"Core Algorithm1 diagnostic on our fixed models; no training reproduction or LID tuning",
        "true_geometry":"d1 and phase-derived tangent used for evaluation only"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("TAPE_READY "+str(out/"tape.npz"),flush=True)


def evaluate(out=OUT):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((out/"manifest.json").read_text())
    assert sha256(out/"tape.npz")==manifest["tape_sha256"]
    assert all(sha256(p)==h for p,h in manifest["source_hashes"].items())
    if (out/"summary.json").exists():raise FileExistsError(out/"summary.json")
    tape=np.load(out/"tape.npz");x=tape["queries"];z=tape["noise"]
    payload=torch.load(VP/"final.pt",map_location="cpu",weights_only=False)
    ve=load_checkpoint(VE,device="cpu");ve.model.double().eval()
    predictors={"ve_original":lambda q,s:diffusion_denoiser_output(ve.model,q,s,preconditioning="edm",data_std=1.)}
    for kind in ("posterior","noise"):
        model=load_field(payload,kind,"average","cpu")
        predictors["vp_"+kind]=lambda q,s,m=model,k=kind:posterior(m,q,s,k)
    for arm in ("original_lr","low_lr"):
        model=load_refinement_field(REFINED,arm,"average","cpu")
        predictors["ve_refined_"+arm]=lambda q,s,m=model:diffusion_denoiser_output(m,q,s,preconditioning="edm",data_std=1.)
    rows=[];start=time.monotonic()
    for name,predict in predictors.items():
        accumulated={}
        for scale in tape["scales"]:
            observations=x[:,None,:]+scale*z
            tensor=torch.from_numpy(observations.reshape(-1,x.shape[-1]))
            with torch.no_grad():
                b=torch.cat([predict(q,float(scale)) for q in tensor.split(256)]).numpy().reshape(observations.shape)
            scores=(b-observations)/scale**2
            result=analyze_scores(scores,tape["tangent"])
            for key,value in {"scores":scores,**result}.items():accumulated.setdefault(key,[]).append(value)
            values,counts=np.unique(result["dimension_estimate"],return_counts=True)
            row={"model":name,"lambda":float(scale),"count":len(x),
                "dimension_mean":float(result["dimension_estimate"].mean()),
                "dimension_mae":float(np.abs(result["dimension_estimate"]-1).mean()),
                "fraction_exact_d1":float(np.mean(result["dimension_estimate"]==1)),
                "smallest_vector_tangent_cosine_mean":float(result["smallest_vector_tangent_cosine"].mean()),
                "smallest_vector_tangent_cosine_min":float(result["smallest_vector_tangent_cosine"].min()),
                "estimated_tangent_capture_mean":float(result["estimated_tangent_subspace_capture"].mean()),
                "dimension_counts":dict(zip(values.tolist(),counts.tolist()))}
            rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(out/f"{name}_arrays.npz",scales=tape["scales"],query_ids=tape["query_ids"],
            **{k:np.stack(v) for k,v in accumulated.items()})
    assert all(sha256(p)==h for p,h in manifest["source_hashes"].items())
    (out/"summary.json").write_text(json.dumps({"rows":rows,"all_sources_unchanged":True,
        "elapsed_seconds":time.monotonic()-start},indent=2)+"\n")
    with (out/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","evaluate"])
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    (prepare if args.mode=="prepare" else evaluate)(args.output)
