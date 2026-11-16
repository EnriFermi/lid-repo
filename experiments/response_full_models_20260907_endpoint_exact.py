"""Exact-latent Spiral replay of the frozen endpoint inference control.

This avoids attributing float32 off-support rounding to tiny-scale geometry.
All lambda<.01 results remain out-of-training-range diagnostics, not a benchmark.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_endpoint import head_value_jacobian,stable_terms
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from models.training import load_checkpoint,diffusion_denoiser_output


def validate_fixture(fixture,mean,rms):
    q=np.asarray(fixture["queries"]);raw=np.asarray(fixture["query_raw"])
    np.testing.assert_array_equal(fixture["normalization_mean"],np.asarray(mean))
    assert float(fixture["normalization_rms"])==float(rms)
    assert q.dtype==np.float64 and raw.dtype==np.float64 and q.shape==raw.shape==(64,30)
    np.testing.assert_allclose(q,(raw-np.asarray(mean))/rms,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(fixture["scales"],np.logspace(-5,-2,13),rtol=1e-13,atol=0)
    assert len(np.unique(fixture["query_ids"]))==64


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    hashes={str(p):sha256(p) for p in (Path(__file__),args.ve_fixture,args.vp_fixture,
        Path("experiments/response_full_models_20260907_endpoint.py"),Path("models/neural_fields.py"),
        Path("models/training.py"),Path("experiments/response_full_training_20260907.py"))}
    rows=[];records=[];gates=[]
    for family in ("ve_edm","vp_noise","vp_posterior"):
        fixture_path=args.ve_fixture if family=="ve_edm" else args.vp_fixture
        fixture=np.load(fixture_path)
        if family=="ve_edm":
            checkpoint=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective/e1_spiral_pca/model.pt")
            loaded=load_checkpoint(checkpoint,device=args.device);field=loaded.model.double().eval()
            mean=loaded.normalization_mean.cpu().numpy();rms=loaded.normalization_scale
            data_std=float(loaded.config.diffusion_data_std)
            wrapper=lambda x,s:diffusion_denoiser_output(field,x,s,preconditioning="edm",data_std=data_std)
            mode="preserved_best"
        else:
            checkpoint=args.root/"vp_spiral_paired/final.pt"
            payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
            field=load_field(payload,family[3:],"average",args.device)
            mean=payload["normalization_mean"].cpu().numpy();rms=payload["normalization_scale"]
            data_std=1.;wrapper=lambda x,s:posterior(field,x,s,family[3:]);mode="average"
        hashes[str(checkpoint)]=sha256(checkpoint);validate_fixture(fixture,mean,rms)
        q=torch.from_numpy(fixture["queries"]).to(args.device);accumulated={}
        for i,s in enumerate(fixture["scales"]):
            s=float(s);cin=1/math.sqrt(s*s+(data_std**2 if family=="ve_edm" else 1.))
            condition=s if family=="ve_edm" else math.log(s)
            with torch.no_grad():
                h,j=head_value_jacobian(field,cin*q,condition)
                data=stable_terms(h,j,q,s,family,data_std)
            if i in (0,len(fixture["scales"])-1):
                b,jb=active_jacobian(lambda x:wrapper(x,s),q[:2],active=30,batch_size=2)
                model_j=data["canonical_jacobian_minus_identity"][:2]+torch.eye(30,dtype=q.dtype,device=q.device)[None]
                torch.testing.assert_close(b,data["posterior"][:2],rtol=1e-10,atol=1e-10)
                torch.testing.assert_close(jb,model_j,rtol=1e-10,atol=1e-10)
                gates.append({"family":family,"lambda":s,"max_autograd_jacobian_error":float((jb-model_j).abs().max())})
            for key,value in data.items():accumulated.setdefault(key,[]).append(value.cpu().numpy())
            rows.append({"family":family,"mode":mode,"lambda":s,"out_of_training_range":s<.01,
                "response_mean":float(data["response"].mean()),"correction_mean":float(data["correction"].mean()),
                "full_mean":float(data["full"].mean()),"response_minus_ambient_abs_max":float(data["response_minus_ambient"].abs().max())})
        np.savez_compressed(args.output/f"spiral_{family}.npz",**{k:fixture[k] for k in ("queries","query_raw","query_ids","scales",
            "normalization_mean","normalization_rms","latent_t","frozen_queries")},
            out_of_training_range=np.asarray(fixture["scales"])<.01,
            **{key:np.stack(value) for key,value in accumulated.items()})
        records.append({"family":family,"checkpoint":str(checkpoint),"checkpoint_sha256":hashes[str(checkpoint)],
            "fixture":str(fixture_path),"fixture_sha256":hashes[str(fixture_path)],"mode":mode,
            "maximum_old_float32_query_displacement":float(np.linalg.norm(fixture["frozen_queries"]-fixture["queries"],axis=-1).max())})
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"records":records,"gates":gates,
        "query_contract":"Same64 latent IDs; double-precision exact generator points, normalized by each original checkpoint mean/rms",
        "scope":"Exact-query population/model scale-window bridge; allbelow.01 extrapolation, not in-range benchmark performance",
        "no_training_or_model_selection":True,"all_sources_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ve-fixture",type=Path,required=True);p.add_argument("--vp-fixture",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
