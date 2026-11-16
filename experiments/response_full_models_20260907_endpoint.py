"""Fixed-model endpoint extrapolation, explicitly not benchmark evaluation.

All lambda<.01 observations are outside training support. Analytic residual
algebra avoids subtracting two almost identical posterior/data coordinates.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from experiments.fm_mechanism_fast_trace import _linear_jacobian,_layer_norm_value_jacobian,_silu_value_jacobian
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from models.training import load_checkpoint,diffusion_denoiser_output

SCALES=np.logspace(-8,0,33)


def head_value_jacobian(field,inputs,condition):
    """Exact Jacobian with respect to H's native spatial input, not canonical r."""
    condition=torch.full((len(inputs),),float(condition),dtype=inputs.dtype,device=inputs.device)
    embedded=field.condition_embedding(condition)
    state=field.input(inputs)+field.input_condition(embedded)
    jac=field.input.weight[None].expand(len(inputs),-1,-1)
    for block in field.blocks:
        h,j=_layer_norm_value_jacobian(block.norm,state,jac)
        h=block.hidden(h)+block.condition(embedded);j=_linear_jacobian(block.hidden,j)
        h,j=_silu_value_jacobian(h,j)
        state=(state+block.output(h))*2**-.5
        jac=(jac+_linear_jacobian(block.output,j))*2**-.5
    h,j=_layer_norm_value_jacobian(field.output[0],state,jac)
    h,j=_silu_value_jacobian(h,j)
    return field.output[-1](h),_linear_jacobian(field.output[-1],j)


def stable_terms(raw,native_jac,queries,scale,family,data_std=1.):
    """Canonical R and C using exact wrapper algebra without b-r cancellation."""
    d=queries.shape[-1];eye=torch.eye(d,dtype=queries.dtype,device=queries.device)[None]
    if family in ("vp_noise","vp_posterior"):
        cin=1/math.sqrt(1+scale**2);canonical_jac=cin*native_jac
        if family=="vp_noise":
            normalized_displacement=-raw
            posterior_value=queries-scale*raw
            jacobian_minus_identity=-scale*canonical_jac
            response_minus_d=-scale*torch.diagonal(canonical_jac,dim1=-2,dim2=-1).sum(-1)
        else:
            normalized_displacement=(raw-queries)/scale
            posterior_value=raw;jacobian_minus_identity=canonical_jac-eye
            response_minus_d=torch.diagonal(canonical_jac,dim1=-2,dim2=-1).sum(-1)-d
    elif family=="ve_edm":
        denom=scale**2+data_std**2;cin=1/math.sqrt(denom)
        cout=scale*data_std/math.sqrt(denom);skip=data_std**2/denom
        canonical_jac=cin*native_jac
        normalized_displacement=-scale/denom*queries+data_std/math.sqrt(denom)*raw
        posterior_value=skip*queries+cout*raw
        jacobian_minus_identity=-scale**2/denom*eye+cout*canonical_jac
        response_minus_d=-d*scale**2/denom+cout*torch.diagonal(canonical_jac,dim1=-2,dim2=-1).sum(-1)
    else:raise ValueError(family)
    response=d+response_minus_d;correction=normalized_displacement.square().sum(-1)
    return {"posterior":posterior_value,"response":response,"correction":correction,"full":response+correction,
        "response_minus_ambient":response_minus_d,"normalized_displacement":normalized_displacement,
        "head_value":raw,"head_native_jacobian":native_jac,"head_canonical_jacobian":canonical_jac,
        "head_native_trace":torch.diagonal(native_jac,dim1=-2,dim2=-1).sum(-1),
        "head_canonical_trace":torch.diagonal(canonical_jac,dim1=-2,dim2=-1).sum(-1),
        "head_native_jacobian_frobenius":native_jac.square().sum((-1,-2)).sqrt(),
        "head_norm":raw.square().sum(-1).sqrt(),
        "canonical_jacobian_minus_identity":jacobian_minus_identity,
        "canonical_jacobian_minus_identity_frobenius":jacobian_minus_identity.square().sum((-1,-2)).sqrt()}


def cases(root):
    for law,dataset in (("spaghetti","e8_spaghetti_pca"),("spiral","e1_spiral_pca")):
        ve_arrays=root/("ve_spaghetti_shared256" if law=="spaghetti" else "ve_shared256")/dataset/"arrays.npz"
        yield law,"ve_edm","preserved_best",Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective")/dataset/"model.pt",ve_arrays
        for family in ("noise","posterior"):
            yield law,"vp_"+family,"average",root/f"vp_{law}_paired/final.pt",root/f"vp_{law}_span/{family}_average_arrays.npz"


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=[Path(__file__),Path("models/neural_fields.py"),Path("models/training.py"),
        Path("experiments/fm_mechanism_fast_trace.py"),Path("experiments/response_full_training_20260907.py"),
        Path("experiments/response_full_models_20260907_vp.py"),Path("experiments/response_full_models_20260907_spaghetti_local.py")]
    hashes={str(p):sha256(p) for p in inputs};rows=[];records=[];gates=[]
    for law,family,mode,checkpoint,arrays_path in cases(args.root):
        hashes[str(checkpoint)]=sha256(checkpoint);hashes[str(arrays_path)]=sha256(arrays_path)
        saved=np.load(arrays_path);q=torch.from_numpy(saved["queries"][:64]).double().to(args.device)
        if family=="ve_edm":
            loaded=load_checkpoint(checkpoint,device=args.device);field=loaded.model.double().eval()
            data_std=float(loaded.config.diffusion_data_std)
            assert loaded.config.diffusion_preconditioning=="edm"
            wrapper=lambda x,s:diffusion_denoiser_output(field,x,s,preconditioning="edm",data_std=data_std)
            train_range=[float(loaded.config.sigma_min),float(loaded.config.sigma_max)]
        else:
            payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
            field=load_field(payload,family.removeprefix("vp_"),mode,args.device);data_std=1.
            wrapper=lambda x,s:posterior(field,x,s,family.removeprefix("vp_"))
            train_range=[payload["config"]["lambda_min"],payload["config"]["lambda_max"]]
        assert field.config.dropout==0 and isinstance(field.output[0],torch.nn.LayerNorm)
        assert all(block.norm.eps>0 for block in field.blocks) and field.output[0].eps>0
        records.append({"law":law,"family":family,"mode":mode,"checkpoint":str(checkpoint),
            "checkpoint_sha256":hashes[str(checkpoint)],"architecture":field.config.to_dict(),"data_std":data_std,
            "training_scale_range":train_range,"query_arrays":str(arrays_path),"count":len(q),
            "query_ids":saved["query_ids"][:64].tolist(),"normalization_rms":float(saved["normalization_rms"]),
            "below_training_range":"ALL lambda<.01 are extrapolation, not benchmark performance",
            "native_condition":"sigma with internal log transform" if family=="ve_edm" else "log lambda with linear transform"})
        accumulated={}
        for scale in SCALES:
            scale=float(scale);den=scale**2+(data_std**2 if family=="ve_edm" else 1.)
            native=q/math.sqrt(den);condition=scale if family=="ve_edm" else math.log(scale)
            with torch.no_grad():
                raw,j=head_value_jacobian(field,native,condition)
                data=stable_terms(raw,j,q,scale,family,data_std)
            if any(np.isclose(scale,s,rtol=1e-12,atol=0) for s in (1e-8,1e-4,.01,1.)):
                value,replay_j=active_jacobian(lambda x:wrapper(x,scale),q[:2],active=q.shape[-1],batch_size=2)
                stable_j=data["canonical_jacobian_minus_identity"][:2]+torch.eye(q.shape[-1],device=q.device,dtype=q.dtype)[None]
                torch.testing.assert_close(value,data["posterior"][:2],atol=1e-10,rtol=1e-10)
                torch.testing.assert_close(replay_j,stable_j,atol=1e-10,rtol=1e-10)
                direct=((value-q[:2])/scale).square().sum(-1)
                # Tiny-scale direct subtraction is diagnostic only. Exact stable
                # residual tests below prevent mistaking cancellation for C=0.
                rel=(direct-data["correction"][:2]).abs()/(1+data["correction"][:2])
                if scale>=1e-4:torch.testing.assert_close(direct,data["correction"][:2],atol=1e-8,rtol=1e-8)
                gates.append({"law":law,"family":family,"lambda":scale,
                    "max_value_difference":float((value-data["posterior"][:2]).abs().max()),
                    "max_jacobian_difference":float((replay_j-stable_j).abs().max()),
                    "max_direct_vs_stable_c_relative_to_1_plus_c":float(rel.max())})
            if any(np.isclose(scale,s,rtol=1e-12,atol=0) for s in (.01,.1,1.)):
                index=int(np.argmin(abs(saved["scales"]-scale)))
                for key,oldkey in (("posterior","posterior"),("response","response"),("correction","correction_original")):
                    np.testing.assert_allclose(data[key].cpu().numpy(),saved[oldkey][index,:64],rtol=1e-9,atol=1e-9)
            for key,value in data.items():accumulated.setdefault(key,[]).append(value.cpu().numpy())
            rows.append({"law":law,"family":family,"mode":mode,"lambda":scale,"ambient_dim":q.shape[-1],
                "out_of_training_range":scale<train_range[0],
                **{key+"_mean":float(value.mean()) for key,value in data.items() if value.ndim==1},
                "head_native_jacobian_frobenius_max":float(data["head_native_jacobian_frobenius"].max()),
                "head_norm_max":float(data["head_norm"].max()),
                "response_minus_ambient_abs_max":float(data["response_minus_ambient"].abs().max())})
        np.savez_compressed(args.output/f"{law}_{family}.npz",scales=SCALES,queries=q.cpu().numpy(),
            query_ids=saved["query_ids"][:64],normalization_rms=saved["normalization_rms"],
            normalization_mean=saved["normalization_mean"],out_of_training_range=SCALES<train_range[0],
            **{key:np.stack(value) for key,value in accumulated.items()})
        print(json.dumps({"law":law,"family":family,"complete_scales":len(SCALES)}),flush=True)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert all(sha256(path)==value for path,value in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"cases":records,"scales":SCALES.tolist(),
        "gates":gates,"dtype":"float64 native-input full Jacobian; canonical affine factors explicit",
        "question":"Fixed architecture endpoint obstruction, not a measured Gaussian tube or an explanation of all in-range errors",
        "response_claim":"Bounded native spatial J forces VP-noise and EDM response to ambient D",
        "full_claim":"C need not vanish or converge; full does NOT automatically tend D",
        "sampling":"Original fixed first64 test queries; no noisy draws, training or model/scale selection",
        "below_support_warning":"All scales below.01 are out-of-training-range extrapolation, not benchmark failure/improvement",
        "source_checkpoints_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
