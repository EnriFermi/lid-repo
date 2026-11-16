"""Training-only affine-span projection of preserved rendered FM posteriors.

The first diagnostic reuses the original64-query/all9-scale exact full traces;
only the projected trace is newly differentiated. The span rank is estimated
from training singular values, never from intrinsic-dimension labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.spiral_spaghetti_convergence import posterior
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b""):digest.update(part)
    return digest.hexdigest()


def fit_affine_span(points,relative_threshold=1e-5):
    """Deterministic full SVD on the declared training subset, float64."""
    if not 0<relative_threshold<1:raise ValueError("Invalid relative rank threshold")
    points=points.double()
    center=points.mean(0)
    _,singular,vh=torch.linalg.svd(points-center,full_matrices=False)
    rank=int((singular>relative_threshold*singular[0]).sum())
    if rank<=0:raise ValueError("Empty affine span")
    return center,vh[:rank].T.contiguous(),singular


def projected_trace(predict,points,basis,batch_size=8):
    """Exact tr(PJ) from rank(P) vector-Jacobian products, no trace sampling."""
    values,traces=[],[]
    for batch in points.split(batch_size):
        inputs=batch.detach().requires_grad_(True)
        output=predict(inputs)
        probes=basis.T.to(inputs)[:,None,:].expand(-1,len(inputs),-1)
        gradients=torch.autograd.grad(output,inputs,grad_outputs=probes,is_grads_batched=True)[0]
        traces.append((gradients*probes).sum((0,2)).detach())
        values.append(output.detach())
    return torch.cat(values),torch.cat(traces)


def correction_components(values,points,center,basis,scale):
    values,points=values.double(),points.double()
    basis,center=basis.to(points),center.to(points)
    error=values-points
    span=(error@basis)@basis.T
    off=error-span
    query_off=points-center-((points-center)@basis)@basis.T
    projected=center+((values-center)@basis)@basis.T
    original=error.square().sum(-1)/scale**2
    span_c=span.square().sum(-1)/scale**2
    off_c=off.square().sum(-1)/scale**2
    q_off_c=query_off.square().sum(-1)/scale**2
    projected_c=(projected-points).square().sum(-1)/scale**2
    torch.testing.assert_close(original,span_c+off_c,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(projected_c,span_c+q_off_c,rtol=1e-10,atol=1e-10)
    return {"correction_original":original,"correction_span":span_c,
        "correction_offspan":off_c,"query_offspan_correction":q_off_c,
        "correction_projected":projected_c}


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.source/"protocol.json").read_text())
    config=protocol["config"]
    scales=config["scales"]
    if args.count!=int(config["query_count"]):
        raise ValueError("First probe must reuse the original64-query arrays")
    manifests,rows=[],[]
    for dataset in args.datasets:
        source=args.source/dataset
        output=args.output/dataset;output.mkdir()
        base_path=source/"base.pt"
        base=torch.load(base_path,map_location="cpu",weights_only=False)
        mean,rms=base["mean"],base["rms"]
        data_root=args.source/"exact_subset/benchmarks"/dataset
        train_raw=np.load(data_root/"train/dataset.npy",mmap_mode="r")
        train_raw=train_raw.reshape(len(train_raw),-1)
        fit_ids=np.load(source/"split_ids.npz")["fit"][:args.fit_count]
        train=(torch.from_numpy(np.array(train_raw[fit_ids],dtype=np.float32))-mean)/rms
        center,basis,singular=fit_affine_span(train,args.relative_rank_threshold)
        test_raw=np.load(data_root/"test/dataset.npy",mmap_mode="r")
        test_raw=test_raw.reshape(len(test_raw),-1)
        points=(torch.from_numpy(np.array(test_raw[:args.count],dtype=np.float32))-mean)/rms
        train_off=train.double()-center-((train.double()-center)@basis)@basis.T
        test_off=points.double()-center-((points.double()-center)@basis)@basis.T
        record={"dataset":dataset,"fit_count":len(fit_ids),"rank":basis.shape[1],
            "relative_rank_threshold":args.relative_rank_threshold,
            "fit_subset":"First declared fit IDs, excluding original selection holdout",
            "top_singular_values":singular[:25].tolist(),"smallest_retained_singular":float(singular[basis.shape[1]-1]),
            "largest_discarded_singular":float(singular[basis.shape[1]]) if basis.shape[1]<len(singular) else None,
            "train_offspan_rms_per_coordinate":float(train_off.square().mean().sqrt()),
            "test_offspan_rms_per_coordinate":float(test_off.square().mean().sqrt()),
            "test_offspan_max_norm":float(test_off.norm(dim=-1).max()),
            "normalization_source":str(base_path),"base_sha256":sha256(base_path),
            "normalization_rms":float(rms),"intrinsic_dimension":1,
            "label_note":"Both audited generators have one latent parameter; Spaghetti train label20 is not its generic intrinsic dimension",
            "stored_test_labels":np.unique(np.load(data_root/"test/lid.npy")).tolist(),
            "stages":{},"query_ids":list(range(args.count))}
        torch.save({"center":center,"basis":basis,"singular_values":singular,
            "queries":points,"fit_ids":torch.from_numpy(fit_ids),"mean":mean,"rms":rms},output/"span_and_queries.pt")
        for stage in args.stages:
            path=source/f"{stage}.pt";digest=sha256(path)
            payload=torch.load(path,map_location="cpu",weights_only=False)
            architecture=payload.get("architecture",dict(ambient_dim=points.shape[1],hidden_dim=config["hidden_dim"],
                depth=config["depth"],condition_dim=config["condition_dim"],zero_init_output=True))
            field=ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(architecture)).to(args.device).eval()
            field.load_state_dict(payload["model_state"],strict=True)
            original_path=source/f"{stage}_predictions.npz"
            original=np.load(original_path)
            arrays={"posterior":[],"response_original":[],"response_span":[],"response_offspan":[]}
            maximum_replay_error=0.
            for index,scale in enumerate(scales):
                value,span_r=projected_trace(lambda p:posterior(field,p,scale),points.to(args.device),basis.to(args.device))
                value,span_r=value.cpu(),span_r.double().cpu()
                terms=correction_components(value,points,center,basis,scale)
                original_r=torch.from_numpy(original["test_response"][index]).double()
                replay_error=float((terms["correction_original"]-torch.from_numpy(original["test_correction"][index])).abs().max())
                maximum_replay_error=max(maximum_replay_error,replay_error)
                np.testing.assert_allclose(terms["correction_original"].numpy(),original["test_correction"][index],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(original["test_full"][index],original["test_response"][index]+original["test_correction"][index],rtol=1e-12,atol=1e-12)
                arrays["posterior"].append(value.numpy())
                for name,tensor in {"response_original":original_r,"response_span":span_r,"response_offspan":original_r-span_r,**terms}.items():
                    arrays.setdefault(name,[]).append(tensor.numpy())
                original_full=original_r+terms["correction_original"]
                projected_full=span_r+terms["correction_projected"]
                arrays.setdefault("full_original",[]).append(original_full.numpy())
                arrays.setdefault("full_projected",[]).append(projected_full.numpy())
                row={"dataset":dataset,"stage":stage,"lambda":scale,"count":len(points),"span_rank":basis.shape[1],
                    "response_mae":float((original_r-1).abs().mean()),"full_mae":float((original_full-1).abs().mean()),
                    "projected_response_mae":float((span_r-1).abs().mean()),"projected_full_mae":float((projected_full-1).abs().mean()),
                    "response_mean":float(original_r.mean()),"projected_response_mean":float(span_r.mean()),
                    "offspan_response_mean":float((original_r-span_r).mean()),
                    **{name+"_mean":float(value.mean()) for name,value in terms.items()},
                    "correction_harm_fraction":float(((original_full-1).abs()>(original_r-1).abs()).double().mean()),
                    "projected_correction_harm_fraction":float(((projected_full-1).abs()>(span_r-1).abs()).double().mean())}
                rows.append(row)
                print(json.dumps(row),flush=True)
            np.savez_compressed(output/f"{stage}_arrays.npz",**{k:np.stack(v) for k,v in arrays.items()})
            assert sha256(path)==digest
            record["stages"][stage]={"checkpoint":str(path),"sha256":digest,
                "source_arrays_sha256":sha256(original_path),"maximum_correction_replay_error":maximum_replay_error}
        assert sha256(base_path)==record["base_sha256"]
        manifests.append(record)
    manifest={"source":str(args.source),"device":args.device,"model_dtype":"float32",
        "component_arithmetic_dtype":"float64","full_trace":"Preserved original exact coordinate trace, not Hutchinson",
        "projected_trace":"Exact rank-many batched VJPs; original query and model precision",
        "scales":scales,"source_sha256":sha256(__file__),"datasets":manifests,
        "scope":"Global training-data affine-span diagnostic; no local-d labels used by intervention; no new model training"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {"rows":len(rows),"output":str(args.output),"all_checkpoints_unchanged":True}


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source",type=Path,default=Path("artifacts/spiral_spaghetti_convergence_20260905/run"))
    result.add_argument("--output",type=Path,required=True)
    result.add_argument("--device",default="cuda:1")
    result.add_argument("--datasets",nargs="+",default=["e1_spiral_pca","e8_spaghetti_pca"])
    result.add_argument("--stages",nargs="+",default=["base","lower_lr_average"])
    result.add_argument("--count",type=int,default=64)
    result.add_argument("--fit-count",type=int,default=4096)
    result.add_argument("--relative-rank-threshold",type=float,default=1e-5)
    return result


if __name__=="__main__":print(json.dumps(run(parser().parse_args())),flush=True)
