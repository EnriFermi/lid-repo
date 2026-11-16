"""Paired fresh-Adam VE refinement: original constant LR versus a small LR.

Portable source checkpoints omit optimizer/RNG state. BOTH arms reset AdamW;
this is not an exact historical continuation. They share epoch permutations,
the actual original EDM loss, all sigma/noise draws and4096 update steps.
"""
from __future__ import annotations
import argparse,copy,csv,json,math,time
from pathlib import Path
import numpy as np
import torch

from experiments.fm_mechanism_fast_trace import _linear_jacobian,_layer_norm_value_jacobian,_silu_value_jacobian
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve import SCALES
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS,make_splits,_array_sha256
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField
from models.training import diffusion_ve_dsm_loss,load_checkpoint


def edm_readout(field,x,sigma,data_std=1.):
    """Exact full EDM denoiser value/trace, including LOG internal conditioning."""
    if x.dtype!=torch.float64 or field.input.weight.dtype!=torch.float64:
        raise ValueError("Exact diagnostic uses float64")
    s=torch.full((len(x),),float(sigma),device=x.device,dtype=x.dtype)
    denom=s.square()+data_std**2
    cin=denom.rsqrt();cskip=data_std**2/denom;cout=s*data_std/denom.sqrt()
    embedded=field.condition_embedding(s)
    state=field.input(cin[:,None]*x)+field.input_condition(embedded)
    jac=field.input.weight[None].expand(len(x),-1,-1)*cin[:,None,None]
    for block in field.blocks:
        h,j=_layer_norm_value_jacobian(block.norm,state,jac)
        h=block.hidden(h)+block.condition(embedded)
        j=_linear_jacobian(block.hidden,j)
        h,j=_silu_value_jacobian(h,j)
        state=(state+block.output(h))*2**-.5
        jac=(jac+_linear_jacobian(block.output,j))*2**-.5
    h,j=_layer_norm_value_jacobian(field.output[0],state,jac)
    h,j=_silu_value_jacobian(h,j)
    raw=field.output[-1](h)
    tr=torch.einsum("ih,bhi->b",field.output[-1].weight,j)
    b=cskip[:,None]*x+cout[:,None]*raw
    r=x.shape[-1]*cskip+cout*tr
    c=(b-x).square().sum(-1)/sigma**2
    return b,r,c


def original_loss(model,x,cfg,generator):
    return diffusion_ve_dsm_loss(model,x,sigma_min=cfg["sigma_min"],sigma_max=cfg["sigma_max"],
        generator=generator,loss_weighting=cfg["diffusion_loss_weighting"],
        data_std=cfg["diffusion_data_std"],preconditioning=cfg["diffusion_preconditioning"])


def state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def save_checkpoint(path,fields,opts,objective_rng,shuffle_rng,source,step,average=None):
    torch.save({"schema":"ve_paired_refinement_v1","step":step,
        "architecture":source["architecture"],"training_config":source["training_config"],
        "normalization":source["normalization"],"models":{k:state_cpu(v) for k,v in fields.items()},
        "optimizers":{k:v.state_dict() for k,v in opts.items()},
        "objective_rng":{k:v.get_state() for k,v in objective_rng.items()},
        "shuffle_rng":shuffle_rng.get_state(),"averages":average},path)


def load_refinement_field(path,arm,mode="average",device="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    field=ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    field.load_state_dict(payload["averages"if mode=="average"else"models"][arm])
    return field.to(device).double().eval()


def diagnostics(field,queries,scales):
    model=copy.deepcopy(field).double().eval()
    result={"posterior":[],"response":[],"correction_original":[],"full":[]}
    with torch.no_grad():
        for s in scales:
            batches=[edm_readout(model,q,s) for q in queries.double().split(32)]
            b,r,c=[torch.cat([v[i] for v in batches]).cpu().numpy() for i in range(3)]
            for key,value in [("posterior",b),("response",r),("correction_original",c),("full",r+c)]:
                result[key].append(value)
    return {k:np.asarray(v) for k,v in result.items()}


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    before=sha256(args.source)
    source=torch.load(args.source,map_location="cpu",weights_only=True)
    loaded=load_checkpoint(args.source,device=args.device)
    cfg=source["training_config"]
    if cfg["diffusion_preconditioning"]!="edm" or cfg["diffusion_data_std"]!=1. or cfg["dropout"]!=0:
        raise ValueError("This fixed diagnostic covers preserved EDM/data_std1/dropout0")
    if args.average_steps>args.steps or args.average_steps%64:raise ValueError("Invalid averaging window")
    dataset=args.dataset
    true_lid=DATASET_SPECS[dataset].true_lid
    train_raw,val_raw,_=make_splits(dataset,train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    ds=json.loads((args.source.parent/"dataset_manifest.json").read_text())
    assert _array_sha256(train_raw)==ds["splits"]["train"]["sha256"]
    assert _array_sha256(val_raw)==ds["splits"]["validation"]["sha256"]
    mean,rms=loaded.normalization_mean.cpu(),loaded.normalization_scale
    train=((torch.from_numpy(train_raw)-mean)/rms).to(args.device)
    validation=((torch.from_numpy(val_raw)-mean)/rms).to(args.device)
    fixture=np.load(args.query_arrays)
    queries=torch.from_numpy(fixture["queries"]).to(args.device)
    fields={name:copy.deepcopy(loaded.model).float() for name in ["original_lr","low_lr"]}
    lrs={"original_lr":cfg["learning_rate"],"low_lr":args.low_lr}
    opts={name:torch.optim.AdamW(field.parameters(),lr=lrs[name],weight_decay=cfg["weight_decay"])
          for name,field in fields.items()}
    for a,b in zip(fields["original_lr"].parameters(),fields["low_lr"].parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
    assert all(not opt.state for opt in opts.values())
    shuffle_rng=torch.Generator().manual_seed(args.seed+17)
    objective_rng={name:torch.Generator(device=args.device).manual_seed(args.seed+31) for name in fields}
    manifest={"source":str(args.source),"source_checkpoint_sha256":before,"source_best_epoch":loaded.best_epoch,
        "dataset":dataset,"true_lid":true_lid,
        "source_sha256":sha256(__file__),"original_loss_source_sha256":sha256("models/training.py"),
        "exact_trace_helper_sha256":sha256("experiments/fm_mechanism_fast_trace.py"),
        "query_arrays_sha256":sha256(args.query_arrays),"steps":args.steps,"average_steps":args.average_steps,
        "lrs":lrs,"reset":"BOTH AdamW states and RNGs fresh; source portable checkpoint has no optimizer state",
        "sampler":"Original CPU epoch randperm, all minibatches including short last; two equally seeded source-loss CUDA generators",
        "objective":"Unmodified models.training.diffusion_ve_dsm_loss, EDM weighting/preconditioning, sigma .01--16",
        "selection":"No best-checkpoint or LID selection: fixed last, weight average and sparse last-window function average",
        "snapshot_scales":[.01,.1,.316227766,1.],"snapshot_queries":64,"snapshot_interval":64,
        "snapshot_count":args.average_steps//64,"seed":args.seed}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save_checkpoint(args.output/"step_000000.pt",fields,opts,objective_rng,shuffle_rng,source,0)
    initial=diagnostics(fields["original_lr"],queries,SCALES)
    np.savez_compressed(args.output/"initial_arrays.npz",queries=queries.cpu().numpy(),query_ids=fixture["query_ids"],
        query_raw=fixture["query_raw"],scales=np.asarray(SCALES),**initial)
    for key in ["posterior","response","correction_original"]:
        np.testing.assert_allclose(initial[key],fixture[key],rtol=1e-9,atol=1e-9)
    average={name:{k:torch.zeros_like(v,dtype=torch.float64,device="cpu") for k,v in field.state_dict().items()}
        for name,field in fields.items()}
    snapshots={name:[] for name in fields};snapshot_steps=[];history=[]
    step=0;epoch=0;start=time.monotonic()
    while step<args.steps:
        epoch+=1;permutation=torch.randperm(len(train),generator=shuffle_rng)
        for offset in range(0,len(train),cfg["batch_size"]):
            if step>=args.steps:break
            step+=1;ids=permutation[offset:offset+cfg["batch_size"]].to(args.device);x=train[ids]
            row={"step":step,"epoch":epoch,"batch_count":len(x)}
            for name,field in fields.items():
                field.train();opts[name].zero_grad(set_to_none=True)
                loss=original_loss(field,x,cfg,objective_rng[name])
                if not torch.isfinite(loss):raise FloatingPointError(name)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(field.parameters(),cfg["gradient_clip_norm"])
                opts[name].step()
                row[name+"_loss"]=float(loss.detach());row[name+"_grad_norm"]=float(norm)
                if step>args.steps-args.average_steps:
                    for key,value in field.state_dict().items():
                        average[name][key].add_(value.detach().cpu().double(),alpha=1/args.average_steps)
            if step>args.steps-args.average_steps and (step-(args.steps-args.average_steps))%64==0:
                snapshot_steps.append(step)
                for name,field in fields.items():snapshots[name].append(diagnostics(field,queries[:64],manifest["snapshot_scales"]))
            if step%256==0 or step==args.steps:
                assert torch.equal(objective_rng["original_lr"].get_state(),objective_rng["low_lr"].get_state())
                history.append(row);print(json.dumps({"elapsed_seconds":time.monotonic()-start,**row}),flush=True)
                (args.output/"status.json").write_text(json.dumps({"status":"training","step":step,"elapsed_seconds":time.monotonic()-start})+"\n")
            if step in [args.steps//2,args.steps]:
                save_checkpoint(args.output/f"step_{step:06d}.pt",fields,opts,objective_rng,shuffle_rng,source,step)
    averages={name:{k:v.float() for k,v in state.items()} for name,state in average.items()}
    save_checkpoint(args.output/"final.pt",fields,opts,objective_rng,shuffle_rng,source,step,averages)
    rows=[]
    for name,field in fields.items():
        for mode in ["last","average"]:
            model=copy.deepcopy(field)
            if mode=="average":model.load_state_dict(averages[name])
            data=diagnostics(model,queries,SCALES)
            eval_rng=torch.Generator(device=args.device).manual_seed(args.seed+101)
            model.eval()
            with torch.no_grad():risk=float(original_loss(model,validation,cfg,eval_rng))
            np.savez_compressed(args.output/f"{name}_{mode}_arrays.npz",queries=queries.cpu().numpy(),
                query_ids=fixture["query_ids"],query_raw=fixture["query_raw"],scales=np.asarray(SCALES),**data)
            for i,s in enumerate(SCALES):
                rows.append({"arm":name,"mode":mode,"lambda":s,"response_mean":float(data["response"][i].mean()),
                    "correction_mean":float(data["correction_original"][i].mean()),
                    "response_d_mae":float(np.abs(data["response"][i]-true_lid).mean()),
                    "full_d_mae":float(np.abs(data["full"][i]-true_lid).mean()),"original_edm_validation_risk":risk})
    temporal_rows=[]
    for name,sequence in snapshots.items():
        data={key:np.stack([item[key] for item in sequence]) for key in sequence[0]}
        b=data["posterior"].mean(0);r=data["response"].mean(0)
        scales=np.asarray(manifest["snapshot_scales"])
        c=((b-queries[:64].cpu().numpy()[None])**2).sum(-1)/scales[:,None]**2
        variation=((data["posterior"]-b[None])**2).sum(-1).mean(0)/scales[:,None]**2
        np.testing.assert_allclose(data["correction_original"].mean(0),c+variation,rtol=1e-10,atol=1e-10)
        np.savez_compressed(args.output/f"{name}_temporal.npz",steps=np.asarray(snapshot_steps),scales=scales,
            query_ids=fixture["query_ids"][:64],mean_function_posterior=b,mean_function_response=r,
            mean_function_correction=c,mean_function_full=r+c,posterior_variation=variation,**data)
        for i,s in enumerate(scales):temporal_rows.append({"arm":name,"lambda":float(s),"snapshots":len(sequence),
            "mean_snapshot_correction":float(data["correction_original"][:,i].mean()),
            "mean_function_correction":float(c[i].mean()),"posterior_variation":float(variation[i].mean()),
            "mean_snapshot_full_d_mae":float(np.abs(data["full"][:,i]-true_lid).mean()),
            "mean_function_full_d_mae":float(np.abs(r[i]+c[i]-true_lid).mean())})
    for filename,data in [("metrics",rows),("temporal_metrics",temporal_rows),("history",history)]:
        (args.output/f"{filename}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (args.output/f"{filename}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    assert sha256(args.source)==before and sha256(__file__)==manifest["source_sha256"]
    (args.output/"status.json").write_text(json.dumps({"status":"complete","step":step,"elapsed_seconds":time.monotonic()-start,
        "source_checkpoint_unchanged":True,"actual_snapshots":len(snapshot_steps)})+"\n")
    return {"output":str(args.output),"elapsed_seconds":time.monotonic()-start}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["e8_spaghetti_pca","e6_exp_pca"],default="e8_spaghetti_pca")
    p.add_argument("--source",type=Path,default=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective/e8_spaghetti_pca/model.pt"))
    p.add_argument("--query-arrays",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=4096)
    p.add_argument("--average-steps",type=int,default=1024)
    p.add_argument("--low-lr",type=float,default=2e-5)
    p.add_argument("--seed",type=int,default=20261107)
    return p


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args())),flush=True)
