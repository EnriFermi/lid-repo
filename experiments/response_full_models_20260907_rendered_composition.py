"""Fixed two-pass transfer to two preserved rendered rectified posterior models.

Exact identity-basis VJP traces; no Hutchinson, full Jacobian archive, new
training, fit, iteration count or quality-selected checkpoint. D784 results
use each source's original normalization and are not pooled with D30.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import projected_trace,sha256
from experiments.response_full_models_20260907_clean_anchor import write_rows
from experiments.spiral_spaghetti_convergence import posterior,exact_trace
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


def exact_two_pass(field,queries,scale,batch_size=2):
    identity=torch.eye(queries.shape[-1],device=queries.device,dtype=queries.dtype)
    predict=lambda x:posterior(field,x,scale)
    b1,r1=projected_trace(predict,queries,identity,batch_size)
    b2,r2=projected_trace(lambda x:predict(predict(x)),queries,identity,batch_size)
    c1=((b1-queries)/scale).square().sum(-1);c2=((b2-queries)/scale).square().sum(-1)
    return {"base_posterior":b1,"posterior":b2,"base_response":r1,"response":r2,
        "base_correction":c1,"correction":c2,"base_full":r1+c1,"full":r2+c2}


def estimated_seconds(pilot_seconds,count=64,datasets=2,scales=9,pilot_count=2):
    return pilot_seconds*count/pilot_count*datasets*scales


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    protocol_path=args.source/"protocol.json";protocol=json.loads(protocol_path.read_text());cfg=protocol["config"]
    assert cfg["query_count"]==64 and cfg["average_steps"]==500 and cfg["learning_rate"]==.0002
    assert cfg["base_steps"]==8000 and cfg["continuation_steps"]==2000
    assert args.batch_size in (2,4)
    paths=[Path(__file__),protocol_path,Path("experiments/response_full_models_20260907_span.py"),
        Path("experiments/response_full_models_20260907_clean_anchor.py"),Path("experiments/spiral_spaghetti_convergence.py"),Path("models/neural_fields.py")]
    hashes={str(p):sha256(p) for p in paths};rows=[];records=[];gates=[]
    manifest={"source_hashes":hashes,"checkpoint_rule":"Both same_lr_average.pt, originalLR2e-4 final500 weight average; selected by historical protocol, not quality",
        "original_training":"8000 base plus2000 continuation updates; native rectified posterior, width256depth3D784",
        "query_rule":"Original test IDs0:64, normalized in float32 using each originalbase.pt mean/RMS, then promoted exactly tofloat64",
        "scales":cfg["scales"],"passes":2,"batch_size":args.batch_size,
        "trace":"Exact trJ via projected_trace with I784, full coordinate basis, ordinary autograd through actual composition",
        "precision":"Float64 model forward/trace and correction, storedfloat32weights promoted before loading; cachedfloat32 R/C replay tolerances fixed1e-4",
        "cached_value_limit":"Historical same_lr_average predictions containR/C/full but NOposteriorvectors; b is independently replayed against originalexact_trace backend, not fictitiousoldvectorcache",
        "normalization_scope":"Ancillary rendered/input-model transfer, distinct canonical normalization fromD30; do not pool scales or results",
        "maximum_wall_seconds":600,"records":records,"gates":gates,"completed":False}
    for dataset in ("e1_spiral_pca","e8_spaghetti_pca"):
        source=args.source/dataset;cp=source/"same_lr_average.pt";base_path=source/"base.pt";old_path=source/"same_lr_average_predictions.npz"
        data_path=args.source/"exact_subset/benchmarks"/dataset/"test/dataset.npy";label_path=data_path.with_name("lid.npy")
        for p in (cp,base_path,old_path,data_path,label_path):hashes[str(p)]=sha256(p)
        payload=torch.load(cp,map_location="cpu",weights_only=False);base=torch.load(base_path,map_location="cpu",weights_only=False)
        assert payload["averaged_updates"]==500 and payload["architecture"]["ambient_dim"]==784
        raw=np.load(data_path,mmap_mode="r");raw=np.array(raw[:64].reshape(64,-1),dtype=np.float32)
        q32=(torch.from_numpy(raw)-base["mean"])/base["rms"];q=q32.to(args.device).double()
        field=ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"])).to(device=args.device,dtype=torch.float64)
        field.load_state_dict(payload["model_state"],strict=True);field.eval()
        legacy=ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"])).to(args.device)
        legacy.load_state_dict(payload["model_state"],strict=True);legacy.eval();old=np.load(old_path)
        oracle=None
        if dataset=="e1_spiral_pca":
            hashes[str(args.spiral_oracle)]=sha256(args.spiral_oracle);oracle=np.load(args.spiral_oracle)
            np.testing.assert_array_equal(q.cpu().numpy(),oracle["queries"])
            np.testing.assert_array_equal(np.arange(64),oracle["query_ids"])
            np.testing.assert_array_equal(np.asarray(cfg["scales"]),oracle["scales"])
        record={"dataset":dataset,"checkpoint":str(cp),"averaged_updates":500,"architecture":payload["architecture"],
            "mean":base["mean"].numpy().tolist(),"rms":float(base["rms"]),"ground_truth_d":1,
            "stored_labels":np.unique(np.load(label_path)).tolist(),"oracle":"Matched rendered784Spiral reference" if oracle is not None else "No matched renderedSpaghetti reference available; noD30substitution"}
        records.append(record)
        (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        torch.cuda.synchronize(args.device)
        pilot_start=time.monotonic();pilot=exact_two_pass(field,q[:2],float(cfg["scales"][0]),args.batch_size)
        torch.cuda.synchronize(args.device);pilot_seconds=time.monotonic()-pilot_start
        estimate=estimated_seconds(pilot_seconds)
        independent_b,independent_r=exact_trace(field,q[:2],float(cfg["scales"][0]))
        torch.testing.assert_close(pilot["base_posterior"],independent_b,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(pilot["base_response"],independent_r,rtol=1e-10,atol=1e-10)
        gates.append({"dataset":dataset,"pilot_two_queries_seconds":pilot_seconds,"estimated_alltwo_laws_9scales_seconds":estimate,
            "old_chunked_backend_value_max":float((pilot["base_posterior"]-independent_b).abs().max()),
            "old_chunked_backend_trace_max":float((pilot["base_response"]-independent_r).abs().max())})
        if estimate>600:
            manifest["status"]="partial_cost_gate";(args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print(json.dumps(gates[-1]),flush=True);return
        accum={}
        for i,s in enumerate(cfg["scales"]):
            if time.monotonic()-started>600:
                manifest["status"]="partial_wall_gate";(args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");return
            start=time.monotonic();data=exact_two_pass(field,q,float(s),args.batch_size)
            data={k:v.cpu().numpy() for k,v in data.items()}
            for key,oldkey in (("base_response","test_response"),("base_correction","test_correction"),("base_full","test_full")):
                np.testing.assert_allclose(data[key],old[oldkey][i],rtol=1e-4,atol=1e-4)
            with torch.no_grad():legacy_b=posterior(legacy,q32.to(args.device),float(s)).double().cpu().numpy()
            np.testing.assert_allclose(data["base_posterior"],legacy_b,rtol=2e-5,atol=2e-6)
            gate={"dataset":dataset,"lambda":float(s),"legacy_float32_value_max":float(np.max(abs(data["base_posterior"]-legacy_b))),
                "legacy_float32_response_max":float(np.max(abs(data["base_response"]-old["test_response"][i]))),
                "legacy_float32_correction_max":float(np.max(abs(data["base_correction"]-old["test_correction"][i]))),"seconds":time.monotonic()-start}
            gates.append(gate)
            row={"dataset":dataset,"lambda":float(s),"count":64}
            for prefix,tag in (("base_","one_pass"),("","two_pass")):
                row[tag+"_response_mean"]=float(data[prefix+"response"].mean());row[tag+"_correction_mean"]=float(data[prefix+"correction"].mean())
                for readout in ("response","full"):
                    row[tag+"_"+readout+"_d_mae"]=float(abs(data[prefix+readout]-1).mean())
                    row[tag+"_"+readout+"_oracle_mae"]=float(abs(data[prefix+readout]-oracle[readout][i]).mean()) if oracle is not None else None
                row[tag+"_posterior_oracle_error_over_lambda2"]=float(np.square(data[prefix+"posterior"]-oracle["posterior"][i]).sum(-1).mean()/s**2) if oracle is not None else None
            rows.append(row)
            for k,v in data.items():accum.setdefault(k,[]).append(v)
            np.savez_compressed(args.output/f"{dataset}.npz",queries=q.cpu().numpy(),query_ids=np.arange(64),scales=np.asarray(cfg["scales"][:i+1]),
                normalization_mean=base["mean"].numpy(),normalization_rms=float(base["rms"]),**{k:np.stack(v) for k,v in accum.items()})
            print(json.dumps({**row,**gate}),flush=True)
            (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            write_rows(args.output,"summary",rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    manifest.update(completed=True,status="complete",all_sources_unchanged=True,elapsed_seconds=time.monotonic()-started)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/spiral_spaghetti_convergence_20260905/run"))
    p.add_argument("--spiral-oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracles_v2/rendered_spiral_arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=2)
    run(p.parse_args())
